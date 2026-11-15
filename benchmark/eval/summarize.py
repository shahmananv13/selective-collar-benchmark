"""Summary tables from the per-file CSVs written by evaluate.py.

    python summarize.py                      # all systems found in <results-dir>/per_file
    python summarize.py --systems nemotron vbx

Writes to <results-dir>/tables/:
  table5_summary.csv / .tex        mean +/- SD per system x dataset x condition (paper Table 5 format)
  table6_paired_tests.csv / .tex   Shapiro-Wilk on paired differences + Wilcoxon signed-rank
                                   (Selective vs Traditional, Selective vs No Collar; Table 6 format)
  forgiveness_behaviour.csv / .tex cross-system forgiveness behaviour
  system_ranking.csv               system ranking per condition and whether it changes
  timing.csv                       mean per-file scoring time per condition
  common_files.csv, dropped_files.csv   the common file set and every dropped file with its reason

file_set = "common": files for which EVERY included system has a non-empty hypothesis and defined
metrics in all three conditions (used for the LaTeX tables and all cross-system comparisons).
file_set = "all": every scored file of that system (CSV only).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (COMPONENTS, CONDITION_LABELS, CONDITIONS, DATASETS, DEFAULT_RESULTS_DIR, MAIN_METRICS,
                    PUBLISHED, PUBLISHED_NOTE, PUBLISHED_SYSTEM, TABLE_METRICS, dataset_label)

TEX_COND = {"no_collar": "No Collar (0.0\\,s)", "selective": "Selective Collar (0.25\\,s)",
            "traditional": "Traditional Collar (0.25\\,s)"}


# ------------------------------------------------------------------------- loading
def load_results(results_dir, systems=None, datasets=None):
    """Return (per-file dataframe, status dataframe) for the requested systems/datasets."""
    pf = Path(results_dir) / "per_file"
    frames, status = [], []
    for ds in (datasets or list(DATASETS)):
        for p in sorted(pf.glob(f"*_{ds}.csv")):
            system = p.name[: -len(f"_{ds}.csv")]
            if systems and system not in systems:
                continue
            df = pd.read_csv(p, dtype={"file_id": str})
            if len(df):
                frames.append(df)
            sp = pf / f"{system}_{ds}_status.csv"
            if sp.exists():
                status.append(pd.read_csv(sp, dtype={"file_id": str}))
    if not frames:
        raise SystemExit(f"no per-file results found in {pf}")
    df = pd.concat(frames, ignore_index=True)
    st = pd.concat(status, ignore_index=True) if status else pd.DataFrame(
        columns=["system", "dataset", "file_id", "status"])
    for col in ("trad_half_width_s", "selective_collar_s", "selective_variant", "der_backend"):
        if col in df and df[col].nunique(dropna=True) > 1:
            raise SystemExit(f"per-file results mix different '{col}' values {sorted(df[col].dropna().unique())}; "
                             "re-run evaluate.py for all systems with one configuration")
    return df, st


def common_sets(df, st):
    """Per dataset: the common file set, plus a record of every dropped file and why."""
    common, dropped = {}, []
    for ds, g in df.groupby("dataset"):
        systems = sorted(g["system"].unique())
        # a file is usable for a system when all main metrics are defined in all conditions
        ok = {}
        for s, gs in g.groupby("system"):
            piv = gs.pivot_table(index="file_id", columns="condition", values=TABLE_METRICS, dropna=False)
            full = piv.reindex(columns=pd.MultiIndex.from_product([TABLE_METRICS, CONDITIONS]))
            ok[s] = set(full.index[full.notna().all(axis=1)])
        scored = {s: set(g.loc[g["system"] == s, "file_id"]) for s in systems}
        common[ds] = sorted(set.intersection(*ok.values())) if ok else []
        stat = {(r.system, r.file_id): r.status for r in st[st["dataset"] == ds].itertuples()}
        universe = set(st.loc[(st["dataset"] == ds) & (st["status"] != "excluded_by_protocol"), "file_id"]) \
            | set(g["file_id"])
        for fid in sorted(universe - set(common[ds])):
            why = []
            for s in systems:
                if fid in ok[s]:
                    continue
                if fid in scored[s]:
                    why.append(f"{s}: metric undefined in at least one condition")
                else:
                    why.append(f"{s}: {stat.get((s, fid), 'not scored')}")
            dropped.append({"dataset": ds, "file_id": fid, "reason": "; ".join(why)})
        for r in st[(st["dataset"] == ds) & (st["status"] == "excluded_by_protocol")] \
                .drop_duplicates("file_id").itertuples():
            dropped.append({"dataset": ds, "file_id": r.file_id, "reason": "excluded by protocol"})
    return common, pd.DataFrame(dropped, columns=["dataset", "file_id", "reason"])


def restrict(df, common):
    keep = pd.Series(False, index=df.index)
    for ds, ids in common.items():
        keep |= (df["dataset"] == ds) & df["file_id"].isin(ids)
    return df[keep]


# ------------------------------------------------------------------------- table 5
def table5(df, file_set):
    rows = []
    for (s, ds), g in df.groupby(["system", "dataset"]):
        for c in CONDITIONS:
            gc = g[g["condition"] == c]
            row = {"file_set": file_set, "system": s, "dataset": ds, "condition": c, "n_files": len(gc)}
            for m in TABLE_METRICS:
                row[f"{m}_mean"] = gc[m].mean()
                row[f"{m}_sd"] = gc[m].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def published_table5():
    rows = []
    for ds, d in PUBLISHED.items():
        for c in CONDITIONS:
            row = {"file_set": "published", "system": PUBLISHED_SYSTEM, "dataset": ds, "condition": c,
                   "n_files": d["n_files"]}
            for m in TABLE_METRICS:
                mean, sd = d[c][m]
                row[f"{m}_mean"] = mean
                row[f"{m}_sd"] = sd if sd is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def fmt_pm(mean, sd):
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f}" if pd.isna(sd) else f"{mean:.3f} $\\pm$ {sd:.3f}"


def table5_tex(t5, meta_note):
    lines = [
        "% Generated by benchmark/eval/summarize.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{Summary statistics (mean $\\pm$ SD) of diarization metrics by system, dataset and "
        "evaluation condition.}",
        "\\label{tab:multimodel_summary}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{|l|l|l|c|c|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{System} & \\textbf{Condition} & \\textbf{DER} & \\textbf{JER} & "
        "\\textbf{CDER} & \\textbf{SER} & \\textbf{BER} & \\textbf{MS} & \\textbf{FA} & \\textbf{SC} \\\\",
        "\\hline",
    ]
    for ds in [d for d in DATASETS if d in set(t5["dataset"])]:
        g = t5[t5["dataset"] == ds]
        systems = [s for s in g["system"].unique() if s != PUBLISHED_SYSTEM] + \
                  ([PUBLISHED_SYSTEM] if PUBLISHED_SYSTEM in set(g["system"]) else [])
        first_ds = True
        for s in systems:
            gs = g[g["system"] == s].set_index("condition")
            n = int(gs["n_files"].iloc[0])
            name = "Pyannote 3.1$^{\\dagger}$" if s == PUBLISHED_SYSTEM else tex_escape(s)
            for i, c in enumerate(CONDITIONS):
                r = gs.loc[c]
                cells = [fmt_pm(r[f"{m}_mean"], r[f"{m}_sd"]) for m in MAIN_METRICS] + \
                        [fmt_pm(r[f"{m}_mean"], np.nan) for m in COMPONENTS]
                dcell = f"\\multirow{{{3 * len(systems)}}}{{*}}{{{dataset_label(ds)}}}" if first_ds else ""
                scell = f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{{name}\\\\($N={n}$)}}}}" if i == 0 else ""
                first_ds = False
                lines.append(" & ".join([dcell, scell, TEX_COND[c]] + cells) + " \\\\")
            lines.append("\\cline{2-11}" if s != systems[-1] else "\\hline")
    lines += [
        "\\end{tabular}",
        "\\\\[2pt]",
        "\\parbox{0.95\\textwidth}{\\footnotesize $^{\\dagger}$Values reproduced from Table~5 of the "
        "original Selective Collar study (Pyannote~3.1; MSDWild $N=488$, VoxConverse $N=232$); not "
        "re-computed here. " + meta_note + "}",
        "\\end{table*}",
        "% requires \\usepackage{multirow}",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- table 6
def stars(p):
    if pd.isna(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def paired_tests(df, file_set, alpha=0.05):
    comps = [("selective", "traditional"), ("selective", "no_collar")]
    rows = []
    for (s, ds), g in df.groupby(["system", "dataset"]):
        for a, b in comps:
            for m in TABLE_METRICS:
                piv = g.pivot_table(index="file_id", columns="condition", values=m, dropna=False)
                if a not in piv or b not in piv:
                    continue
                piv = piv[[a, b]].dropna()
                x, y = piv[a].to_numpy(float), piv[b].to_numpy(float)
                d = x - y
                n = len(d)
                row = {"file_set": file_set, "system": s, "dataset": ds,
                       "comparison": f"{a}_vs_{b}", "metric": m, "n_files": n,
                       "n_nonzero_diff": int(np.count_nonzero(d)),
                       f"mean_{a}": x.mean() if n else np.nan, "mean_other": y.mean() if n else np.nan,
                       "mean_diff": d.mean() if n else np.nan, "median_diff": np.median(d) if n else np.nan,
                       "shapiro_W": np.nan, "shapiro_p": np.nan, "normal": "", "test_used": "Wilcoxon",
                       "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "paired_t_p": np.nan}
                row["mean_first"] = row.pop(f"mean_{a}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if n >= 3 and np.ptp(d) > 0:
                        row["shapiro_W"], row["shapiro_p"] = stats.shapiro(d)
                        row["normal"] = "Yes" if row["shapiro_p"] >= alpha else "No"
                    if n >= 1 and np.count_nonzero(d) > 0:
                        w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                        row["wilcoxon_stat"], row["wilcoxon_p"] = w.statistic, w.pvalue
                        if n >= 2:
                            row["paired_t_p"] = stats.ttest_rel(x, y).pvalue
                    elif n >= 1:
                        row["wilcoxon_p"] = 1.0     # all paired differences are exactly zero
                row["significance"] = stars(row["wilcoxon_p"])
                row["direction"] = "up" if row["mean_diff"] > 0 else "down" if row["mean_diff"] < 0 else "equal"
                rows.append(row)
    cols = ["file_set", "system", "dataset", "comparison", "metric", "n_files", "n_nonzero_diff",
            "shapiro_W", "shapiro_p", "normal", "test_used", "wilcoxon_stat", "wilcoxon_p", "significance",
            "direction", "mean_first", "mean_other", "mean_diff", "median_diff", "paired_t_p"]
    return pd.DataFrame(rows, columns=cols)


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "$< 0.001$"
    return f"{p:.4f}"


def fmt_shapiro(p):
    if pd.isna(p):
        return "n/a"
    return "$\\approx 0$" if p < 1e-4 else f"{p:.4f}"


def table6_tex(t6, comparison, other_label, label):
    g = t6[(t6["comparison"] == comparison) & t6["metric"].isin(MAIN_METRICS)]
    lines = [
        "% Generated by benchmark/eval/summarize.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        f"\\caption{{Detailed paired hypothesis tests (Wilcoxon signed-rank): Selective vs. {other_label} "
        "Collar, per system.}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{|l|l|l|c|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{System} & \\textbf{Metric} & \\textbf{Shapiro-Wilk $p$} & "
        "\\textbf{Normal?} & \\textbf{Test Used} & \\textbf{$p$-value} & "
        "\\textbf{Significance ($\\alpha=0.05$)} & \\textbf{Direction} & "
        f"\\textbf{{Selective Mean vs {other_label} Mean}} \\\\",
        "\\hline",
    ]
    arrow = {"up": "$\\uparrow$", "down": "$\\downarrow$", "equal": "$=$"}
    for ds in [d for d in DATASETS if d in set(g["dataset"])]:
        gd = g[g["dataset"] == ds]
        systems = list(gd["system"].unique())
        first_ds = True
        for s in systems:
            gs = gd[gd["system"] == s].set_index("metric")
            for i, m in enumerate([m for m in MAIN_METRICS if m in gs.index]):
                r = gs.loc[m]
                dcell = f"\\multirow{{{5 * len(systems)}}}{{*}}{{{dataset_label(ds)}}}" if first_ds else ""
                scell = (f"\\multirow{{5}}{{*}}{{\\shortstack[l]{{{tex_escape(s)}\\\\($N={int(r['n_files'])}$)}}}}"
                         if i == 0 else "")
                first_ds = False
                lines.append(" & ".join([
                    dcell, scell, m, fmt_shapiro(r["shapiro_p"]), r["normal"] or "n/a", r["test_used"],
                    fmt_p(r["wilcoxon_p"]), r["significance"], arrow[r["direction"]],
                    f"{r['mean_first']:.4f} vs {r['mean_other']:.4f}"]) + " \\\\")
            lines.append("\\cline{2-10}" if s != systems[-1] else "\\hline")
    lines += ["\\end{tabular}", "\\end{table*}", "% requires \\usepackage{multirow}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- forgiveness behaviour
def forgiveness(df, file_set):
    rows = []
    for (s, ds), g in df.groupby(["system", "dataset"]):
        nc, sel, tr = (g[g["condition"] == c].set_index("file_id") for c in CONDITIONS)
        tot_b = sel["total_boundaries"].sum()
        snapped = (sel["altered_starts"] + sel["altered_ends"]).sum()
        per_file_frac = ((sel["altered_starts"] + sel["altered_ends"]) / sel["total_boundaries"])
        rows.append({
            "file_set": file_set, "system": s, "dataset": ds, "n_files": len(nc),
            "ref_boundaries_total": int(tot_b), "boundaries_snapped_selective": int(snapped),
            "frac_boundaries_snapped_pooled": snapped / tot_b if tot_b else np.nan,
            "frac_boundaries_snapped_mean_per_file": per_file_frac.mean(),
            "frac_starts_snapped_pooled": sel["altered_starts"].sum() / (tot_b / 2) if tot_b else np.nan,
            "frac_ends_snapped_pooled": sel["altered_ends"].sum() / (tot_b / 2) if tot_b else np.nan,
            "frac_boundaries_forgiven_traditional": 1.0,
            "trad_ref_removed_s_mean": tr["ref_removed_s"].mean(),
            "trad_ref_removed_s_max": tr["ref_removed_s"].max(),
            "trad_ref_removed_pct_mean": tr["ref_removed_pct"].mean(),
            "trad_ref_removed_pct_pooled": 100 * tr["ref_removed_s"].sum() / tr["ref_speech_total_s"].sum(),
            "trad_ref_segments_erased_total": int(tr["n_ref_segments_erased"].sum()),
            "trad_ref_segments_erased_frac": tr["n_ref_segments_erased"].sum() / tr["n_ref_segments"].sum(),
            "selective_ref_removed_s_mean": sel["ref_removed_s"].mean(),
            "DER_no_collar": nc["DER"].mean(), "DER_selective": sel["DER"].mean(),
            "DER_traditional": tr["DER"].mean(),
            "DER_gap_no_collar_minus_traditional": nc["DER"].mean() - tr["DER"].mean(),
            "DER_gap_no_collar_minus_selective": nc["DER"].mean() - sel["DER"].mean(),
            "DER_rel_drop_traditional_pct": 100 * (nc["DER"].mean() - tr["DER"].mean()) / nc["DER"].mean(),
            "DER_rel_drop_selective_pct": 100 * (nc["DER"].mean() - sel["DER"].mean()) / nc["DER"].mean(),
        })
    out = pd.DataFrame(rows)
    # ranks per dataset and condition (1 = lowest mean DER)
    for c in CONDITIONS:
        out[f"DER_rank_{c}"] = out.groupby("dataset")[f"DER_{c}"].rank(method="min").astype("Int64")
    rank_cols = [f"DER_rank_{c}" for c in CONDITIONS]
    changes = out.groupby("dataset")[rank_cols].apply(
        lambda g: bool((g.nunique(axis=1) > 1).any())).to_dict()
    out["ranking_changes_between_conditions"] = out["dataset"].map(changes)
    return out


def published_forgiveness():
    rows = []
    for ds, d in PUBLISHED.items():
        nc, sel, tr = (d[c]["DER"][0] for c in CONDITIONS)
        rows.append({"file_set": "published", "system": PUBLISHED_SYSTEM, "dataset": ds,
                     "n_files": d["n_files"], "DER_no_collar": nc, "DER_selective": sel, "DER_traditional": tr,
                     "DER_gap_no_collar_minus_traditional": nc - tr, "DER_gap_no_collar_minus_selective": nc - sel,
                     "DER_rel_drop_traditional_pct": 100 * (nc - tr) / nc,
                     "DER_rel_drop_selective_pct": 100 * (nc - sel) / nc})
    return pd.DataFrame(rows)


def ranking_table(fg, with_published):
    rows = []
    src = pd.concat([fg, published_forgiveness()], ignore_index=True) if with_published else fg
    for ds, g in src.groupby("dataset"):
        orders = {}
        for c in CONDITIONS:
            gg = g.sort_values(f"DER_{c}")
            orders[c] = list(gg["system"])
            rows.append({"dataset": ds, "includes_published_pyannote": with_published, "condition": c,
                         "ranking_best_to_worst": " < ".join(
                             f"{r.system} ({getattr(r, 'DER_' + c):.3f})" for r in gg.itertuples())})
        changed = len({tuple(o) for o in orders.values()}) > 1
        for r in rows[-3:]:
            r["ranking_changes_between_conditions"] = changed
    return pd.DataFrame(rows)


def forgiveness_tex(fg, half_width):
    lines = [
        "% Generated by benchmark/eval/summarize.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{Forgiveness behaviour across systems: share of reference boundaries snapped by the "
        "Selective Collar, reference speech removed by the Traditional Collar "
        f"($\\pm${half_width:g}\\,s), and the resulting DER gaps.}}",
        "\\label{tab:forgiveness_behaviour}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{System} & \\textbf{$N$} & "
        "\\shortstack{\\textbf{Boundaries}\\\\\\textbf{snapped (Sel.) [\\%]}} & "
        "\\shortstack{\\textbf{Ref.\\ removed}\\\\\\textbf{(Trad.) [s/file]}} & "
        "\\shortstack{\\textbf{Ref.\\ removed}\\\\\\textbf{(Trad.) [\\%]}} & "
        "\\shortstack{\\textbf{Ref.\\ segs}\\\\\\textbf{erased (Trad.) [\\%]}} & "
        "\\shortstack{\\textbf{DER}\\\\\\textbf{No Collar}} & "
        "\\shortstack{\\textbf{$\\Delta$DER}\\\\\\textbf{NC$-$Sel.}} & "
        "\\shortstack{\\textbf{$\\Delta$DER}\\\\\\textbf{NC$-$Trad.}} & "
        "\\shortstack{\\textbf{DER rank}\\\\\\textbf{NC / Sel. / Trad.}} \\\\",
        "\\hline",
    ]
    for ds in [d for d in DATASETS if d in set(fg["dataset"])]:
        g = fg[fg["dataset"] == ds]
        for i, r in enumerate(g.itertuples()):
            dcell = f"\\multirow{{{len(g)}}}{{*}}{{{dataset_label(ds)}}}" if i == 0 else ""
            lines.append(" & ".join([
                dcell, tex_escape(r.system), str(r.n_files),
                f"{100 * r.frac_boundaries_snapped_pooled:.1f}",
                f"{r.trad_ref_removed_s_mean:.2f}", f"{r.trad_ref_removed_pct_mean:.1f}",
                f"{100 * r.trad_ref_segments_erased_frac:.1f}",
                f"{r.DER_no_collar:.3f}", f"{r.DER_gap_no_collar_minus_selective:.3f}",
                f"{r.DER_gap_no_collar_minus_traditional:.3f}",
                f"{r.DER_rank_no_collar} / {r.DER_rank_selective} / {r.DER_rank_traditional}"]) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table*}", "% requires \\usepackage{multirow}"]
    return "\n".join(lines) + "\n"


def timing(df):
    g = df.groupby(["system", "dataset", "condition"], observed=True)
    t = g.agg(n_files=("file_id", "count"), time_core_ms_mean=("time_core_s", lambda x: 1000 * x.mean()),
              time_total_ms_mean=("time_total_s", lambda x: 1000 * x.mean())).reset_index()
    t["condition"] = pd.Categorical(t["condition"], CONDITIONS, ordered=True)
    return t.sort_values(["system", "dataset", "condition"])


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--systems", nargs="+", default=None, help="default: every system with per-file results")
    ap.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS))
    args = ap.parse_args(argv)

    df, st = load_results(args.results_dir, args.systems, args.datasets)
    out = Path(args.results_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    half_width = float(df["trad_half_width_s"].dropna().iloc[0]) if "trad_half_width_s" in df else 0.25

    TEX_COND["traditional"] = f"Traditional Collar ($\pm${half_width:g}\,s)"
    common, dropped = common_sets(df, st)
    dfc = restrict(df, common)
    pd.DataFrame([{"dataset": ds, "file_id": f} for ds, ids in common.items() for f in ids]) \
        .to_csv(out / "common_files.csv", index=False)
    dropped.to_csv(out / "dropped_files.csv", index=False)

    print("Systems per dataset and file sets")
    for ds in common:
        systems = sorted(df.loc[df["dataset"] == ds, "system"].unique())
        per = {s: df[(df["dataset"] == ds) & (df["system"] == s)]["file_id"].nunique() for s in systems}
        nd = (dropped["dataset"] == ds).sum()
        print(f"  {ds}: systems={per}  common set={len(common[ds])} files, dropped={nd}")
    if dfc.empty:
        print("WARNING: the common file set is empty; LaTeX tables fall back to file_set='all'.")

    meta_note = (f"Traditional Collar: $\\pm${half_width:g}\\,s around every reference boundary; "
                 "MS/FA/SC are per-file rates relative to scored reference speech.")
    main_df, main_set = (dfc, "common") if not dfc.empty else (df, "all")

    # Table 5
    t5 = pd.concat([table5(dfc, "common") if not dfc.empty else None, table5(df, "all"), published_table5()],
                   ignore_index=True)
    t5.insert(len(t5.columns), "note", np.where(t5["file_set"] == "published", PUBLISHED_NOTE, ""))
    t5.to_csv(out / "table5_summary.csv", index=False, float_format="%.6g")
    t5_main = pd.concat([t5[t5["file_set"] == main_set], t5[t5["file_set"] == "published"]])
    (out / "table5_summary.tex").write_text(table5_tex(t5_main, meta_note), encoding="utf-8")

    # Table 6
    t6 = pd.concat([paired_tests(dfc, "common") if not dfc.empty else None, paired_tests(df, "all")],
                   ignore_index=True)
    t6.to_csv(out / "table6_paired_tests.csv", index=False, float_format="%.6g")
    t6_main = t6[t6["file_set"] == main_set]
    (out / "table6_paired_tests.tex").write_text(
        table6_tex(t6_main, "selective_vs_traditional", "Traditional", "tab:tests_sel_vs_trad")
        + "\n" + table6_tex(t6_main, "selective_vs_no_collar", "No", "tab:tests_sel_vs_nocollar"),
        encoding="utf-8")

    # Forgiveness behaviour + ranking
    fg_main = forgiveness(main_df, main_set)
    fg = pd.concat([fg_main, forgiveness(df, "all") if main_set == "common" else None, published_forgiveness()],
                   ignore_index=True)
    fg.insert(len(fg.columns), "note", np.where(fg["file_set"] == "published", PUBLISHED_NOTE, ""))
    fg.to_csv(out / "forgiveness_behaviour.csv", index=False, float_format="%.6g")
    (out / "forgiveness_behaviour.tex").write_text(forgiveness_tex(fg_main, half_width), encoding="utf-8")
    rk = pd.concat([ranking_table(fg_main, False), ranking_table(fg_main, True)], ignore_index=True)
    rk.insert(1, "file_set", main_set)
    rk.to_csv(out / "system_ranking.csv", index=False)

    timing(df).to_csv(out / "timing.csv", index=False, float_format="%.4g")

    # console digest
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    show = t5_main[["system", "dataset", "condition", "n_files"] + [f"{m}_mean" for m in TABLE_METRICS]]
    print(f"\nTable 5 (file_set={main_set} + published), means:")
    print(show.round(3).to_string(index=False))
    print(f"\nForgiveness behaviour (file_set={main_set}):")
    print(fg_main[["system", "dataset", "n_files", "frac_boundaries_snapped_pooled", "trad_ref_removed_s_mean",
                   "trad_ref_removed_pct_mean", "DER_gap_no_collar_minus_selective",
                   "DER_gap_no_collar_minus_traditional", "ranking_changes_between_conditions"]]
          .round(3).to_string(index=False))
    sig = t6_main[(t6_main["comparison"] == "selective_vs_traditional") & t6_main["metric"].isin(MAIN_METRICS)]
    print("\nSelective vs Traditional (Wilcoxon):")
    print(sig[["system", "dataset", "metric", "n_files", "shapiro_p", "wilcoxon_p", "significance", "direction",
               "mean_first", "mean_other"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    (out / "tables_info.json").write_text(json.dumps({
        "file_set_used_for_latex": main_set, "trad_half_width_s": half_width,
        "common_set_sizes": {ds: len(v) for ds, v in common.items()},
        "sd": "sample standard deviation (ddof=1) over files", "published_rows": PUBLISHED_NOTE}, indent=2),
        encoding="utf-8")
    print(f"\ntables -> {out}")


if __name__ == "__main__":
    main()
