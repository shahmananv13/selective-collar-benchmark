"""Paper-ready figures (vector PDF + 600 dpi PNG) from the per-file CSVs.

    python make_figures.py                       # all systems found in <results-dir>/per_file
    python make_figures.py --systems nemotron vbx --labels nemotron="Sortformer" vbx=VBx

Writes to <results-dir>/figures/:
  fig_boxplots_<dataset>.{pdf,png}     (a) 5 metrics x 3 conditions, one row per system   [Fig. 7 style]
  fig_boxplots_all.{pdf,png}               same, all datasets stacked
  fig_der_decomposition.{pdf,png}      (b) stacked MS/FA/SC per system x condition        [Fig. 8 style]
  fig_intervention_density.{pdf,png}   (c) boundaries forgiven vs files sorted by #segments [Fig. 3 style]
  fig_data_ablation.{pdf,png}          (d) reference seconds removed by the collar         [Fig. 4 style]
  fig_der_by_system.{pdf,png}          (e) compact mean DER by system and condition

IEEE widths: single column 3.5 in, double column 7.16 in; serif 8 pt.
Condition colours follow the published paper: No Collar blue, Selective orange, Traditional green.
Cross-system panels use the common file set (see summarize.py); the published Pyannote 3.1 bars are
aggregate values copied from the paper (hatched) because per-file predictions are unavailable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from common import (COMPONENTS, CONDITION_COLORS, CONDITION_LABELS, CONDITIONS, DATASETS,  # noqa: E402
                    DEFAULT_RESULTS_DIR, MAIN_METRICS, PUBLISHED, PUBLISHED_SYSTEM, dataset_label)
from summarize import common_sets, load_results, restrict  # noqa: E402

W1, W2 = 3.5, 7.16
INK, MUTED, GRID = "#222222", "#555555", "#d9d9d9"
DISPLAY = {"vbx": "VBx", PUBLISHED_SYSTEM: "Pyannote 3.1\n(published)"}
# component encoding inside a condition hue: full colour -> lighter tints
COMP_TINT = {"MS": 1.0, "FA": 0.62, "SC": 0.30}
COMP_LABEL = {"MS": "Missed Speech (MS)", "FA": "False Alarm (FA)", "SC": "Speaker Confusion (SC)"}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "lines.linewidth": 0.9,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.columnspacing": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "hatch.linewidth": 0.4,
    })


def tint(color, t):
    r, g, b = to_rgb(color)
    return (1 - t * (1 - r), 1 - t * (1 - g), 1 - t * (1 - b))


def name(system):
    return DISPLAY.get(system, system)


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


def cond_legend(fig, y=0.0, extra=(), ncol=None):
    short = {"no_collar": "NC", "selective": "Sel", "traditional": "Trad"}
    handles = [Patch(facecolor=CONDITION_COLORS[c], edgecolor="none", label=f"{short[c]}: {CONDITION_LABELS[c]}")
               for c in CONDITIONS] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, y))


def trad_label(hw):
    return f"Traditional Collar ($\\pm${hw:g} s)"


# ------------------------------------------------------------------ (a) boxplots
def fig_boxplots(df, rows, out, stem):
    """rows = [(dataset, system)]; 5 metric columns, 3 condition boxes per panel."""
    n = len(rows)
    fig, axes = plt.subplots(n, len(MAIN_METRICS), figsize=(W2, 1.3 * n + 0.45), squeeze=False)
    for i, (ds, s) in enumerate(rows):
        g = df[(df["dataset"] == ds) & (df["system"] == s)]
        nfiles = g["file_id"].nunique()
        for j, m in enumerate(MAIN_METRICS):
            ax = axes[i, j]
            data = [g.loc[g["condition"] == c, m].dropna().to_numpy() for c in CONDITIONS]
            bp = ax.boxplot(data, positions=range(len(CONDITIONS)), widths=0.62, patch_artist=True,
                            showfliers=True, medianprops=dict(color=INK, linewidth=0.9),
                            whiskerprops=dict(color=MUTED, linewidth=0.5), capprops=dict(color=MUTED, linewidth=0.5),
                            boxprops=dict(linewidth=0.5, edgecolor=MUTED),
                            flierprops=dict(marker="o", markersize=1.2, markerfacecolor=MUTED,
                                            markeredgecolor="none", alpha=0.45))
            for patch, c in zip(bp["boxes"], CONDITIONS):
                patch.set_facecolor(CONDITION_COLORS[c])
                patch.set_alpha(0.85)
            # short position labels: identity is never carried by colour alone (orange/green are
            # hard to tell apart for red-green colour-blind readers)
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels(["NC", "Sel", "Trad"], fontsize=6)
            ax.tick_params(axis="x", length=0, pad=1)
            # axis limit from whiskers / 97.5th percentile so a few extreme files (CDER can be >> 1)
            # do not flatten the boxes; the number of points beyond the axis is printed
            hi = [max(np.percentile(d, 97.5), w.get_ydata()[1]) for d, w in zip(data, bp["whiskers"][1::2]) if len(d)]
            top = 1.08 * max(hi) if hi and max(hi) > 0 else 1.0
            ax.set_ylim(0, top)
            n_off = int(sum((d > top).sum() for d in data))
            if n_off:
                ax.text(0.98, 0.98, f"{n_off} pts > axis", transform=ax.transAxes, ha="right", va="top",
                        fontsize=5, color=MUTED)
            ax.margins(x=0.08)
            if i == 0:
                ax.set_title(m, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{dataset_label(ds)}\n{name(s)} (N={nfiles})", fontsize=7)
    fig.tight_layout(rect=(0, 0.35 / (1.3 * n + 0.45), 1, 1), h_pad=0.6, w_pad=0.5)
    cond_legend(fig)
    save(fig, out, stem)


# ----------------------------------------------------- (b) stacked DER decomposition
def published_means(ds):
    return {c: {m: PUBLISHED[ds][c][m][0] for m in COMPONENTS + ["DER"]} for c in CONDITIONS}


def fig_decomposition(df, datasets, systems, out, with_published=True):
    fig, axes = plt.subplots(1, len(datasets), figsize=(W2, 2.45), squeeze=False)
    bw = 0.26
    for ax, ds in zip(axes[0], datasets):
        groups = [s for s in systems if ((df["dataset"] == ds) & (df["system"] == s)).any()]
        entries = []
        for s in groups:
            g = df[(df["dataset"] == ds) & (df["system"] == s)]
            entries.append((s, {c: g[g["condition"] == c][COMPONENTS + ["DER"]].mean().to_dict()
                                for c in CONDITIONS}, g["file_id"].nunique(), False))
        if with_published and ds in PUBLISHED:
            entries.append((PUBLISHED_SYSTEM, published_means(ds), PUBLISHED[ds]["n_files"], True))
        top = 0
        for gi, (s, means, nfiles, pub) in enumerate(entries):
            for ci, c in enumerate(CONDITIONS):
                x = gi + (ci - 1) * (bw + 0.02)
                bottom = 0.0
                for comp in COMPONENTS:
                    v = means[c][comp]
                    ax.bar(x, v, bw, bottom=bottom, color=tint(CONDITION_COLORS[c], COMP_TINT[comp]),
                           edgecolor="white", linewidth=0.5, hatch="////" if pub else None)
                    bottom += v
                total = means[c]["DER"] if pub else bottom      # published DER (components are rounded)
                ax.text(x, bottom + 0.004, f"{total:.3f}", ha="center", va="bottom", fontsize=5.5,
                        rotation=90, color=INK)
                top = max(top, bottom)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([f"{name(s)}\nN={n}" for s, _, n, _ in entries])
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        ax.set_title(dataset_label(ds), fontweight="bold")
        ax.tick_params(axis="x", length=0)
    axes[0, 0].set_ylabel("Mean error rate")
    comp_handles = [Patch(facecolor=tint("#444444", COMP_TINT[k]), edgecolor="white", label=COMP_LABEL[k])
                    for k in COMPONENTS]
    cond_handles = [Patch(facecolor=CONDITION_COLORS[c], edgecolor="none", label=CONDITION_LABELS[c])
                    for c in CONDITIONS]
    handles = [h for pair in zip(cond_handles, comp_handles) for h in pair]   # column-major: row 1 = conditions
    if with_published:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="////",
                             label="published aggregate (see caption)"))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.0)
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig_der_decomposition")


# ------------------------------------------------------- (c) intervention density
def fig_intervention(df, datasets, systems, out):
    nrow, ncol = len(datasets), len(systems)
    width = W2 if ncol > 1 else W1
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 1.55 * nrow + 0.4), squeeze=False, sharey="row")
    for i, ds in enumerate(datasets):
        for j, s in enumerate(systems):
            ax = axes[i, j]
            g = df[(df["dataset"] == ds) & (df["system"] == s) & (df["condition"] == "selective")]
            g = g.sort_values(["n_ref_segments", "file_id"])
            x = np.arange(len(g))
            if len(g):
                ax.plot(x, np.zeros(len(g)), color=CONDITION_COLORS["no_collar"])
                ax.plot(x, g["total_boundaries"], color=CONDITION_COLORS["traditional"])
                ax.plot(x, g["altered_starts"] + g["altered_ends"], color=CONDITION_COLORS["selective"],
                        linewidth=0.6)
                frac = (g["altered_starts"] + g["altered_ends"]).sum() / max(g["total_boundaries"].sum(), 1)
                ax.text(0.03, 0.95, f"snapped: {100 * frac:.1f}% of boundaries", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6.5, color=INK)
                ax.annotate("Trad", (x[-1], g["total_boundaries"].iloc[-1]), xytext=(2, 0),
                            textcoords="offset points", fontsize=6, color=INK, va="center", annotation_clip=False)
                ax.annotate("Sel", (x[-1], (g["altered_starts"] + g["altered_ends"]).iloc[-1]), xytext=(2, 0),
                            textcoords="offset points", fontsize=6, color=INK, va="center", annotation_clip=False)
            if i == 0:
                ax.set_title(name(s).replace("\n", " "), fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{dataset_label(ds)}\nboundaries forgiven", fontsize=7)
            if i == nrow - 1:
                ax.set_xlabel("Audio files (sorted by no. of reference segments)", fontsize=7)
            ax.set_ylim(bottom=0)
            ax.margins(x=0.01)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.tight_layout(rect=(0, 0.3 / (1.55 * nrow + 0.4), 1, 1), h_pad=0.6, w_pad=0.6)
    handles = [plt.Line2D([], [], color=CONDITION_COLORS[c], label=lab) for c, lab in (
        ("no_collar", "No Collar (none)"), ("selective", "Selective Collar (boundaries snapped)"),
        ("traditional", "Traditional Collar (every boundary)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3 if ncol > 1 else 1, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig_intervention_density")


# ------------------------------------------------------------- (d) data ablation
def fig_ablation(df, datasets, out, hw):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(W1, 1.5 * len(datasets) + 0.55), squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        g = df[(df["dataset"] == ds) & (df["condition"] == "traditional")].drop_duplicates("file_id")
        g = g.sort_values(["ref_overlap_s", "file_id"])
        x = np.arange(len(g))
        ax.plot(x, g["ref_removed_s"], color=CONDITION_COLORS["traditional"], linewidth=0.6)
        ax.plot(x, np.zeros(len(g)), color=CONDITION_COLORS["no_collar"], linewidth=2.2, clip_on=False, zorder=4)
        ax.plot(x, np.zeros(len(g)), color=CONDITION_COLORS["selective"], linewidth=0.9, clip_on=False, zorder=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        if len(g):
            ax.text(0.03, 0.95, f"{dataset_label(ds)} (N={len(g)}): mean {g['ref_removed_s'].mean():.1f} s "
                    f"= {g['ref_removed_pct'].mean():.1f}% of reference speech per file",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
        ax.set_ylabel("Reference speech\nremoved (s)", fontsize=7)
        ax.set_ylim(bottom=0, top=(g["ref_removed_s"].max() * 1.25 if len(g) else 1))
        ax.margins(x=0.01)
    axes[-1, 0].set_xlabel("Audio files (sorted by reference overlap duration)", fontsize=7)
    fig.tight_layout(rect=(0, 0.36 / (1.5 * len(datasets) + 0.55), 1, 1), h_pad=0.5)
    handles = [plt.Line2D([], [], color=CONDITION_COLORS["no_collar"], label="No Collar (0 s)"),
               plt.Line2D([], [], color=CONDITION_COLORS["selective"], label="Selective Collar (0 s)"),
               plt.Line2D([], [], color=CONDITION_COLORS["traditional"], label=trad_label(hw))]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig_data_ablation")


# ------------------------------------------------------------ (e) DER by system
def fig_der_by_system(df, datasets, systems, out, with_published=True):
    fig, axes = plt.subplots(1, len(datasets), figsize=(W1, 2.2), squeeze=False)
    bw = 0.25
    for ax, ds in zip(axes[0], datasets):
        entries = []
        for s in systems:
            g = df[(df["dataset"] == ds) & (df["system"] == s)]
            if len(g):
                entries.append((s, {c: g.loc[g["condition"] == c, "DER"].mean() for c in CONDITIONS}, False))
        if with_published and ds in PUBLISHED:
            entries.append((PUBLISHED_SYSTEM, {c: PUBLISHED[ds][c]["DER"][0] for c in CONDITIONS}, True))
        top = 0
        for gi, (s, means, pub) in enumerate(entries):
            for ci, c in enumerate(CONDITIONS):
                x = gi + (ci - 1) * (bw + 0.02)
                ax.bar(x, means[c], bw, color=CONDITION_COLORS[c], edgecolor="white", linewidth=0.5,
                       hatch="////" if pub else None)
                ax.text(x, means[c] + 0.003, f"{means[c]:.3f}", ha="center", va="bottom", fontsize=5,
                        rotation=90, color=INK)
                top = max(top, means[c])
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([name(s) for s, _, _ in entries], fontsize=6)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, top * 1.3 if top > 0 else 1)
        ax.set_title(dataset_label(ds), fontweight="bold")
    axes[0, 0].set_ylabel("Mean DER")
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.8)
    extra = [Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="published aggregate")] \
        if with_published else []
    cond_legend(fig, extra=extra, ncol=2)
    save(fig, out, "fig_der_by_system")


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS))
    ap.add_argument("--labels", nargs="*", default=[], metavar="SYSTEM=LABEL",
                    help="display names, e.g. --labels vbx=VBx nemotron=\"Streaming Sortformer\"")
    ap.add_argument("--no-published", action="store_true", help="omit the published Pyannote 3.1 bars")
    ap.add_argument("--file-set", choices=["common", "all"], default="common",
                    help="files used in cross-system panels (default: common set)")
    args = ap.parse_args(argv)
    for kv in args.labels:
        k, _, v = kv.partition("=")
        DISPLAY[k] = v

    style()
    df, st = load_results(args.results_dir, args.systems, args.datasets)
    hw = float(df["trad_half_width_s"].dropna().iloc[0]) if "trad_half_width_s" in df else 0.25
    CONDITION_LABELS["traditional"] = trad_label(hw)
    CONDITION_LABELS["selective"] = "Selective Collar (0.25 s conditional snap)"
    CONDITION_LABELS["no_collar"] = "No Collar (0.0 s, strict baseline)"
    df_x = df
    if args.file_set == "common":
        common, _ = common_sets(df, st)
        dfc = restrict(df, common)
        if dfc.empty:
            print("WARNING: common file set is empty; using all scored files per system instead.")
        else:
            df_x = dfc
    datasets = [d for d in DATASETS if d in set(df_x["dataset"])]
    systems = sorted(df_x["system"].unique()) if not args.systems else \
        [s for s in args.systems if s in set(df_x["system"])]
    out = Path(args.results_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {out}  (systems={systems}, datasets={datasets}, "
          f"files={ {d: df_x[df_x['dataset'] == d]['file_id'].nunique() for d in datasets} })")

    pub = not args.no_published
    all_rows = []
    for ds in datasets:
        rows = [(ds, s) for s in systems if ((df_x["dataset"] == ds) & (df_x["system"] == s)).any()]
        all_rows += rows
        fig_boxplots(df_x, rows, out, f"fig_boxplots_{ds}")
    fig_boxplots(df_x, all_rows, out, "fig_boxplots_all")
    fig_decomposition(df_x, datasets, systems, out, pub)
    fig_intervention(df_x, datasets, systems, out)
    fig_ablation(df, datasets, out, hw)      # reference-only quantity: every scored file
    fig_der_by_system(df_x, datasets, systems, out, pub)


if __name__ == "__main__":
    main()
