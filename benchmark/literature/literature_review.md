# Literature review: multi-model validation of the Selective Collar

Prepared 2026-11-15 for the ICAIIC 2027 short paper extending Vinod et al. [vinod2026selective].
Citation keys refer to `references.bib` in this folder (38 entries: 35 checked online today, 3 reused from the journal paper's reference list and marked `[REUSED]` in the .bib).

How the evidence was checked: metadata (title, authors, venue, year, pages, DOI) was confirmed against Crossref, the ISCA archive or the arXiv listing. Every DER number below was read from the paper PDF/HTML or the model card itself, and its location is given in the tables. Where a table could only be read from flattened PDF text, this is stated.

Things to act on before submission are collected in Section 6 (cautions) and Section 7 (unverified items).

---

## 1. RQ1: Multi-system benchmarking studies and their collar/overlap settings

**Synthesis.** Recent comparative studies do not share a scoring protocol, and several say so themselves.

- Lanzendörfer et al. [lanzendorfer2025benchmarking] compare pyannote 3.1, the pyannoteAI API, Sortformer, Streaming Sortformer (v2) and DiariZen on CALLHOME, VoxConverse v0.3, AMI and AliMeeting. They score with pyannote.metrics using "a 0.25 second collar and skip_overlap=False". They report that missed speech is the dominant error for all models except Sortformer, which shows more speaker confusion on VoxConverse (attributed to its four-speaker limit), and that missed segments average roughly 350 ms, which they link to boundary precision on longer segments rather than to short segments.
- SDBench [pacheco2025sdbench] covers 13 datasets (including VoxConverse and MSDWild) and six systems (pyannote 3.1, pyannoteAI API, Deepgram, AWS Transcribe, Picovoice Falcon, SpeakerKit). It makes the opposite choice: "Unlike other benchmarking tools that use forgiveness collars depending on the dataset in DER calculation, we chose not to apply them across all datasets." NeMo and VBx are not among its systems.
- Pálka et al. [palka2026vbx] compare pyannote 3.1 and DiariZen variants with AHC or VBx clustering on eight test sets including MSDWild and VoxConverse, "without collars".
- DiaPer [landini2024diaper] compares a VAD+VBx+OSD cascade with EEND models on 13 wide-band sets and uses a per-dataset collar: 0 s by default, 0.25 s "in cases where a challenge or the authors proposed differently", which includes both VoxConverse and MSDWild (Table II).
- Serafini et al. [serafini2023experimental] benchmark eight systems from three families (clustering-based including VBx, EEND/EEND-VC, separation-guided) on two-speaker telephone speech with a "fair" setup: overlap scored, 0.25 s collar.
- Fischbach [fischbach2024comparative] compares pyannote, CLEAVER and NeMo but bypasses DER entirely because the reference annotation is incomplete, scoring through a downstream dialect classifier instead.

**Same system, different numbers.** Because of these choices, the same model appears with very different DERs. pyannote 3.1 on VoxConverse is 11.3% in its model card and in [palka2026vbx] (no collar), while the powerset paper reports the same family at 10.4% without collar and 5.8% with a 250 ms collar; on MSDWild the corresponding pair is 28.5% and 18.6% [plaquet2023powerset]. Landini et al. [landini2022vbx] show VBx on CALLHOME moving from 4.42% (0.25 s collar, no overlap) to 14.21% (0.25 s collar, overlap scored) to 21.77% (no collar, overlap scored), and note "the mix of evaluation setups ... with different collars and criteria for including or excluding overlapped speech" in published AMI results. Bredin [bredin2023pyannote] lists "different configuration of the reported diarization error rate (e.g. with or without forgiveness collars)" among the reasons published results are hard to compare, and states that no comparable VoxConverse baseline could be found for that reason. Park et al. [park2022msdd] likewise report two settings per dataset.

**What was not found.** No study located explicitly demonstrates a *ranking inversion* between toolkits caused by the collar choice. The evidence is that settings are inconsistent and that the collar changes absolute DER by a large, system- and dataset-dependent amount. The new paper should not claim that prior work shows rank changes.

### Table 1. Benchmark studies: protocol

| Study | Systems | Datasets (relevant) | Collar | Overlap scored? | Source location |
|---|---|---|---|---|---|
| [lanzendorfer2025benchmarking] | pyannote 3.1, pyannoteAI, Sortformer, Sortformer v2 (+stream), DiariZen | CALLHOME, VoxConverse v0.3 (dev+test), AMI, AliMeeting | 0.25 s | Yes (`skip_overlap=False`) | Sec. 2.2 |
| [pacheco2025sdbench] | pyannote 3.1, pyannoteAI API, Deepgram, AWS Transcribe, Picovoice Falcon, SpeakerKit | 13 sets incl. VoxConverse, MSDWild | 0 s | Not stated explicitly | Sec. 2.2, 2.4 |
| [palka2026vbx] | pyannote 3.1, DiariZen + AHC / VBx variants | AMI, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3, RAMC, VoxConverse | 0 s | Not stated explicitly | Sec. 4, Table 2 |
| [landini2024diaper] | VAD+VBx+OSD, DiaPer variants | 13 sets incl. MSDWild (few.val as test), VoxConverse test | Per dataset: 0.25 s for VoxConverse and MSDWild | Yes (Table XIII caption) | Sec. IV-B, Tables II, XIII |
| [serafini2023experimental] | 8 systems, 3 families (incl. VBx, SA-EEND, EEND-VC) | 4 two-speaker CTS sets | 0.25 s | Yes | evaluation-setup paragraph |
| [bredin2023pyannote], [plaquet2023powerset], [pyannote2023card31] | pyannote 2.1 / 3.x | 9-11 sets incl. VoxConverse v0.3, MSDWild | 0 s (plus a 250 ms row in [plaquet2023powerset]) | Yes | Table 1 of each paper; card benchmark table |
| [landini2022vbx] | VBx vs prior work | CALLHOME, AMI, DIHARD II | Three setups: forgiving (0.25 s, no overlap), fair (0.25 s, overlap), full (0 s, overlap) | Varies | Sec. 3.4, Table 2 |
| [park2022msdd] | NeMo MSDD | CALLHOME, AMI, CH109 | Two setups: 0.25 s without overlap; 0 s with overlap | Varies | Sec. 4.2 |
| [fischbach2024comparative] | pyannote 2.1, CLEAVER, NeMo | German dialect recordings | DER not used | n/a | Sec. 1 |

---

## 2. RQ2: The three systems and their reported DER on VoxConverse and MSDWild

**pyannote/speaker-diarization-3.1 (hybrid).** A powerset segmentation model applied on a sliding window, followed by speaker embedding and agglomerative clustering [plaquet2023powerset], [bredin2023pyannote]. The model card reports 11.3% DER on VoxConverse v0.3 and 25.3% on MSDWild under "no forgiveness collar" and "evaluation of overlapped speech" [pyannote2023card31]; [palka2026vbx] reproduces both numbers. The powerset paper gives the only like-for-like collar comparison found for these two datasets (Table 2 below). The original MSDWild baseline for an earlier pyannote release is 21.96% on few-talker val with a 0.25 s collar and overlap counted [liu2022msdwild]. The segmentation-3.0 model card lists MSDWild and VoxConverse among its training sets (training/development portions per [plaquet2023powerset]).

**BUT VBx (clustering-based).** x-vectors are extracted every 0.25 s from overlapping 1.5 s sub-segments of VAD-defined speech, clustered by AHC and refined with a Bayesian HMM [landini2022vbx]. The primary paper uses oracle VAD and does not evaluate VoxConverse or MSDWild; it states the system does not handle overlapped speech. VoxConverse results come from the BUT VoxSRC-20 system description [landini2021but], scored with the challenge's 0.25 s collar: on the development set, VBx gives 4.36% with the full (fused) VAD and 5.69% with an energy-based VAD. On the evaluation set the full submission obtained 8.12%. On the 232-file test set and MSDWild few.val, [landini2024diaper] reports a 16 kHz VAD+VBx+OSD cascade at 6.1% and 16.9% (0.25 s collar, overlap scored). That cascade uses a neural VAD and overlap detection, so it is not the same configuration as energy VAD + vanilla VBx.

**NVIDIA Nemotron-3-Diarization (end-to-end).** Documentation exists in two forms: the Hugging Face model card (released 2026-09-23) [nvidia2026nemotroncard] and an arXiv preprint dated 2026-11-09, "Submitted to ICASSP 2027", not peer reviewed [medennikov2026nemotron]. It extends Streaming Sortformer [medennikov2025streaming], itself built on Sortformer [park2025sortformer], from four to eight speakers, replaces the FastConformer-Transformer stack with a 31-layer Transformer encoder at an 80 ms frame step, and adds a learned upsampling layer that produces 10 ms predictions. Neither the card nor the preprint reports DER on VoxConverse or MSDWild. Both state that all overlap is scored, with 0 s collar everywhere except CALLHOME (0.25 s). The same holds for [park2025sortformer] and [medennikov2025streaming]: collar 0 s on DIHARD III, 0.25 s on CALLHOME and CH109, VoxConverse used only for training. The only VoxConverse evidence for the Sortformer family is third-party [lanzendorfer2025benchmarking], where per-dataset values appear only in a figure.

### Table 2. Reported DER on VoxConverse and MSDWild

| System / benchmark | Dataset | Reported DER (%) | Collar | Overlap scored? | Source |
|---|---|---|---|---|---|
| pyannote 3.1 pipeline | VoxConverse v0.3 test | 11.3 (FA 4.1 / Miss 3.4 / Conf 3.8) | 0 s | Yes | [pyannote2023card31] benchmark table |
| pyannote 3.1 pipeline | MSDWild | 25.3 (FA 5.8 / Miss 8.0 / Conf 11.5) | 0 s | Yes | [pyannote2023card31] benchmark table |
| pyannote 3.1 (re-reported) | VoxConverse / MSDWild | 11.3 / 25.3 | 0 s | Unstated | [palka2026vbx] Table 2 |
| pyannote 3.1 + constrained VBx | VoxConverse / MSDWild | 11.2 / 22.8 | 0 s | Unstated | [palka2026vbx] Table 2 |
| DiariZen Large + VBx | VoxConverse / MSDWild | 8.7 / 17.0 | 0 s | Unstated | [palka2026vbx] Table 2 |
| Powerset, compound training | VoxConverse v0.3 / MSDWild | 11.6 / 29.2 | 0 s | Yes | [plaquet2023powerset] Table 1 (a) |
| Powerset + domain adaptation | VoxConverse v0.3 / MSDWild | 10.4 / 28.5 | 0 s | Yes | [plaquet2023powerset] Table 1 (a) |
| Powerset + domain adaptation, same outputs | VoxConverse v0.3 / MSDWild | 5.8 / 18.6 | 0.25 s | Yes | [plaquet2023powerset] Table 1 (a) |
| pyannote 2.1 default pipeline | VoxConverse v0.3 | 11.2 (FA+Miss 7.3 / Conf 3.9) | 0 s | Yes | [bredin2023pyannote] Table 1 |
| pyannote (earlier release), audio-only baseline | MSDWild few-talker val | 21.96 (MS 5.5 / FA 3.16 / SE 13.3) | 0.25 s | Yes | [liu2022msdwild] Table 3 |
| pyannote (earlier release), audio-only baseline | MSDWild many-talker val | 43.15 | 0.25 s | Yes | [liu2022msdwild] Table 4 |
| VBx, full fused VAD, no overlap handling | VoxConverse dev (216 files) | 4.36 (Miss 3.10 / FA 0.47 / Spk 0.79) | 0.25 s | Yes (challenge protocol) | [landini2021but] Table 2 |
| VBx + reclustering, energy VAD | VoxConverse dev | 5.69 (Miss 2.99 / FA 1.78 / Spk 0.92) | 0.25 s | Yes | [landini2021but] Table 6 |
| VBx + reclustering, oracle VAD | VoxConverse dev | 3.09 (Miss 2.30 / FA 0 / Spk 0.79) | 0.25 s | Yes | [landini2021but] Table 6 |
| BUT full submission (VBx + OSD) | VoxSRC-20 evaluation set | 8.12 | 0.25 s | Yes | [landini2021but] Sec. 3 |
| VAD + VBx + OSD, 16 kHz | VoxConverse test (232 files) / MSDWild few.val | 6.1 / 16.9 | 0.25 s | Yes | [landini2024diaper] Table XIII (b) |
| VAD + VBx + OSD, 8 kHz | VoxConverse test / MSDWild few.val | 6.7 / 18.8 | 0.25 s | Yes | [landini2024diaper] Table XIII (b) |
| DiaPer, best fine-tuned | VoxConverse test / MSDWild few.val | 18.2 / 13.4 | 0.25 s | Yes | [landini2024diaper] Table XIII (b) |
| DiariZen | VoxConverse (dev+test) | 5.2 | 0.25 s | Yes | [lanzendorfer2025benchmarking] Sec. 3.1 |
| Sortformer, Streaming Sortformer | VoxConverse | Values only in Fig. 1; elevated speaker confusion noted | 0.25 s | Yes | [lanzendorfer2025benchmarking] Sec. 3.1 |
| Nemotron-3-Diarization | VoxConverse / MSDWild | **Not reported** | n/a | n/a | [nvidia2026nemotroncard], [medennikov2026nemotron] |
| Nemotron-3-Diarization, offline (30.4 s) | DIHARD III eval / CALLHOME part 2 | 12.73 / 9.10 | 0 s / 0.25 s | Yes | [nvidia2026nemotroncard] evaluation tables |
| Streaming Sortformer, 1.04 s latency, no post-processing | DIHARD III eval / CALLHOME part 2 | 19.89 / 12.00 | 0 s / 0.25 s | Yes | [medennikov2025streaming] Table 1 (c) |
| VBx, oracle VAD | CALLHOME | 4.42 / 14.21 / 21.77 | 0.25 s no-overlap / 0.25 s / 0 s | No / Yes / Yes | [landini2022vbx] Sec. 5.1, Table 2 |

(a) Column order was reconstructed from flattened PDF text: six left-hand columns (AISHELL-4, AliMeeting, AMI IHM, AMI SDM, Ego4D, MSDWild) then REPERE, VoxConverse, in-domain average, DIHARD III. The reading is internally consistent (the in-domain averages 23.5 and 21.7 match the text of the paper), but check the PDF before quoting the collar pair in the manuscript.
(b) Values read from the arXiv HTML table and consistent with the flattened PDF text.
(c) Read from the arXiv HTML table; the fetch reported partial garbling of that table, so re-check the row before quoting.

---

## 3. RQ3: Architectural families and boundary behaviour

**Clustering-based (VBx).** Segment boundaries are inherited from the VAD, and speaker labels are assigned at the x-vector rate: every 0.25 s from 1.5 s windows, with segments shorter than 0.1 s discarded [landini2022vbx]. Speaker change points are therefore quantised to roughly a quarter of a second, the same order as the 0.25 s collar. Vanilla VBx assigns one speaker per frame; [landini2022vbx] notes that under the fair setup performance drops because of "overlapped speech (which our system does not handle)". On VoxConverse dev, even with oracle VAD and no speaker error the DER floor is 2.3% without overlap handling [landini2021but]. VAD quality matters directly: on VoxConverse dev, an energy VAD has 1.01% miss and 4.44% false alarm against 1.71% and 2.05% for a DNN VAD (Table 1), and moving from the fused VAD to the energy VAD raises DER from 4.30% to 5.69%, mostly through false alarm (0.47 to 1.78) [landini2021but]. The same paper observes that oracle VAD brings only a further 1.21% because "the boundaries of speech segments in the oracle labels are not perfect (for this reason 0.25 s forgiveness collar is used in the challenge)".

**End-to-end (EEND family, Sortformer).** EEND casts diarization as frame-wise multi-label classification with permutation-free training and handles overlap natively [fujita2019eend]. Frame resolution is coarse: EEND-EDA uses features every 100 ms, reduced to 50 ms at inference for zero-collar test sets "to obtain more fine-grained results" [horiguchi2022eda]. Sortformer and Streaming Sortformer predict at 80 ms [medennikov2025streaming]. Nemotron 3 keeps the 80 ms encoder step but adds learned upsampling to 10 ms, motivated as "enabling accurate detection of short turns and precise boundaries"; its ablation (Table 3, pooled zero-collar DER, offline) lists 10.42% for the full model and 12.69% for the "no upsampling (80 ms)" variant [medennikov2026nemotron] (read from flattened PDF text; re-check the row before quoting). The Sortformer paper also reports results with and without a timestamp post-processing step [park2025sortformer]. End-to-end models have a fixed output-speaker capacity (four for Sortformer, eight for Nemotron 3), which shows up as speaker confusion on many-speaker recordings [lanzendorfer2025benchmarking]. Related variants are TS-VAD [medennikov2020tsvad] and DiaPer [landini2024diaper]; DiaPer is far behind VBx on VoxConverse (18.2% vs 6.1%) but ahead on MSDWild (13.4% vs 16.9%), showing a family-by-dataset interaction on exactly the two corpora used here.

**Hybrid (EEND-VC, pyannote, DiariZen).** Local end-to-end segmentation on short windows followed by embedding clustering across windows [kinoshita2021eendvc]. pyannote 2.1 outputs speaker activity every 16 ms on 5 s windows and optionally fills within-speaker gaps, a step whose value "depends more on the instructions given to the pool of human annotators than to the data itself" [bredin2023pyannote]. The 3.x powerset model replaces multi-label outputs with powerset classes [plaquet2023powerset]; the pyannote/segmentation-3.0 model card (fetched today, not in the .bib) describes 10 s windows and seven classes, i.e. at most three speakers per window and two simultaneous speakers. DiariZen follows the same design with WavLM features [han2025diarizen], and [palka2026vbx] reports about 0.5 DER gain on VoxConverse from filling gaps shorter than 0.5 s, because short pauses are annotated as speech there. Multi-scale clustering in NeMo MSDD trades temporal resolution against embedding quality with scales from 0.5 s to 3 s [park2022msdd]. WhisperX-style pipelines attach pyannote diarization to ASR words; the "collar" in [bain2023whisperx] is a 200 ms tolerance for word-timestamp precision/recall, not a DER collar.

**Implication for the new paper.** The three systems place boundaries at different granularities (about 16 ms hybrid, 10 ms upsampled from 80 ms end-to-end, VAD boundaries plus 0.25 s label steps for VBx) and differ in overlap handling. No located work measures how these differences interact with a forgiveness rule.

---

## 4. RQ4: Evaluation-protocol literature

**Origin and removal of the collar.** The md-eval tool from the NIST Rich Transcription evaluations (2002-2009) allows forgiveness collars of any size, with ±250 ms used historically, and optional exclusion of overlap [mcknight2020phonetic]. DIHARD departed from this: "Contrary to practice in the NIST RT evaluations, NO forgiveness collar will be applied to the reference segments prior to scoring and overlapping speech WILL be evaluated" [ryant2020dihard3plan]; see also [ryant2019dihard2], [ryant2021dihard3]. DIHARD II also introduced the Jaccard error rate. CHiME-6 followed DIHARD (no collar, overlap scored) [medennikov2020tsvad].

**The two datasets used here keep the collar.** VoxConverse was annotated finely enough to allow "evaluation without a forgiveness collar", yet its baseline and the VoxSRC challenges apply 0.25 s "to compensate for small inconsistencies in annotation", with overlap scored [chung2020voxconverse], [huh2024voxsrc]. The VoxConverse authors themselves suggest their lower DER relative to DIHARD "may be attributed to the use of a 0.25-second forgiveness collar". MSDWild specifies "a 0.25-second forgiving collar ... for all metrics, and overlapped speeches are taken into account" [liu2022msdwild]. Later work on both datasets is split between the dataset protocol (0.25 s) and the pyannote/DIHARD convention (0 s); see Table 1.

**Critiques.** McKnight et al. argue in both directions: removing collars "can unfairly penalize" systems with correct boundaries because phoneme-boundary uncertainty depends on position and phoneme, "so that the use of a uniform forgiveness collar is inadequate" [mcknight2020phonetic]; and in a human-annotation study on AMI, "the use of forgiveness collars is shown to be unhelpful", with human-review DER moving from 11.93% to 18.94% (one ground truth) between 250 ms and 0 ms collars and toy examples where the collar either hides a 16.7% error entirely or slightly raises DER [mcknight2022human]. Landini et al. recommend the no-collar, overlap-scored setup while defining a "fair" intermediate one, "as no realistic human annotation can achieve frame-precision" [landini2022vbx].

**Segment-level and alternative metrics.** CDER [cheng2022cssd] and SER/BER [liu2022ber] move to segment counting with IoU thresholds; these are covered in the journal paper and not repeated here. Collar-aware training [kalda2022collar] moves the tolerance into the loss for speaker change detection.

**2025-2026 work on annotation ambiguity.** Horiguchi et al. show that loose boundaries in ASR-oriented corpora inflate DER and that "collar tolerance is largely ineffective" for this, because the mismatch comes from pauses inside reference segments rather than from boundary misalignment [horiguchi2025repurpose]. Their follow-up decomposes DER exactly into pause-attributable and core components and states that "DERs with different collar settings are not directly comparable because both scored errors and the reference speaker-time denominator change"; it proposes, as future work, keeping the original denominator and treating errors inside collar regions as a separate component [horiguchi2026decomposition]. The Nemotron 3 preprint switches to forced-alignment references for AMI, AliMeeting and NOTSOFAR-1 for the same reason and warns that results under different reference labels are not comparable [medennikov2026nemotron]. This is the closest current line of work to the Selective Collar: it shares the aim of keeping the denominator intact, but addresses pauses, not boundary jitter, and does not condition forgiveness on speaker identity.

---

## 5. Gap statement

Existing multi-system benchmarks each fix a single scoring rule (0.25 s collar in [lanzendorfer2025benchmarking], [serafini2023experimental]; no collar in [pacheco2025sdbench], [palka2026vbx]; per-dataset in [landini2024diaper]) and report one DER per system. Where the same outputs are scored under two rules [plaquet2023powerset], [landini2022vbx], [park2022msdd], only one system or one architectural family is involved and both rules are unconditional (the collar region is removed for every boundary, correct or not). The Selective Collar itself was validated on a single hybrid system [vinod2026selective].

No located study scores the *identical hypotheses* of architecturally different systems (clustering-based, end-to-end, hybrid) under no collar, an unconditional collar and a conditional (timing-and-identity) collar, on common data, to test whether the amount of forgiveness each rule grants depends on the architecture. For Nemotron-3-Diarization in particular, no DER on VoxConverse or MSDWild has been published under any rule.

---

## 6. Claims the new paper can safely make

1. **Collar and overlap settings are inconsistent across recent multi-system benchmarks**, including on VoxConverse and MSDWild: 0.25 s in [lanzendorfer2025benchmarking], [landini2024diaper]; 0 s in [pacheco2025sdbench], [palka2026vbx], [pyannote2023card31]. Bredin names collar configuration as a cause of non-comparable results [bredin2023pyannote].
2. **Both datasets were released with a 0.25 s collar and overlap scored** [chung2020voxconverse], [huh2024voxsrc], [liu2022msdwild], whereas DIHARD removed the collar [ryant2019dihard2], [ryant2020dihard3plan].
3. **The collar changes absolute DER substantially for a fixed system**: e.g. 4.42% to 21.77% for VBx on CALLHOME across forgiving and full setups [landini2022vbx]; and, subject to footnote (a), 10.4% to 5.8% on VoxConverse and 28.5% to 18.6% on MSDWild for a powerset pyannote model [plaquet2023powerset].
4. **DERs under different collar settings are not directly comparable because the collar alters the reference-time denominator as well as the scored errors** [horiguchi2026decomposition]; this supports the denominator-preserving design of the Selective Collar.
5. **A uniform collar is a crude remedy for boundary uncertainty**: boundary uncertainty is phoneme- and position-dependent [mcknight2020phonetic], and uniform collars were found unhelpful in a human-annotation study [mcknight2022human]; collars also do not address pause-annotation ambiguity [horiguchi2025repurpose].
6. **The three evaluated systems represent three architectural families with different temporal granularity and overlap handling**: VBx labels 0.25 s-spaced x-vectors on VAD segments and does not handle overlap [landini2022vbx]; pyannote uses frame-level powerset local segmentation plus clustering [plaquet2023powerset], [bredin2023pyannote]; Nemotron 3 is a Sortformer-type end-to-end model with an 80 ms encoder step upsampled to 10 ms and an eight-speaker limit [medennikov2026nemotron], [nvidia2026nemotroncard].
7. **VAD choice measurably affects clustering-based diarization on VoxConverse**: energy VAD vs fused neural VAD gives 5.69% vs 4.30% DER on the development set at 0.25 s collar, driven by false alarm [landini2021but].
8. **Nemotron-3-Diarization has no published DER on VoxConverse or MSDWild**; its documentation reports DIHARD III, CALLHOME, AliMeeting, AMI and NOTSOFAR-1 only, at 0 s collar except CALLHOME [nvidia2026nemotroncard], [medennikov2026nemotron]. The same is true of the Sortformer and Streaming Sortformer papers [park2025sortformer], [medennikov2025streaming].
9. **Prior multi-system benchmarks report a single scoring rule per study, and the Selective Collar has so far been validated on one system** [vinod2026selective]; the present study is, to the authors' knowledge, the first to score identical outputs of three architecture families under conditional and unconditional forgiveness. (Phrase as "to our knowledge"; this rests on a focused one-day search, not an exhaustive survey.)

**Cautions (things not to claim, or to disclose).**

- **Training-data overlap.** The Nemotron model card lists "VoxConverse v0.3 ... Development and test" among training data [nvidia2026nemotroncard], and the preprint lists VoxConverse-v0.3 as training data [medennikov2026nemotron]. Nemotron's VoxConverse test results are therefore probably not held-out and should be disclosed as such. pyannote's segmentation model was also trained on portions of VoxConverse and MSDWild [plaquet2023powerset]. VBx's components were not trained on either. Cross-system accuracy comparisons on VoxConverse are confounded; comparisons of *how each scoring rule treats each system's output* are less affected.
- **Do not claim prior evidence of ranking inversions** caused by collar choice; none was found.
- **MSDWild file count.** The MSDWild paper (Table 2) and [landini2024diaper] (Table II) give 490 recordings for few-talker val; the task description says 488. Reconcile and state the reason (e.g. missing or unreadable files) in the paper.
- **Speaker capacity.** VoxConverse test has up to 21 speakers per recording [chung2020voxconverse], above Nemotron's eight-speaker output; expect speaker confusion that is unrelated to boundaries, as seen for four-speaker Sortformer [lanzendorfer2025benchmarking].
- **VBx reference numbers are not for the same configuration.** Published VoxConverse-test and MSDWild numbers for VBx [landini2024diaper] use a neural VAD and overlap detection; energy-VAD numbers exist only for VoxConverse dev [landini2021but].
- The Nemotron preprint is not peer reviewed; cite it as a preprint alongside the model card.

---

## 7. UNVERIFIED or partially verified items (not in `references.bib`)

| Item | Status |
|---|---|
| NIST RT-09 Meeting Recognition Evaluation Plan (Sec. 6, collar definition) | **UNVERIFIED.** The document could not be retrieved (NIST URL returned an HTML page, mirror hosts unreachable). The 0.25 s RT collar is supported here only through secondary sources that were read: [mcknight2020phonetic], [ryant2020dihard3plan]. Do not quote the plan's wording. |
| J. G. Fiscus, J. Ajot, M. Michel, J. S. Garofolo, "The Rich Transcription 2006 Spring Meeting Recognition Evaluation," LNCS, 2006, pp. 309-322, doi 10.1007/11965152_28 | Metadata confirmed on Crossref; **body not read**, so no claim is attributed to it. |
| X. Anguera et al., "Speaker Diarization: A Review of Recent Research," IEEE TASLP, vol. 20, pp. 356-370, 2012, doi 10.1109/TASL.2011.2125954 | Metadata confirmed on Crossref; **body not read**. |
| NIST md-eval / `dscore` scoring tools | Referred to in [ryant2020dihard3plan], [landini2022vbx]; repositories not opened today. |
| Sortformer / Streaming Sortformer per-dataset DER on VoxConverse in [lanzendorfer2025benchmarking] | Values exist only in that paper's Fig. 1 and could not be read from text. |
| Overlap handling in [pacheco2025sdbench] and [palka2026vbx] DER | Not stated explicitly in either paper; marked "unstated" above. |
| pyannote segmentation-3.0 output frame step | Not stated on the model card; 16 ms is documented for the 2.1 model [bredin2023pyannote]. |
| Venue of [pacheco2025sdbench] and [horiguchi2026decomposition] | Only arXiv listings found; cited as preprints. |
| Journal-paper references reused without re-fetching ([bredin2017pyannotemetrics], [park2022review]) | Bibliographic data copied from the published IEEE Access reference list. |
| "0.25 s collar with overlap skipped roughly halves DER" (vendor blog) | Seen only in a search summary of a vendor blog; not used. |
