# Signature Verification — Final Benchmark Report

All numbers below are read from `benchmark/results/*.json` (byte-reproducible;
see `DETERMINISM.md`). Full experiment history: `benchmark/experiments.md`.

## Final result (October 2026): frozen 0–10 clearance score, 1:1 (one reference vs one questioned)

The score (`benchmark/clearance_score.py`) was defined and frozen **before** any optimisation:
`10 × (0.4·D + 0.3·R + 0.3·O)` with D = clean discrimination (skilled + random EER),
R = skilled EER under 11 deterministic capture conditions (tinted paper, shadow, ruled lines,
large canvas, printed text, faint ink, scale 0.5×/2×, rotation +20°/−15°, phone photo) and
O = operating point at the production thresholds. More than 1% forgeries auto-matched caps it at 4.
All images are harmonised first (removes the CEDAR scan-session shortcut, EXP-000).

CEDAR, 55 writers, four disjoint image slices per writer: dev 1–6 (fitting only),
validation 7–12 (every keep/revert decision), test 13–18 (milestones), **holdout 19–24 (scored once)**.

| slice | baseline (b7e932a) | final | role |
|---|---|---|---|
| validation | 3.88 | **5.24** | drove decisions (optimistic) |
| test | 5.26 | **6.00** | never tuned on |
| **holdout** | **4.93** | **5.78** | scored exactly once |

Holdout detail (`clearance_holdout_baseline.json` → `clearance_holdout_final.json`):

| metric | baseline | final |
|---|---|---|
| clean skilled-forgery EER | 11.0% | **9.7%** |
| clean different-writer EER | 3.9% | **2.9%** |
| genuine MATCH / REVIEW / NO MATCH | 14.3 / 79.2 / 6.5% | **28.5 / 65.0 / 6.5%** |
| skilled forgeries MATCH / NO MATCH | 0.0 / 77.2% | **0.0 / 81.1%** |
| printed text near signature (EER) | 31.6% | **9.8%** |
| rotation +20° / −15° (EER) | 17.2 / 19.6% | **13.4 / 13.9%** |
| scale 0.5× / 2× (EER) | 17.4 / 13.1% | 17.2 / 12.8% |
| phone-photo simulation (EER) | 12.4% | **10.4%** |
| skilled MATCH in any capture condition | 0% | 0% |

What changed (EXP-016…019): signature-ink isolation (printed text, ruled lines),
stroke-quality signals (pressure pattern, curvature), calibration on harmonised images,
out-of-fold percentile thresholds, symmetric stroke alignment. Rejected after measurement
(EXP-020…024): elastic-deformation and local-pattern signals, preprocessing re-tune, the
SigNet pretrained embedding (no gain; GPDS non-commercial weights) and pair-conditional
scale/rotation invariance (better scale robustness, lower score; kept on branch
`exp/scale-rotation-invariance`).

**Honest reading.** The 7/10 target was not reached. With one reference, about two in three
genuine signatures still go to manual review, and the forgery EER on clean scans is about
9–12% depending on the slice. Validation gains overstate generalisation (+1.36 vs +0.85 on the
holdout). The largest remaining lever is more than one reference specimen per customer
(EXP-015: 3 specimens cut skilled EER by about a third), followed by real bank images.

## Update (EXP-015): 55-writer evaluation and stroke-level signals

The numbers further down were measured on 12 writers with a two-signal fusion and are kept
as history. Re-measured on **full CEDAR (55 writers)** with an untouched test slice
(images never used for fitting or threshold choice; see `benchmark/build_cedar_eval.py`),
the previous system's 1:1 skilled EER was **16.5%**, not 11.1%. The system now also compares
stroke direction after alignment, slant, ink rhythm (horizontal / vertical profiles) and
pen width.

| untouched test slice, 55 writers | previous | **now** |
|---|---|---|
| 1:1 skilled EER / AUC (equalised scans) | 16.5% / 0.901 | **9.3% / 0.952** |
| 1:1 skilled EER / AUC (raw scans) | 16.4% / 0.897 | **10.4% / 0.949** |
| 1:1 different-writer EER (equalised) | 5.6% | **3.8%** |
| 3-specimen skilled EER (equalised) | 9.4% | **6.1%** |
| 1 specimen: genuine MATCH / skilled MATCH / skilled NO MATCH | 0.4% / 0.0% / 49.4% | **15.9% / 0.0% / 73.8%** |
| 3 specimens: genuine MATCH / skilled MATCH / skilled NO MATCH | 72.7% / 1.5% / 58.3% | 29.4% / **0.0%** / **92.3%** |
| `compare` request latency (1 specimen) | ~340 ms | ~370 ms |

With one specimen most genuine signatures are still routed to manual review (79%); a
single original cannot support straight-through clearing at zero forgery acceptance.
With three specimens the new ACCEPT cut-off is deliberately strict (it sits above the
worst dev impostor), which lowers genuine auto-accept compared with the old cut-off that
was tuned on only 12 writers and let 1.5% of skilled forgeries through. Details, caveats
and the review-gate record: `benchmark/experiments.md` EXP-015.

## Data and protocol (why the old numbers were not comparable)

* 76 distinct images (12 CEDAR writers × 4 genuine + 2 skilled forgeries, plus 4
  synthetic). The old 25/25/25 pair benchmark re-used images and under-used the data.
* **1:1**: 73 genuine / 98 skilled / 1058 random pairs.
  **3-specimen (writer-dependent)**: 48 genuine / 96 skilled / 2112 random trials.
* Thresholds are never chosen on the data they are reported on: writer-disjoint
  2-fold CV (thresholds picked on 6 writers, counted on the other 6), plus a
  12-fold leave-one-writer-out check (EXP-012).
* **Dataset shortcut found:** CEDAR forgeries were scanned in a different session.
  Paper brightness alone separates genuine from skilled pairs with AUC 1.000.
  `raw` = files as stored; `harmonized` = paper/ink appearance equalised (shortcut removed).

## Baseline (legacy v2.0) vs final (v3, through EXP-014)

| metric | legacy raw | legacy harmonized | **final raw** | final harmonized |
|---|---|---|---|---|
| 1:1 skilled AUC | 0.877 | 0.810 | **0.934** | 0.933 |
| 1:1 skilled EER | 19.3% | 27.5% | **11.1%** | 11.1% |
| 1:1 random AUC | 0.978 | 0.978 | **0.992** | 0.993 |
| 1:1 random EER | 9.6% | 8.5% | **2.8%** | 4.1% |
| 1:1 at dev-selected EER cut (held-out): FRR | 20.5% | 27.4% | **13.7%** | 11.0% |
| ... FAR skilled / random | 19.4% / 0.4% | 31.6% / 0.4% | **11.2% / 0.4%** | 9.2% / 0.4% |
| 3-specimen skilled AUC | 0.934 (max) | 0.866 (max) | **0.973** | 0.986 |
| 3-specimen skilled EER | 15.1% | 18.2% | **6.25%** | 6.25% |
| 3-specimen random EER | 4.2% | 5.9% | **2.4%** | 2.1% |

Legacy loses 8 EER points when the scanner shortcut is removed. The final system
is not shortcut-dependent (equal or better on harmonized input). CV-refit
fusion estimate (fusion fitted on the other writer fold): 1:1 skilled AUC 0.927 / EER 14.0%.

### Operating point that drives decisions (3 specimens, held-out writers)

| band outcome | legacy (max-of-3, raw) | **final (raw)** |
|---|---|---|
| genuine auto-accepted (ACCEPT) | 65% | **88%** (2-fold CV) / 75% (leave-one-writer-out) |
| genuine rejected | 6% | 8% |
| **skilled forgeries auto-accepted** | **3.1% (3/96)** | **0% (0/96)** (both CV schemes) |
| random forgeries auto-accepted | 0% | 0% (0/960 2-fold, 0/2112 LOWO) |
| skilled forgeries rejected | 73% | 79% |

0/96 has a 95% upper bound of ≈ 3.1% (rule of three); 12 writers is a small sample.

### Production thresholds (log-odds; `src/core/config.py::DecisionThresholds`)

| mode | ACCEPT ≥ | REJECT < | hard reject (SIG_DIFF_01) < |
|---|---|---|---|
| ≥ 2 specimens | 4.650 | 1.427 | 0.625 |
| 1 specimen | 13.938 | −0.278 | −3.834 |

Rule: ACCEPT = highest impostor logit + 1.0; REJECT = 5th percentile of genuine
(`benchmark/select_thresholds.py`). Leave-one-writer-out cut-off sd = 0.13 logit.
One specimen cannot support straight-through clearing (no tested rule reached 0
held-out false accepts), so single-specimen cheques are effectively routed to REVIEW.

## Robustness (3 specimens, production operating point; `robustness_exp014.json`)

| perturbation | genuine ACCEPT / REJECT / INCONCLUSIVE | skilled ACCEPT |
|---|---|---|
| none | 75% / 4% / 0% | 0% |
| rotation +5° | 73% / 6% / 0% | 0% |
| rotation −10° | 73% / 6% / 0% | 0% |
| rotation +15° | 67% / 10% / 0% (was 25% / 31% before EXP-014) | 0% |
| scale ×0.75 / ×1.5 | 65% / 8% · 60% / 4% | 0% |
| scale ×0.5 | 12% / 15% / 0% (known limitation) | 0% |
| JPEG q20 / q10 | 71% / 4% · 56% / 4% | 0% |
| noise σ10 | 71% / 6% / 0% | 0% |
| noise σ25 | 33% / 10% / 6% | 0% |
| thicker pen (+1 px) | 60% / 4% / 0% | 0% |
| blur σ3 | 0% / 0% / **100%** (quality gate; never a false REJECT) | 3% (1 trial) |
| contrast ×0.3 | 0% / 6% / 77% | 0% |

No perturbation lifts skilled forgeries into ACCEPT, except a single trial under
heavy blur that passed the quality gate.

## Cheque path (composed cheques, `cheque_path_exp009.json`)

Genuine queries through the full cheque pipeline (`cheque_path_exp014.json`):
ACCEPT 65%, REJECT 10%, INCONCLUSIVE 4% (direct-scan upper bound: 75% / 4%);
skilled ACCEPT 0%.

## Runtime and resources (Apple Silicon, single thread)

| path | latency |
|---|---|
| legacy `verify` (1 specimen) | 8.4 ms |
| v3 `verify` (1 specimen, incl. quality gate, NLM denoise, alignment, diagnostics) | 165 ms |
| v3 `verify_against_references` (3 specimens) | 317 ms |
| benchmark pair comparison with cached features | ~2.8 ms |

Peak RSS during the benchmark is ~475 MB. No GPU is used.

## Determinism

Benchmark, robustness and cheque-path JSON are byte-identical across separate
processes (3 runs each, at every experiment), and so is `VerificationResult`
across 3 fresh interpreters (`TestCrossProcessDeterminism`). Sources and controls:
`DETERMINISM.md`.

## Tests

101 tests (`python -m pytest signature_verification_system/tests -q`) covering:
genuine, skilled forgery, different writer, near-identical look-alike, rotation,
scale, blur / low contrast / blank / black / micro / noise images, polarity,
multi-specimen ordering, explanation faithfulness, API hardening, pipeline
end-to-end, and cross-process determinism.

## Known limitations and the data that would fix them

* **12 writers is small.** EER moves ~1–2 points per sample. The full CEDAR (55
  writers × 24 genuine + 24 forgeries), GPDS or MCYT would give real confidence
  intervals and allow a held-out test split separate from threshold selection.
* **Real cheques with enrolled specimens are missing.** The cheque path is measured
  on composed cheques only.
* **Look-alike signatures:** a constructed different-author look-alike (synthetic
  writer B) outscores the author's own genuine variant. Keypoint and shape
  features alone cannot separate it.
* **Scale ×0.5** still degrades genuine acceptance (12% ACCEPT). Rotation up to 15°
  is now handled by RANSAC alignment (EXP-014). Forgeries are still not accepted,
  so the remaining cost lands on manual review.
* **Security posture is demo-grade:** no auth or rate limiting, and the audit
  chain is unkeyed SHA-256 (see EXP-010).
