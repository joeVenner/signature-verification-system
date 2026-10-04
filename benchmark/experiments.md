# Experiment Log

Reproduce any row (from the workspace parent directory):

```bash
python -m signature_verification_system.benchmark.run_benchmark \
    --system <legacy_v2_0|current> --condition <harmonized|raw> --repeats 3 \
    --out signature_verification_system/benchmark/results/<name>.json
```

## Protocol (fixed for every experiment)

* 76 distinct images (SHA-256 de-duplicated from the 150 manifest entries):
  12 CEDAR writers × (4 genuine + 2 skilled forgeries) + 3 synthetic images.
* **1:1**: 73 genuine pairs, 98 skilled-forgery pairs, 1058 random (cross-writer) pairs.
* **Writer-dependent (WD)**: enroll 3 genuines, query held-out genuine (48),
  skilled forgeries (96), other writers' genuines (2112).
* **Thresholds are never chosen on the data they are reported on.** Writers are
  split into two disjoint folds (odd/even CEDAR index). ACCEPT / REJECT
  cut-offs are chosen on one fold (ACCEPT: zero dev impostors above it;
  REJECT: ≤5% dev genuines below it) and applied to the other; results are
  pooled over both test folds.
* AUC / EER are threshold-free and computed exactly (Mann-Whitney U; all
  unique score thresholds).
* **Headline condition = `harmonized`** (see EXP-000).

Caveat: 12 writers is small. One skilled pair = 1.0% of the 98 skilled pairs;
differences below ~2 EER points are within noise.

---

## EXP-000 — Baseline + dataset shortcut audit

* **Hypothesis:** the existing 25/25/25 benchmark (skilled EER 16%) under-uses
  the data and may be inflated.
* **Finding 1:** the 150 manifest images are only 76 distinct images; the
  expanded protocol above uses all valid comparisons.
* **Finding 2 (critical):** a trivial probe that compares only the *median
  paper brightness* of two images separates genuine from skilled-forgery
  pairs with **AUC = 1.000** (random pairs: 0.520). CEDAR forgeries were
  scanned in a different session (paper ≈ 244 vs 224 grey levels). Ink
  darkness (AUC 0.82) and paper noise (0.79) leak too.
* **Change:** added `benchmark/conditions.py::harmonize`, a label-agnostic
  flat-field normalisation (paper→white, ink p90→fixed darkness). After it the
  brightness probe drops to AUC 0.500.

| legacy v2.0 (1:1)            | raw    | harmonized |
|------------------------------|--------|------------|
| skilled AUC                  | 0.877  | 0.810      |
| skilled EER                  | 19.3%  | 27.5%      |
| random AUC                   | 0.978  | 0.978      |
| random EER                   | 9.6%   | 8.5%       |
| CV genuine auto-accept       | 33%    | 18%        |
| CV skilled auto-accept (FAR) | 2%     | 1%         |
| WD max-of-3 skilled EER      | 15.1%  | 18.2%      |
| WD max-of-3 skilled auto-accept | 3%  | 8%         |

* Determinism: 3 in-process repeats + 2 fresh processes → byte-identical JSON.
* Runtime: ~8–10 ms per pair comparison (single thread, Apple Silicon), ~220 MB RSS.
* **Decision:** BASELINE. ~8 EER points of the legacy skilled-forgery
  performance came from scanner appearance, not from the signature. All later
  experiments report `harmonized` as the headline.
* Per-feature AUCs (raw, genuine-vs-skilled / genuine-vs-random): HOG 0.81/0.94,
  skeleton 0.80/0.96, contour 0.75/0.78, stroke-width 0.66/0.80,
  hesitation 0.63/**0.54**, curvature 0.58/0.70, Hu 0.58/0.66. Hesitation carries no
  writer identity; its skilled AUC is consistent with the scan artefact.

---

## EXP-001 — v3 representation: photometric normalisation + shape/keypoint fusion

* **Hypothesis:** the legacy fused score is limited by (a) intensity-sensitive
  features that partly measure the scanner, and (b) a coarse, alignment-rigid
  representation (16-px HOG on a stretched canvas). A photometrically
  normalised ink map, described by a coarse gradient-direction grid (global
  layout) plus geometrically verified SIFT keypoints (fine stroke detail),
  should separate skilled forgeries better without using the shortcut.
* **Selection (no test-set tuning):** 10 candidate similarities were prototyped
  and judged on per-fold AUC (fold 0 vs fold 1 must agree). ECC-aligned
  correlation was dropped (fold AUC 0.79 vs 0.90 = unstable) and it degraded
  the fusion under CV. Skeleton chamfer added nothing over the gradient grid.
* **Change:** `src/preprocessing/normalization.py`, `src/verification/features.py`,
  `src/verification/similarity.py`; `DeterministicVerifier.verify()` now scores
  with a 3-parameter logistic fusion (`FusionModel`, fitted by
  `benchmark/fit_fusion.py`). The legacy v2 sub-scores remain in the response
  as diagnostics (computed on normalised pixels) and are available via
  `legacy_verify()`.
* **Fusion honesty check:** with the fusion re-fitted per writer fold and
  scored on the held-out fold (`fit_fusion.py`): skilled AUC **0.886** / EER
  **19.3%**, random AUC **0.984** / EER **8.1%**. Benchmark rows below use the
  production coefficients (fitted on all 12 writers, 3 parameters), so treat
  the CV-refit numbers as the conservative estimate.

| harmonized condition (honest)     | legacy v2.0 | v3 (EXP-001) |
|-----------------------------------|-------------|--------------|
| 1:1 skilled AUC / EER             | 0.810 / 27.5% | **0.901 / 16.4%** |
| 1:1 random AUC / EER              | 0.978 / 8.5%  | **0.992 / 5.5%**  |
| CV genuine auto-accept            | 18%         | 29%          |
| CV skilled auto-accept (FAR)      | 1%          | 2%           |
| CV random rejected                | 90%         | 93%          |
| WD max-of-3 skilled AUC / EER     | 0.866 / 18.2% | **0.949 / 12.0%** |
| WD max-of-3 genuine auto-accept   | 54%         | **83%**      |
| WD max-of-3 skilled auto-accept   | 8%          | **4%**       |
| ms / pair                         | 7.6         | 37.6         |

| raw condition (scanner shortcut available) | legacy v2.0 | v3 |
|-------------------------------------|-------------|----|
| 1:1 skilled AUC / EER               | 0.877 / 19.3% | 0.868 / 20.5% |
| 1:1 random AUC / EER                | 0.978 / 9.6%  | 0.986 / 6.8%  |
| WD max-of-3 skilled EER             | 15.1%       | 16.7%         |

* **Reading the raw row:** legacy loses 8.2 EER points when the shortcut is
  removed; v3 does *better* on harmonized input than on raw, so it is not using
  the shortcut. Its raw-input loss is a robustness gap. *Hypothesis (not yet
  verified):* a single internal normalisation pass leaves residual paper
  speckle that the benchmark's extra pass removes; the random EER also
  degrades (5.5 → 6.8), which fits noise rather than leakage.
* Determinism: 2 in-process repeats × 3 processes → byte-identical JSON (sha256 5cde4c56…).
* Tests: 57/57 pass.
* **Decision: KEEP.** Honest skilled EER −11.1 points, random EER −3.0, WD skilled
  auto-accept halved, with a 4× runtime cost (still < 40 ms/pair).
* **Open issues → next targets:** (1) raw-vs-harmonized robustness gap;
  (2) `VerificationCalibration` thresholds are still on the legacy score scale
  and must be re-selected for the probability scale; (3) multi-reference
  aggregation (max-of-3 ≫ mean-of-3 for skilled: 12.0% vs 18.8% EER).

---

## EXP-002 — Two-pass photometric normalisation (raw-input robustness)

* **Problem (from EXP-001):** v3 scored worse on raw scans (skilled EER 20.5%)
  than on pre-normalised scans (16.4%). Raw production inputs are what matter.
* **Diagnosis:** ink masks were nearly identical between raw and pre-normalised
  inputs, but raw inputs produced ~2× the SIFT keypoints (e.g. 125 vs 58);
  sub-ink scan texture survived one pass and generated spurious keypoints.
* **Variants tested (writer-disjoint CV-refit, raw input, skilled AUC / EER):**

| variant | skilled | random |
|---|---|---|
| 1 pass (EXP-001) | 0.886 / 19.3% | 0.984 / 8.1% |
| ink smoothing σ 1.1 / 1.4 / 1.8 | 0.878 / 0.872 / 0.879, EER 19–22% | 0.98–0.99 |
| absolute floor 0.10 / 0.15 / 0.20 | 0.863 / 0.882 / 0.851 | 0.975–0.982 |
| post-rescale relative floor 0.1–0.3 | ≤ 0.886, EER ≥ 19.3% | ≤ 0.984 |
| **2 passes** | **0.915 / 13.5%** | **0.988 / 4.1%** |
| 3 passes | 0.938 / 12.3% | 0.992 / 2.7% |
| 4 passes | 0.926 / 13.5% | 0.992 / 4.1% |

* Rejected hypotheses: "more smoothing" and "texture amplified by the ink
  rescale" (relative floors) do not reproduce the gain. The exact mechanism of
  the second pass (paper re-estimate on a flattened image) is not fully
  isolated; the effect is a stable plateau from 2 to 4 passes. **2 passes chosen**
  as the smallest value on the plateau (3 vs 2 differs by ~1 skilled pair = noise;
  picking it would be tuning on noise).
* **Change:** `normalization.NORMALIZATION_PASSES = 2`; fusion refitted
  (`shape 24.925, keypoint 45.020, bias −21.696`).

| production path (raw input)    | EXP-001 | EXP-002 |
|--------------------------------|---------|---------|
| 1:1 skilled AUC / EER          | 0.868 / 20.5% | **0.916 / 15.2%** |
| 1:1 random AUC / EER           | 0.986 / 6.8%  | **0.992 / 4.1%**  |
| CV skilled rejected            | 29%     | **48%** |
| CV skilled auto-accept         | 2%      | 2%      |
| CV random rejected             | 89%     | **97%** |
| WD max-of-3 skilled EER        | 16.7%   | **12.0%** |
| WD max-of-3 skilled auto-accept| 3%      | 4%      |
| WD max-of-3 genuine auto-accept| 42%     | **81%** |

| harmonized condition (shortcut control) | EXP-001 | EXP-002 |
|---|---|---|
| 1:1 skilled AUC / EER | 0.901 / 16.4% | 0.933 / 12.3% |
| 1:1 random AUC / EER | 0.992 / 5.5% | 0.994 / 2.8% |
| WD max-of-3 skilled auto-accept | 4% | **14%** ⚠ |

* ⚠ **FAR warning:** under the harmonized condition the WD max-of-3 ACCEPT cut
  (set at the highest dev-fold impostor score) lets 14% of test-fold skilled
  forgeries through. With ~48 skilled trials per fold, a "max of dev
  impostors" cut is brittle. The production (raw) path stays at 4%. This is
  the explicit target of the threshold/aggregation iteration.
* Still not using the shortcut: raw (15.2%) remains *worse* than harmonized (12.3%).
* Determinism: byte-identical across 3 processes (sha256 8cef0ace…). Tests 57/57.
* **Decision: KEEP.** Production-path skilled EER −5.3 points, random −2.7.

---

## EXP-003 — Multi-reference aggregation (writer-dependent, 3 enrolled specimens)

* **Hypothesis:** averaging or max-ing the *fused* score across references wastes
  information; a genuine signature may match one specimen's layout and another's
  stroke detail. Taking each similarity's best value across references before
  fusion should separate skilled forgeries better.
* **Candidates (raw input, production fusion, CV zones on held-out writer fold):**

| aggregation | skilled AUC / EER | random EER | genuine auto-accept | skilled auto-accept | skilled rejected |
|---|---|---|---|---|---|
| mean of logits | 0.948 / 12.5% | 2.1% | 75% | 4% | 70% |
| max of logits (EXP-002 "max") | 0.955 / 12.0% | 2.6% | 81% | 4% | 82% |
| min of logits | 0.906 / 18.8% | 4.2% | 67% | 4% | 52% |
| median of logits | 0.949 / 10.4% | 2.0% | 73% | 4% | 71% |
| top-2 mean of logits | 0.958 / 10.4% | 2.1% | 77% | 4% | 82% |
| **feature-wise max, then fuse** | **0.968 / 8.3%** | **2.1%** | **81%** | **3%** | **80%** |
| cohort z-norm (max) | 0.974 / 8.3% | 4.4% | 56% | 1% | 82% |

* Fold check (selection rule: must win in both writer folds): feature-wise max
  skilled AUC 0.997 / 0.944 vs max-of-logits 0.975 / 0.936. z-norm was more
  fold-stable on skilled pairs (0.974 / 0.973) but doubles random EER and cuts
  genuine auto-accept to 56%, so it was not chosen.
* **Change:** `similarity.compare_multi()` + `DeterministicVerifier.verify_against_references()`;
  unusable (no-ink) references are skipped; all-unusable → INCONCLUSIVE.
  New tests `tests/test_multi_reference.py` (ordering invariance, determinism,
  blank references, empty list).
* **Result (benchmark `WD[system]` row):** skilled AUC 0.955 → **0.968**, EER 12.0% →
  **8.3%**, random EER 2.6% → 2.1%, skilled auto-accept 4% → 3%, genuine auto-accept
  unchanged (81%). 1:1 metrics unchanged (single-reference path untouched).
* Caveat: 48 genuine WD trials, so one trial = 2.1 points; the AUC gain is the
  more reliable signal.
* Determinism: byte-identical across 3 processes. Tests 63/63.
* **Decision: KEEP.**

---

## EXP-004 — Validated ACCEPT / REVIEW / REJECT operating points (FAR control)

* **Problems found:**
  1. API tiers still used legacy score cut-offs (0.78 / 0.52 / 0.45) that mean
     nothing on the v3 probability scale.
  2. The 4-decimal rounded probability saturates: in one CV fold the ACCEPT
     cut became 1.000000001 (a random pair scored 1.0000), so nothing could be
     auto-accepted, while the raw probability is over-confident (45% of skilled
     forgeries ≥ 0.5 under balanced priors).
  3. "Accept above the highest dev impostor" let 3/96 skilled forgeries through
     on held-out writers (3-specimen mode).
* **Change:** decisions now use the unrounded fused **logit**
  (`VerificationResult.match_logit`, `decision_band`, `reference_count`);
  `src/adjudication/thresholds.classify_band()` (pure) maps logit → band with
  separate single- and multi-specimen operating points in `DecisionThresholds`;
  `DecisionEngine` routes ACCEPT→GREEN-eligible, REVIEW→AMBER, REJECT→RED
  (below lowest genuine seen → `SIG_DIFF_01`), INCONCLUSIVE→RED four-eyes.
  Legacy probability cut-offs remain only for results that carry no logit.
  Values produced by `benchmark/select_thresholds.py` (order statistics only).
* **Rule candidates (writer-disjoint CV, held-out fold counts):**

| rule (3 specimens) | genuine auto-accept | skilled auto-accept | random auto-accept |
|---|---|---|---|
| max dev impostor | 81% | 3/96 | 0/960 |
| **max dev impostor + 1.0 logit** | **71%** | **0/96** | **0/960** |
| max dev impostor + 2.0 logit | 58% | 0/96 | 0/960 |
| Gaussian tail of dev skilled (z = 3.09) | 48% | 0/96 | 0/960 |

| rule (1 specimen) | genuine auto-accept | skilled auto-accept | random auto-accept |
|---|---|---|---|
| max dev impostor + 1.0 | 23% | 2/98 | 2/482 |
| Gaussian tail (z = 3.09) | 22% | 1/98 | 1/482 |

  Margin 1.0 = smallest tested margin with zero observed false accepts in
  3-specimen CV. Note the margin was chosen by looking at CV results (mild
  selection on held-out data); with 0/96, the rule-of-three 95% upper bound on
  skilled auto-accept is ≈ 3.1%.
* **Production operating points (all writers):** multi accept 6.115 / reject
  1.559 / hard-reject 0.740; single accept 13.613 / reject −0.240 / hard −2.586.
* **Benchmark (raw, `exp004_raw.json`), 3-specimen system:** genuine auto-accept
  71%, genuine rejected 12%, skilled auto-accept **0%** (was 3%), skilled rejected
  80%, random rejected 100%. Harmonized control: skilled auto-accept 0% (EXP-002 ⚠
  was 14%), genuine auto-accept 88%.
* **Single-specimen finding:** no dev-selected rule reaches zero skilled false
  accepts with one specimen. Policy consequence: single-specimen accounts are
  effectively REVIEW-by-default (accept logit 13.6); collect ≥ 3 specimens to
  enable straight-through processing.
* **Benchmark speed:** current-system scoring now caches features per image →
  3 repeats in 8 s (was ~9 min). `ms_per_pair` now excludes feature extraction
  (~35 ms per image, done once).
* Determinism: byte-identical across 3 processes (sha256 a98f39c1…). Tests 63/63.
* **Decision: KEEP.**
* ⚠ **Found while inspecting hardest cases:** the top-scoring impostors are the
  synthetic images, which are white strokes on a *black* background. The
  normaliser assumes dark ink on light paper, so it treats the background as ink.
  The single-specimen accept cut (13.6) is set by that synthetic pair → EXP-005.

---

## EXP-005 — Ink polarity normalisation (correctness fix) + near-identical look-alike finding

* **Bug:** inputs with light ink on a dark background (negative scans,
  white-on-black renders; here, all 6 synthetic files) were normalised as if the
  background were ink.
* **Change:** `normalization.ensure_dark_ink()` flips images whose median pixel
  is < 128 (the background dominates a signature crop). The flip is recorded in
  `NormalizedSignature.polarity_inverted` for the explanation layer. Fusion
  refitted (24.788 / 44.206 / −21.544); thresholds re-selected (multi accept
  6.037 / reject 1.561 / hard 0.750; single accept 15.932 / reject −0.265 / hard −2.584).
* Detector flags exactly the 6 synthetic files and none of the 70 CEDAR files.
* **Result:** CEDAR metrics unchanged within noise (1:1 skilled AUC 0.917 / EER
  15.2%; 3-specimen skilled AUC 0.967 / EER 8.3%, auto-accept 0/96). An inverted
  copy of a query now yields a bit-identical logit (new tests in
  `tests/test_multi_reference.py`).
* **Adversarial near-match surfaced (requirement E):** synthetic writer B
  (different author, deliberately the same layout with pointed instead of
  rounded humps) scores logit **14.93** against writer A, *higher* than A's own
  genuine pair (11.43). Current features (coarse gradient grid + corner-heavy
  SIFT on straight polylines) cannot separate this constructed look-alike. It is
  the pair that sets the single-specimen ACCEPT cut (15.93), which is why
  single-specimen auto-accept is effectively disabled. Not a CEDAR effect;
  logged as a known limitation, and synthetic polylines are not handwriting.
* Determinism: byte-identical across 3 processes. Tests 65/65.
* **Decision: KEEP** (correctness fix; no measured regression).

---

## EXP-006 — Robustness suite + non-local-means input denoising

* **New tool:** `benchmark/robustness.py` perturbs the *questioned* image of every
  3-specimen genuine/skilled trial (48 / 96) with 20 controlled transforms
  (rotation, scale, translation, blur, JPEG, brightness, contrast, additive
  noise, thicker pen) and re-scores at the production operating point.
  Noise is seeded (PCG64, constant + image SHA prefix). Output is byte-identical
  across processes.
* **Diagnosis (EXP-005 system):** forgeries are never lifted into ACCEPT (≤ 3%,
  one trial, under blur σ3 / thicker pen), so FAR control holds under
  perturbation. The worst genuine fragility is mild additive noise: σ=10 sends
  **33%** of genuine queries to REJECT, σ=25 sends **90%**.
* **Variants (CV-refit 1:1 skilled AUC / EER, random EER; genuine REJECT under noise σ10 / σ25):**

| variant | skilled | random EER | noise σ10 | noise σ25 |
|---|---|---|---|---|
| none (EXP-005) | 0.915 / 13.5% | 4.1% | 33% | 90% |
| median 3 / median 5 | 0.855 / 0.827 | 7.1% / 10.8% | 10% / 12% | 38% / 12% |
| Gaussian σ1 | 0.873 / 15.2% | 8.3% | 8% | 8% |
| bilateral | 0.880 / 21.7% | 6.8% | 8% | 6% |
| NLM h7 / h10 / h15 (every pass) | 0.904 / 0.886 / 0.884 | 5.4–6.8% | 4–8% | 4–58% |
| NLM h12 (every pass) | 0.914 / 16.4% | 3.9% | 6% | 10% |
| **NLM h12, once on raw input** | **0.910 / 15.2%** | **2.7%** | **10%** | **12%** |
| + contrast stretch before NLM (q 0.005 / 0.01) | 0.891 / 0.907 | 5.7% / 4.4% | 6–8% | 17–33% |

  The non-monotonic AUC across h (0.886 → 0.914 → 0.884) shows ≈ ±0.015 of
  selection noise on 12 writers. The variant was chosen on design grounds
  (denoise the sensor input once) with the best accuracy/robustness balance.
* **Change:** `normalization.DENOISE_STRENGTH = 12` (cv2.fastNlMeansDenoising on
  the raw grey image, before flat-fielding); fusion refitted (22.852 / 41.984 /
  −19.939); thresholds re-selected (multi accept 5.444 / reject 1.362 / hard 0.848;
  single accept 15.073 / reject −0.375 / hard −3.567).

| benchmark (raw)                        | EXP-005 | EXP-006 |
|----------------------------------------|---------|---------|
| 1:1 skilled AUC / EER                  | 0.917 / 15.2% | 0.915 / 15.2% |
| 1:1 random AUC / EER                   | 0.992 / 4.1%  | 0.991 / 3.9%  |
| 3-spec skilled AUC / EER               | 0.967 / 8.3%  | 0.956 / 13.0% ⚠ |
| 3-spec CV genuine auto-accept          | 71%     | **85%** |
| 3-spec CV genuine rejected             | 12%     | 10%     |
| 3-spec CV skilled auto-accept          | 0/96    | 0/96    |
| 3-spec CV skilled rejected             | 80%     | 77%     |

| robustness (3-spec, production point): genuine ACCEPT / REJECT | EXP-005 | EXP-006 |
|---|---|---|
| no perturbation | 52% / 6% | 65% / 6% |
| noise σ10 | 8% / 33% | **56% / 8%** |
| noise σ25 | 0% / 90% | **23% / 12%** |
| JPEG q10 | 6% / 12% | **40% / 10%** |
| rotation −10° | 19% / 12% | 42% / 10% |
| contrast ×0.5 | 40% / 12% | 25% / 12% ⚠ |
| contrast ×0.3 | 10% / 12% | 2% / **62%**, 8 extraction failures ⚠ |
| blur σ3 | 8% / 15% | 6% / **33%** ⚠ |
| skilled ACCEPT, any perturbation | ≤ 3% | ≤ 3% |

* **Trade-off, stated plainly:** 3-specimen skilled EER worsens 8.3% → 13.0%
  (≈ 2 of 48 genuine trials; AUC −0.011, inside the ±0.015 selection noise),
  and very-low-contrast / heavy-blur inputs degrade, because the absolute NLM
  strength erases faint ink. In exchange the operating point improves (genuine
  auto-accept 71% → 85% at 0/96 forgeries) and realistic scanner noise / JPEG
  no longer cause false rejects.
* **Decision: KEEP**, with contrast ×0.3 and blur σ3 to be handled by the quality
  gate (INCONCLUSIVE instead of a misleading REJECT). A contrast stretch would fix
  them inside the verifier but costs normal-image accuracy (random EER 2.7 → 4.4–5.7%).
* Determinism: benchmark and robustness JSON byte-identical across 3 processes.
  Tests 65/65.

---

## EXP-007 — Signature quality gate (INCONCLUSIVE instead of a misleading verdict)

* **Hypothesis:** the residual robustness failures from EXP-006 (heavy blur,
  washed-out ink) are out-of-distribution inputs. A verifier should refuse to
  score them rather than emit a confident REJECT (false reject) or, worse, ACCEPT.
* **Signals (`src/preprocessing/quality.py`, all contrast-normalised, deterministic):**

| signal | clean CEDAR (70 imgs) | blur σ2 | blur σ3 | contrast ×0.3 | noise σ25 |
|---|---|---|---|---|---|
| ink contrast (paper median − ink p1) | min 70 | 34–60 | 29–49 | 21–35 | 62–114 |
| edge sharpness (p99.5 ∇ / contrast) | min 3.89 | 1.78–2.05 | 1.15–1.40 | 3.83 | 3.94 |
| noise ratio (paper σ / contrast) | max 0.020 | 0 | 0 | 0 | 0.106–0.16 |

* **Gate (blocking):** contrast < 40, sharpness < 1.6, noise ratio > 0.15,
  ink < 150 px, ink box < 32×16 px, empty/invalid image. **Warnings:** noise
  ratio > 0.06, ink touching the border (possibly cropped), polarity inverted.
  Thresholds sit at ≈ 40–60% of the clean-data minimum; they were set from the
  clean-image distribution, not from identity labels.
* **Integration:** both questioned and reference images are gated. Single
  mode: either failing → INCONCLUSIVE. Multi mode: failing references are dropped,
  a failing questioned image → INCONCLUSIVE. `VerificationResult.quality`
  carries every measurement. `DecisionEngine` routes INCONCLUSIVE → RED four-eyes,
  `SIG_IRREG_03`, flag `SIGNATURE_VERIFICATION_INCONCLUSIVE`.
* **Gate firing (all 76 images):** clean 0/76; rotation, scale, translation, JPEG,
  noise σ10, thicker pen 0/76; blur σ2 4/76; contrast ×0.5 5/76; noise σ25 3/76;
  contrast ×0.3 46/76; blur σ3 72/76.

| robustness, genuine queries (3-spec) | REJECT before → after | INCONCLUSIVE after |
|---|---|---|
| clean | 6% → 6% | 0% |
| blur σ3 | **33% → 0%** | 100% |
| contrast ×0.3 | **62% → 8%** | 77% |
| contrast ×0.5 | 12% → 10% | 8% |
| noise σ25 | 12% → 12% | 6% |

* Skilled forgeries: ACCEPT stays 0% everywhere except blur σ3 (3%, 1 of 96
  trials, a forgery query that passed the gate); forgeries that hit the gate
  become INCONCLUSIVE (four-eyes), never ACCEPT.
* Main benchmark unchanged (no dataset image is gated).
* Test change: `test_verification_raw_and_calibrated_scores` previously verified a
  single 4-px straight line against itself and expected 1.0. The gate now
  correctly rejects that as SIGNATURE_TOO_SMALL; the test asserts that and uses
  a curved scribble for the identity check. New tests in `TestQualityGate`.
* Determinism: robustness JSON byte-identical across 3 processes. Tests 70/70.
* **Decision: KEEP.**

---

## EXP-008 — Evidence-based explanations (requirement A)

* **Goal:** replace generic notes with statements that are each a measured
  value against empirical reference distributions; no free-text claims.
* **Reference data:** `benchmark/fit_evidence_reference.py` →
  `src/verification/evidence_reference.json`: quantiles of each signal over 73
  genuine / 96 skilled CEDAR 1:1 pairs, the signal's genuine-vs-skilled AUC, and
  each decision band's composition on held-out writers (from
  `select_thresholds.py` CV).

| signal | AUC (genuine vs skilled) | role |
|---|---|---|
| keypoint similarity (fine stroke detail) | 0.893 | evidence |
| geometrically consistent keypoints (inliers) | 0.885 | evidence |
| layout / stroke-direction similarity | 0.830 | evidence |
| proportions (aspect agreement) | 0.710 | observation only |
| ink-density agreement | 0.682 | observation only |

* **Rules:** CONSISTENT ≥ genuine p25; INCONSISTENT ≤ skilled median; else
  BORDERLINE. AUC < 0.80 signals are never shown as evidence.
  "Confidence" = empirical band reliability, e.g. 3-specimen ACCEPT: 41 held-out
  trials, 41 genuine / 0 skilled / 0 other writer; REJECT: 1036 trials, 5 genuine.
  The balanced-prior probability is deliberately not displayed (over-confident,
  EXP-004). INCONCLUSIVE results list quality failures only, with no evidence and no
  preprocessing claims. Polarity correction is reported only when it happened.
  Rotation correction is **not** claimed, because the pipeline does not do it.
* **Change:** `src/verification/explanation.py` (`build_explanation`,
  `render_text`), `similarity.explanation_signals`, `VerificationResult.explanation`.
  Multi-specimen explanations use each signal's best specimen (mirrors the
  feature-wise max aggregation).
* **Coherence check (all 2256 3-specimen trials):** ACCEPT 30/31 with all evidence
  CONSISTENT (1 shows an INCONSISTENT signal, surfaced honestly); REJECT 2097/2122
  with ≥ 1 INCONSISTENT, 25 mixed, 0 all-consistent; REVIEW is mixed as expected.
* No scoring change → benchmark metrics unchanged. Tests 74/74 (4 new).
* **Decision: KEEP.**

---

## EXP-009 — End-to-end cheque pipeline, cheque-path fidelity, form-rule cleaning

* **Context:** the sample cheques have no enrolled specimens, so the cheque path
  could not be evaluated. (`cheques/cheque_workspace_example.png` is not a
  cheque at all: it is a screenshot of the "Example Domain" web page, mislabelled
  in the manifest.) New tooling:
  * `src/pipeline.py::ChequeVerificationPipeline`: one orchestrator for
    IQA → locate → crop → form cleaning → quality gate → verify → band →
    policy → audit, returning a stage-by-stage report.
  * `scripts/demo_showcase.py`: judge demo. A real CEDAR questioned scan is laid
    into the signing box of `cheque_procedural_pantograph_1001.png` (original
    pixels, multiplicative ink blend, native scale unless too large); every
    stage after composition is the live pipeline. It writes a storyboard PNG per
    scenario. The composition is disclosed on screen and in the PNG.
  * `benchmark/cheque_path.py`: composes all 48 genuine + 96 skilled 3-specimen
    queries onto the cheque and compares the pipeline verdict with verifying the
    clean scan directly ("cheque-path fidelity").
* **Diagnosis:** through the cheque, genuine REJECT rose from 6% (direct) to 23%,
  mean log-odds shift −2.02. Crops contained the printed signatory rule and the
  dashed signing-box border, which the verifier matches as stroke features.
* **Variants:**

| cheque path | genuine A / X / INC | genuine shift | skilled agreement | skilled A |
|---|---|---|---|---|
| detector crop | 50% / 23% / 0% | −2.02 | 0.83 | 0% |
| + grow box to touched ink (iterative) | 42% / 35% / 0% | −3.01 | 0.78 | 0% |
| + grow box (single pass, no chaining) | 42% / 31% / 0% | −2.86 | 0.78 | 0% |
| + printed-rule cleaning only | 56% / 17% / 4% | −1.35 | 0.84 | 0% |
| **+ cleaning + single-pass growth** | **54% / 12% / 4%** | **−0.97** | **0.90** | **0%** |
| direct scan (upper bound) | 65% / 6% / — | 0 | — | 0% |

  The iterative growth chained along the dashed border and swallowed the whole
  signing area (1008×440 px), so chaining was removed.
* **Change:** `preprocessing/form_cleaning.clean_printed_rules` (descender-safe
  removal of rules ≥ 25% of crop width, plus flat dash components ≤ 5 px tall and
  ≥ 3× wider than tall); `locator.refine_signature_bbox` (single pass). Both on
  by default in the pipeline.
* **Residual gap / known limits:** the 4% INCONCLUSIVE is one writer whose long
  diagonal signature had to be shrunk ~2× to fit the box; the gate flags it.
  Scale ×0.5 remains a verifier weakness (EXP-006 robustness: 6% genuine
  ACCEPT). Composed cheques are an approximation of real cheque capture; real
  cheques with enrolled specimens are needed to measure this properly.
* **Engine wording:** decision reasons now cite log-odds and band instead of the
  over-confident probability; demo panels show ICCS return codes (scenario C is
  `IQA_REJECT_06`, i.e. rescan, not a signature verdict).
* Determinism: cheque-path JSON byte-identical across processes. Tests 82/82
  (8 new in `tests/test_pipeline.py`).
* **Decision: KEEP.**

---

## EXP-010 — API / CLI on the single pipeline + input hardening (engineering, no scoring change)

* **API** `/api/v1/cheque/process` now delegates to `ChequeVerificationPipeline`
  (previously duplicated IQA → locate → crop → verify → decide → audit inline),
  so it gets crop refinement, form cleaning, the quality gate, multi-specimen
  verification and explanations. New request field `additional_specimens`
  (≤ 9, JSON base64 list or multipart files); response adds `stages` and `crop_bbox`.
* **CLI** `demo_cli.py cheque --specimen A.png [B.png C.png]` runs the same
  pipeline (multi-specimen) and prints the evidence-based explanation; new
  `demo_cli.py demo --out DIR` runs the judge showcase.
* **Security fixes (pre-existing issues found while integrating):**
  * `resolve_image()` read any server filesystem path a client sent as an image
    string (local file read / path traversal). Removed; only uploads and
    base64 are accepted. Test: `test_server_file_path_is_not_read`.
  * No input size limits. Added `MAX_IMAGE_BYTES` (20 MB), a base64 length cap,
    a decoded-pixel cap (`MAX_IMAGE_PIXELS` = 50 M via a 1/8-scale header
    decode), and bounded multipart reads. Test: `test_oversized_payload_rejected` → HTTP 413.
* Tests 86/86 (4 new API tests). Code-review and security-audit agents run
  on all session changes (findings recorded below once received).

### EXP-010 review gate results (code-reviewer + security-auditor agents)

**Security audit:** WARN for a localhost demo; REJECT for production until auth is added. Fixed now:
* Pixel guard was ineffective (OpenCV fully decodes PNG/TIFF even with
  `IMREAD_REDUCED_*`). Now Pillow reads width/height from the header only
  (`Image.MAX_IMAGE_PIXELS` = 50 M), formats are restricted to PNG/JPEG/TIFF/BMP,
  and `OPENCV_IO_MAX_IMAGE_PIXELS` is set as a second line of defence. Test: a
  64 MP PNG under 2 MB is rejected with 413 before decode.
* Whole-request `Content-Length` cap (64 MB) middleware; bounded upload reads in `/iqa`.
* CPU-bound verification / pipeline / IQA run via `run_in_threadpool` (no event-loop blocking).
* JSON bodies are validated through Pydantic models (`allow_inf_nan=False`,
  `amount ≥ 0`, currency `^[A-Z]{3}$`, bounded text fields, ≤ 9 extra specimens).
  Form inputs get the same checks. Non-object JSON gives 400 (was 500). Tests added.
* Fixed error messages (no exception text echoed); `ValueError` from the pipeline gives 400.
* CORS: explicit allow-list from `SIGVERIFY_CORS_ORIGINS`, `allow_credentials=False`,
  methods GET/POST only.
* `account_no` is masked to its last 4 digits before entering the immutable audit
  chain. Test added.
* Not fixed (production work, documented): authentication / authorisation /
  rate limiting; caller-asserted CAR-LAR / positive-pay / stale flags; HMAC-keyed
  hash chain with an external anchor, append-only triggers and contiguous-sequence
  checks; multi-process ledger write locking; pinned lockfile + `pip-audit`;
  sandboxed image decoding.

**Code review:** WARN. Fixed now:
* H1: the ACCEPT note claimed "no impostor scored this high", which is true only
  in-sample (single-specimen CV ACCEPT contained 2 skilled + 2 random). Reworded
  to point to the band reliability counts.
* H2: band-reliability text now states its basis (2-fold, 6 writers per fold,
  per-fold cut-offs, same-fold random pairs, small sample).
* H3: `detected_signers_count` was the number of detector *proposals* (could
  satisfy a joint-mandate check with noise). Now 1 (only one signature is verified).
* H4: a fallback zone crop (no signature located) could reach ACCEPT/GREEN. It is
  now capped at REVIEW; `DecisionEngine` honours upstream bands that are *more*
  conservative than the logit band (never less).
* M5: `RepresentationParams` now flow into `compare` / `compare_multi` (RANSAC and
  ratio test previously always used defaults); duplicate RANSAC run removed in `verify`.
* M7: SIFT detector created per call (no shared mutable singleton);
  `cv2.setNumThreads(1)` set once at import rather than lazily on first use.
* M8: skipped specimens are noted, including the switch to single-specimen thresholds.
* M10 / M11: reference-quality failures are rendered per specimen; an unknown
  questioned status shows "?" instead of "✗"; multi-specimen evidence is labelled
  "best value per signal".
* M12: polarity is decided by dark-tail vs light-tail extent instead of
  `median < 128` (robust to under-exposed grey paper). Same 6 synthetic files flagged.
* M13: `to_gray` accepts BGRA / uint16 / 1-channel and raises `ValueError` for anything else.
* M14: `is_match` is now true **only** for ACCEPT (REVIEW is not a match).
* **Regression check:** benchmark scores byte-identical to EXP-006 (score sha
  aacdd42cf251; full JSON identical except the repeat-count field). Tests 90/90.
* **Deferred (each changes scoring, so each needs its own benchmarked experiment):**
  M6 `keypoint_similarity` gives credit to unverified matches (< 4 good) on a
  different normalisation; M18 ACCEPT cut-off from a single maximum impostor
  (outlier statistic) and in-sample evidence percentiles; plus the LOW typing/style items.

---

## EXP-011 — No similarity credit for geometrically unverified keypoint matches (review M6)

* **Issue:** with fewer than 4 ratio-test matches, `keypoint_similarity` returned
  `good / min(na, nb)` (no RANSAC, different normalisation), so a sparse
  signature with 3 chance matches could score up to 0.75. That is above every
  genuine median (0.091).
* **Variants (CV-refit; 1:1 skilled AUC / EER, random EER | 3-spec skilled AUC / EER, random EER):**

| variant | 1:1 | 3-specimen |
|---|---|---|
| current | 0.910 / 15.2%, 2.7% | 0.952 / 10.4%, 3.7% |
| **unverified → 0** | **0.910 / 15.2%, 2.7%** | **0.952 / 10.4%, 3.7%** |
| inliers − 2 (chance-corrected) | 0.909 / 16.4%, 2.7% | 0.953 / 10.9%, 2.5% |
| inliers − 3 | 0.901 / 17.6%, 2.7% | 0.949 / 10.4%, 3.8% |

* The path fires on ~10% of pairs (165 of 1560 among the first 40 images),
  almost all low-scoring impostor pairs, so ranking metrics are unchanged. The fix is
  a correctness guard for sparse or small signatures outside this dataset.
  Chance correction does not help, so it is not adopted.
* **Change:** `similarity.keypoint_similarity` returns 0.0 when no geometric
  verification is possible. Fusion refitted (22.788 / 42.060 / −19.890);
  thresholds re-selected (multi accept 5.465 / reject 1.369 / hard 0.849; single
  15.087 / −0.371 / −3.553); evidence reference regenerated.
* **Benchmark (raw):** 1:1 skilled AUC 0.9145 → 0.9147, EER 15.19% (=); random
  AUC 0.9909 (=); 3-spec skilled AUC 0.9559 → 0.9564, EER 13.0% (=); CV
  genuine auto-accept 85%, skilled auto-accept 0/96 (=). Demo outcomes unchanged.
* Determinism: byte-identical across processes. Tests 90/90.
* **Decision: KEEP** (principled guard, no measured regression).

---

## EXP-012 — Stability of the ACCEPT cut-off rule (review M18) — analysis, no change

* **Concern:** ACCEPT = (max impostor logit + 1) relies on one outlier sample.
* **Method:** leave-one-writer-out over the 12 CEDAR writers (cut-off selected
  on 11 writers, counted on the held-out writer), 3-specimen trials. This is a
  finer split than the 2-fold CV used for selection.

| rule | genuine auto-accept | skilled FA | random FA | cut-off mean / sd / range (logit) |
|---|---|---|---|---|
| max impostor + 0 | 81% (39/48) | 0/96 | **3/2112** | 4.35 / 0.27 / [3.75, 4.46] |
| **max impostor + 1 (production)** | 62% (30/48) | 0/96 | 0/2112 | 5.35 / 0.27 / [4.75, 5.46] |
| max impostor + 2 | 52% | 0/96 | 0/2112 | 6.35 / 0.27 |
| Gaussian tail of skilled, z = 3.09 | 58% | 0/96 | 0/2112 | 5.78 / 0.20 |
| mean of top-5 impostors + 1 | 71% | 0/96 | 0/2112 | 4.96 / 0.22 |
| max skilled only + 1 | 81% | 0/96 | 2/2112 | 4.13 / 0.15 |

* **Findings:** the production rule is stable (sd 0.27 logit across folds;
  removing any one writer moves it by at most 0.6). A margin is necessary: with
  no margin, 3 random-writer trials are auto-accepted. Rules that ignore random
  impostors also leak (2/2112). "Top-5 mean + 1" buys +9 pts genuine
  auto-accept, but sits closer to where false accepts appear. With 0 events in
  every safe row, this data cannot rank their FAR (rule-of-three bound ≈ 3.1% for
  skilled for all of them).
* **Decision: NO CHANGE** (FAR control outranks FRR). Choosing top-5 mean + 1 is
  a risk-appetite decision for the bank, documented as an option.

---

## EXP-013 — Single determinism control point + documentation (requirement)

* `src/core/determinism.py`: `GLOBAL_SEED`, `OPENCV_THREADS`,
  `configure_determinism()` (applied at import of the feature extractor),
  `seeded_rng(offset)` (the only RNG; used by robustness noise, same values as before).
* `DETERMINISM.md`: inventory of every potential nondeterminism source and how it
  is handled. Only audit metadata (timestamps, uuid4 record ids) is intentionally
  non-deterministic, and it never feeds scores or decisions.
* New test `TestCrossProcessDeterminism`: 3 fresh interpreters give an identical
  `VerificationResult` JSON. Tests 91/91.
* Robustness JSON byte-identical across processes (`robustness_exp011.json`;
  values reflect the EXP-011 thresholds, e.g. clean genuine ACCEPT 62%).

---

## EXP-014 — RANSAC-aligned layout similarity (rotation / scale robustness)

* **Hypothesis:** the coarse gradient-grid layout descriptor is not rotation
  invariant, so slanted genuine signatures lose layout similarity (robustness:
  +15° rotation sent 31% of genuine queries to REJECT). The keypoint stage
  already estimates a specimen→query similarity transform; warping the specimen
  by it before computing layout similarity should remove that penalty.
* **Design fixed before measuring:** use alignment only with ≥ 6 RANSAC inliers,
  |rotation| ≤ 25°, scale 0.6–1.6; shape = max(unaligned, aligned).
* **Variants (CV-refit 1:1 | 3-spec, fusion refit on all writers, threshold-free):**

| variant | 1:1 skilled AUC / EER, random EER | 3-spec skilled AUC / EER | +15° rotated genuine vs skilled AUC / EER |
|---|---|---|---|
| unaligned (EXP-011) | 0.910 / 15.2%, 2.7% | 0.956 / 13.0% | 0.798 / 33.9% |
| **aligned ≥6, max** | **0.927 / 14.0%, 2.7%** | **0.973 / 6.2%** | **0.945 / 10.4%** |
| aligned ≥6, replace | 0.918 / 15.2%, 4.0% | 0.972 / 6.2% | 0.945 / 10.4% |
| aligned ≥8, replace | 0.914 / 14.0%, 4.1% | 0.972 / 6.2% | 0.926 / 14.1% |
| aligned ≥10, max | 0.913 / 15.2%, 2.7% | 0.958 / 13.0% | 0.834 / 29.2% |

* **Change:** `similarity.aligned_shape()` + `Alignment`; `KeypointMatch.transform`;
  `compare()` uses max(unaligned, aligned); `RepresentationParams.align_*`. The
  fitting scripts now use `compare()`'s shape (same signal as production).
  Fusion refitted (22.948 / 38.455 / −20.058); thresholds re-selected (multi
  4.650 / 1.427 / 0.625; single 13.938 / −0.278 / −3.834); evidence reference
  regenerated (layout AUC 0.830 → 0.880).
* **Results (raw):** 1:1 skilled AUC 0.915 → **0.934**, EER 15.2% → **11.1%**;
  random EER 3.9% → **2.8%**; 3-spec skilled AUC 0.956 → **0.973**, EER 13.0% →
  **6.25%**; CV genuine auto-accept 85% → **88%**, genuine rejected 10% → 8%,
  skilled auto-accept **0/96**. Leave-one-writer-out: genuine auto-accept 62% → **75%**,
  0/96 skilled and 0/2112 random false accepts, cut-off sd 0.27 → 0.13.
  Robustness: clean genuine ACCEPT 62% → 75%; +15° ACCEPT/REJECT 25%/31% → 67%/10%;
  −10° 42% → 73%; JPEG q10 40% → 56%; skilled ACCEPT still 0% (blur σ3: 1 trial).
  Cheque path: genuine ACCEPT 54% → 65%, REJECT 12% → 10%; skilled ACCEPT 0%.
* **Explainability:** the measured relative rotation and scale are now reported
  ("Rotation corrected: −9.5° relative rotation, scale ×0.84, specimen #3"), but
  only when the aligned comparison supplied the score, never for INCONCLUSIVE.
* Cost: verify 137 → 165 ms (1 specimen), 271 → 317 ms (3 specimens).
* Determinism: byte-identical across processes. Tests 97/97 (2 new).
* **Decision: KEEP**: the largest gain since EXP-001, with no FAR regression.

### EXP-014 review gate (code-reviewer): no critical issues; NEEDS_MINOR_REVISIONS → addressed
* Verified correct by the reviewer: transform direction (specimen → query, same
  letter-boxed canvas as the keypoints), rotation/scale extraction, NaN/degenerate
  guards, no unused alignment ever claimed, determinism.
* **Item 1 (max(unaligned, aligned) may inflate impostor scores):** answered with
  measured FAR at fixed operating points. 3-specimen skilled auto-accept stays 0/96
  (2-fold CV and LOWO), random 0/960 and 0/2112; the 1:1 held-out skilled FAR at the
  dev-EER cut *fell* 16.3% → 11.2%. Stricter gates (≥ 8, ≥ 10 inliers) were measured
  in EXP-014 and were worse. Caveat kept: gate values were chosen a priori but
  evaluated on the same 12 writers, so treat the gain as optimistic until it is
  re-measured on more writers.
* Fixed: an explicit `PairSimilarity.alignment_used` flag (no float-equality
  re-derivation; scores byte-identical); the rotation is rendered as "N° clockwise /
  counter-clockwise" (image-coordinate convention documented) and the line is
  worded "Geometric alignment applied"; compare() asymmetry documented; 4 direct
  `aligned_shape` unit tests (rotation recovery, inlier gate, implausible/NaN
  transform, determinism). Confirmed the fusion fit, thresholds and evidence
  reference were regenerated after the scoring change.
* Not changed (low, documented): redundant per-pair recomputation in the
  verify paths (≈ 20–30 ms), the process-wide `cv2.setNumThreads(1)` side effect,
  and robustness noise keyed on the hex SHA prefix (protocol ids are always SHA hex).
* Tests 101/101.

---

## FEATURE-001 — Signature-only comparison mode (no scoring change)

* Requested: compare an original signature with a questioned one and return a
  similarity, without any cheque handling or clearing policy.
* `src/verification/signature_compare.compare_signatures(references, questioned)`
  → `SignatureComparison`: verdict (MATCH / UNCERTAIN - MANUAL REVIEW / NO MATCH /
  INCONCLUSIVE - IMAGE QUALITY), 0–100 similarity score, log-odds and thresholds,
  layout and stroke-detail similarity, alignment, explanation. 1–10 references.
  Same scorer and thresholds as the pipeline; no audit record, no amount/mandate rules.
* **Score scale:** a plain sigmoid of log-odds showed a confidently rejected
  forgery as "50.9/100" because the validated thresholds are not at 0. The score
  is therefore piecewise-linear through decision anchors (NO MATCH threshold → 40,
  MATCH threshold → 70, ±6 log-odds → 0/100). It is monotone and can never contradict
  the verdict: ≥ 70 MATCH, < 40 NO MATCH, 40–70 review.
* Entry points: `POST /api/v1/signature/compare` (JSON base64 or multipart; same
  input limits) and `demo_cli.py compare --reference A.png [B.png …] --questioned Q.png [--json]`.
* Sample: genuine (3 refs) MATCH 82.1; skilled forgery NO MATCH 30.7; different
  person NO MATCH 7.2; genuine with 1 ref UNCERTAIN 53.8 (single-specimen policy);
  blurred INCONCLUSIVE (no score). Tests 112/112 (11 new).


---

## EXP-015 — Stroke-level signals + 55-writer evaluation (1:1 accuracy, no training of a model)

* **Why:** the v3 verifier decided on two signals only (coarse layout cosine, SIFT inliers).
  On full CEDAR (55 writers) the 12-writer numbers did not hold: 1:1 skilled EER **16.5%**
  (not 11.1%), because fusion weights and thresholds had been fitted on 12 writers.
  A human examiner also looks at stroke direction, slant, rhythm and pen width; those were
  not used.
* **Data:** `benchmark/build_cedar_eval.py` builds two disjoint slices of full CEDAR
  (55 writers x 6 genuine + 6 skilled forgeries each): **dev** = images 1-6, **test** =
  images 13-18 (never used for any choice below; scored once). Random pairs are strided
  subsamples (`--max-random 6000`, `--max-random-per-query 40`); capped and uncapped runs
  agree within 0.1 point on zone rates.
* **Candidates measured** (single-signal AUC genuine-vs-skilled on dev, equalised scans):
  layout 0.877, banded-DTW horizontal ink profile 0.877, slant histogram EMD 0.868,
  keypoints 0.855, chamfer/ICP stroke distance 0.857, **local stroke-direction agreement
  after ICP alignment x coverage 0.919**, vertical profile 0.785, stroke width 0.698,
  topology counts 0.59-0.62 (junctions, endpoints, Euler number; dropped), finer gradient
  grids 0.82 (dropped), piecewise 3-part ICP 0.881 (dropped: no gain in fusion).
* **Fusion:** compact 6-signal set (keypoint, stroke_direction, slant, column_profile,
  row_profile, stroke_width), balanced L2 logistic, C = 0.1, writer-disjoint 2-fold CV on
  dev: skilled AUC 0.907 -> **0.945**, EER 15.5% -> **12.5%**; different-writer EER
  5.7% -> 2.6%. Greedy forward selection picked different 6-8 signals per condition and
  plateaued at AUC ~0.95, so a compact fixed set was chosen over the greedy optimum. The
  `layout` signal is still computed and reported but not fused (it received a negative
  weight; redundant with slant + direction).
* **Speed:** ICP on every 4th skeleton point, 6 iterations, two starting transforms
  (identity and the RANSAC keypoint transform; one start only: AUC 0.881 vs 0.916),
  DTW profiles resampled to 128 / 64 samples (no accuracy loss). A pair costs ~10 ms
  (was ~1 ms); a `compare_signatures` request costs ~370 ms vs ~340 ms (denoising dominates).
* **Result on the untouched test slice** (weights/thresholds fitted on dev only):

  | | old, equalised | **new, equalised** | old, raw | **new, raw** |
  |---|---|---|---|---|
  | 1:1 skilled EER | 16.5% | **9.3%** | 16.4% | **10.4%** |
  | 1:1 skilled AUC | 0.901 | **0.952** | 0.897 | **0.949** |
  | 1:1 different-writer EER | 5.6% | **3.8%** | 5.5% | **3.5%** |
  | 3-specimen skilled EER | 9.4% | **6.1%** | 12.7% | **6.4%** |

  Production thresholds (each system's own), ACCEPT / REVIEW / REJECT %, equalised scans:

  | | old 1 spec. | **new 1 spec.** | old 3 spec. | **new 3 spec.** |
  |---|---|---|---|---|
  | genuine | 0.4 / 92.4 / 7.3 | **15.9** / 78.9 / 5.2 | 72.7 / 23.3 / 3.9 | 29.4 / 65.2 / 5.5 |
  | skilled forgery | 0.0 / 50.6 / 49.4 | 0.0 / 26.2 / **73.8** | **1.5** / 40.2 / 58.3 | **0.0** / 7.7 / **92.3** |
  | different writer | 0.0 / 2.9 / 97.1 | 0.0 / 1.1 / 98.9 | 0.2 / 1.7 / 98.2 | 0.0 / 0.2 / 99.8 |

  Honest reading: with **one** specimen the new system is clearly better on every row.
  With **three** specimens the old thresholds accepted more genuine signatures (73%), but
  also 1.5% of skilled forgeries on unseen writers' images; the new thresholds follow the
  rule "ACCEPT = highest dev impostor logit + 1" on 55 writers, which is stricter, so fewer
  genuine are auto-accepted (29%) at 0% forgeries accepted and far more forgeries rejected.
  The rule makes ACCEPT a function of the single worst impostor and so grows stricter with
  more impostor pairs; a percentile-based rule is a business choice (see EXP-012).
* **Determinism:** unchanged guarantees (benchmark `deterministic=True`, new hermetic
  cross-process tests with synthetic signatures). New numerical steps: skeletonisation,
  KD-tree nearest neighbours, a 2x2 SVD, banded DP.
* **Review gate (code-reviewer + security-auditor):**
  * code-reviewer found a real bug: `slant = 1 - EMD` was unbounded below (down to -8; 44%
    of skilled and 77% of different-writer dev pairs were negative). The fit had learned a
    weight on that wide range. Fixed by dividing by the maximum EMD (9) - an exact affine
    map, compensated in the weights (1.877 -> 16.893, bias -24.969 -> -39.985); refitting
    reproduced weight 16.897 / bias -39.988 and identical thresholds. Regression tests added.
    Also: `PairSimilarity.signals` made required, named constants for magic numbers,
    immutable name map, direct unit tests for the Umeyama fit, DTW and ICP primitives.
  * security-auditor: PASS (low risk). Hardened: deterministic cap of 20,000 skeleton points
    (noise images skeletonised to up to 86k points; worst case then 10 specimens + noise was
    12.7 s per request), scipy upper bound, `FusionModel` validates its signal keys.
    Not done (production work, unchanged): per-request concurrency limit and rate limiting.
* **Not verified:** the data-dependent test files (`test_pipeline.py`, `test_api_and_cli.py`,
  `test_multi_reference.py`, `test_signature_compare.py`) are skipped without the repo's
  sample images, which cannot be reconstructed from the CEDAR download (they were resized).
  Their asserted outcomes (e.g. "genuine auto-clears") were tied to the old thresholds and
  must be re-checked when the samples are available. 12-writer figures elsewhere in this
  report are historical.
* Tests: 65 passed, 3 skipped (adds 28 hermetic tests in `tests/test_stroke_geometry.py`).

## EXP-016 — Isolate signature ink from printed rules and text (capture robustness)

* **Problem:** `normalize_signature` cropped to all ink, so a printed caption and form rule
  (`distractor_print`) set the crop (99% of val genuine pairs NO MATCH) and faint ruled
  lines stayed as 0.12-darkness stripes inside the ink map (`ruled_lines`).
* **Change:** `src/preprocessing/isolation.py`, applied to the darkness map before the crop.
  (1) 1-px-tall grayscale opening, length 0.5 x width (odd, zero-padded border): a run is
  removed if it is fainter than ink and <= 50% of it lies within 2 rows of ink, or if it is
  ink-dark and >= 60% of the ink components it touches. (2) Printed-text groups (dilation
  radius 2% of the diagonal): >= 6 glyphs, group height <= 1.5 x median glyph height, width
  >= 6 x height. Never removes all ink.
* **Rejected variants (dev, 660 images):** distance/mass clustering ("keep the main blob")
  cannot work: clean CEDAR has detached parts up to 4.3x the main part's height away with
  equal mass, while the caption sits 1.1-1.5x away. Unfiltered rule opening changed 19 clean
  images (straight underline flourishes); with the OpenCV default border, strokes touching
  the edge passed at half length. Purity + ink-contact filters bring this to 1/660 (a
  scanner border line on a forgery scan).
* **Triggers (dev):** rules removed 1/660 clean, 660/660 ruled_lines, 660/660 distractor;
  text dropped 0/660 clean, 660/660 distractor.
* **Dev clearance:** 5.03 -> 5.18 (R 0.430 -> 0.479); clean skilled EER 11.76% unchanged;
  distractor_print EER 26.8% -> 11.9% (genuine NOMATCH 98.7% -> 4.6%); ruled_lines
  12.1% -> 11.8% (genuine NOMATCH 7.3% -> 5.0%). Other conditions unchanged. No refit.
* **Val clearance (reported once):** 3.88 -> 3.99 (D 0.502, R 0.313, O 0.348); clean
  skilled EER 15.42%; distractor_print 31.4% -> 15.6% (genuine NOMATCH 99% -> 8.1%);
  ruled_lines 17.0% -> 15.9% (genuine NOMATCH 11.3% -> 7.8%).
* **Cost:** ~175 ms mean per extract on dev (proxy, 6 workers), within the prior budget.
* **Limits:** only perfectly horizontal rules; an ink-dark rule fused with the signature
  (signing across the line) is kept; vertical rules and stamps are not handled.

## EXP-017 — Stroke-quality signals: ink pressure pattern and contour curvature

* **Failure analysis (dev, harmonized, production fusion):** the hardest skilled forgeries
  (highest logits) are not fooled by one signal; they sit at the 25-50th genuine
  percentile on *every* shape signal (mean percentile over skilled pairs above the EER
  threshold: keypoint 0.28, direction 0.25, slant 0.33, column 0.23, row 0.32, width 0.48).
  Writers 23, 21, 24, 46, 01 hold most of them. Side-by-sides show the forger copied the
  outline well; what differs is stroke quality: the genuine strokes taper, thin and darken
  in writer-specific places, forgeries have uniform, blunt, rounder strokes. The hardest
  genuine pairs (writers 08, 45, 41, 14) are real intra-writer variation (an extra
  flourish, a shortened name); no signal fixes those.
* **Also found:** `fit_fusion.py`, `select_thresholds.py` and `fit_evidence_reference.py`
  read raw scans while the clearance score harmonizes. They now take `--condition`
  (default `harmonized`). Harmonized refit of the 6 old signals alone: CV skilled EER 12.37%.
* **Candidates** (dev single-signal AUC genuine-vs-skilled; fused = writer-disjoint 2-fold
  CV skilled EER when added to the 6 signals, base 12.37%, 3000 random pairs):

  | signal | AUC | fused EER | note |
  |---|---|---|---|
  | width-histogram EMD / width CV / thin fraction | 0.73 / 0.65 / 0.65 | 12.37 / 12.23 / 12.59 | no gain |
  | darkness-histogram EMD | 0.81 | 10.91 | **rejected: session shortcut** (same- vs cross-session AUC 0.609) |
  | darkness std ratio | 0.73 | 11.63 | session AUC 0.545, superseded |
  | contour curvature histogram EMD | 0.75 | 12.01 | session AUC 0.476; kept (joint gain) |
  | contour roughness / mean curvature | 0.67 / 0.61 | 12.23 / 12.34 | no gain |
  | aligned HOG 4x8 / aligned blurred-ink correlation | 0.87 / 0.88 | 12.97 / 12.62 | redundant with direction (negative weight) |
  | worst-2-cell direction / tight direction / ICP residual | 0.81 / 0.91 / 0.85 | 12.62 / 12.48 / 12.23 | redundant |
  | **pressure pattern** (rank corr. of darkness at ICP-matched points x coverage) | **0.907** | **11.40** | session AUC 0.508 |
  | width rank pattern | 0.885 | 11.98 | weaker twin of pressure |
  | **pressure + curvature** | | **10.55** | both folds improve (11.20/10.88 -> 9.74/9.37) |

  Session test: different-writer pairs only, genuine(w1)-genuine(w2) vs genuine(w1)-forgery(w2);
  the existing stroke_width and slant score 0.456 / 0.463 on it. Re-run on the shipped
  code (curvature on the 512x256 canvas, not the 1024x512 prototype): curvature 0.477,
  pressure_pattern 0.508.
* **Shipped:** `pressure_pattern` and `curvature` in `stroke_geometry.py` (one shared ICP
  alignment and correspondence for direction and pressure). Refit on harmonized dev
  (`--max-random 6000`): CV skilled AUC 0.958, EER **10.58%**, random EER 2.18%; all weights
  positive. Thresholds (`--max-random-per-query 40`): single accept 6.3645 / reject -0.0803;
  CV single-specimen genuine auto-accept 39.3% at 0 skilled accepted.
* **Dev clearance** (in-sample for weights/thresholds): 5.18 -> **5.94** (D 0.681, R 0.533,
  O 0.539); clean skilled EER 11.76% -> 10.08%, random 1.70%.
* **Val clearance (reported once):** 3.99 -> **4.98** (D 0.599, R 0.409, O 0.454, gate off).
  Clean skilled EER 15.42% -> **12.48%** (AUC 0.940), random 2.67%. Operating point:
  genuine MATCH 28.2 / REVIEW 63.8 / NOMATCH 8.0%; skilled MATCH 0.1 / NOMATCH 78.5%; random
  NOMATCH 99.9%. Condition EERs: tinted 12.7, shadow 13.7, ruled 12.6, large canvas 12.4,
  distractor 13.0, faint 13.6, scale 0.5 20.0, scale 2.0 17.7, rotate +20 15.4,
  rotate -15 17.8, phone 13.8.
* **Limits:** curvature is resolution-fragile: on dev, scale 0.5 / 2.0 EERs got worse with
  it (14.9 -> 15.7, 13.4 -> 14.2 vs the harmonized 6-signal refit) while all other
  conditions improved; pressure alone improved all 11. Val genuine NOMATCH is 8.0% (> the
  5% design target), and 20-27% under scale / rotation. Scale-robust curvature is the
  next step.
* **Cost:** serial, same 24 images: extraction 181 -> 179 ms, compare 7.7 -> 7.7 ms per pair
  (no measurable change; the extra work is ~1 ms each).

## EXP-018 — Thresholds from out-of-fold logits (decision D-005)

* **Problem:** the fusion is fitted on the dev pairs, so dev logits from the production
  fusion are in-sample and over-separated. REJECT at their 5th genuine percentile became
  8.0% genuine NOMATCH on val; ACCEPT (max impostor + 1.0) hung on one extreme value.
* **Method (dev only):** `select_thresholds.py` refits the fusion (`fit_fusion._fit`, same
  pair set `--max-random 6000`; full-dev refit reproduces the config weights exactly) on one
  writer fold and scores the other -> out-of-fold (OOF) logits for every within-fold pair
  (cross-fold random pairs dropped). Multi-specimen trials: the same fold fits on
  max-aggregated signals. Rule estimate = **nested**: rule selected on inner OOF logits of the
  training fold (split again by writer), counted on the test fold scored by the fold fit.
  Fusion and cut-offs never see the test writers.
* **Finding:** pooled-OOF rates are tautological (the rule hits its own quantile). Nested
  skilled MATCH runs above the target because a fit on more writers has a larger weight norm
  (full 43.6, folds 34.3 / 42.9, inner 26-44): thresholds read from a smaller fit are lenient
  for the bigger one. The margin absorbs this gap and is sized on the nested check.
* **Single-specimen candidates** (dev, n = 825 genuine / 1980 skilled / 2931 random;
  nested held-out; reject = q5 genuine OOF in all rows; random MATCH 0 in all rows):

  | accept rule | accept | gen MATCH | gen NOMATCH | skl MATCH | skl NOMATCH | O-like |
  |---|---|---|---|---|---|---|
  | max impostor + 1.0 (old rule, OOF) | 6.473 | 49.3% | 3.9% | 0.15% | 68.3% | 0.549 |
  | q99.5 skilled + 0 | 4.038 | 71.4% | 3.9% | 1.26% | 68.3% | 0.660 |
  | q99.5 skilled + 0.5 | 4.538 | 64.4% | 3.9% | 0.91% | 68.3% | 0.624 |
  | q99.75 skilled + 0.5 | 5.012 | 62.6% | 3.9% | 0.61% | 68.3% | 0.615 |
  | q99.9 skilled + 0.75 | 5.850 | 54.8% | 3.9% | 0.30% | 68.3% | 0.577 |
  | **q99.75 skilled + 1.0** | **5.512** | **54.9%** | **3.9%** | **0.30%** | **68.3%** | **0.577** |
  | q99.5 skilled + 1.25 | 5.288 | 54.8% | 3.9% | 0.30% | 68.3% | 0.577 |

  The random term (q99.9) never binds. Reject q4 instead of q5: nested genuine NOMATCH
  3.2%, O-like 0.620 vs 0.645 (at q99.75 + 0), so q5 kept. Per outer fold the chosen rule
  gives skilled MATCH 1/1008 and 5/972, genuine NOMATCH 22/420 (5.2%) and 10/405.
* **Multi-specimen** (n = 330 / 1980 / 6480): even the old max-impostor + 1.0 rule gives
  nested skilled MATCH 0.81% on OOF logits (signal-wise max amplifies the scale gap); the
  same quantiles need margin 1.5: q99.75 skilled + 1.5 -> genuine MATCH 64.8%, NOMATCH 3.9%,
  skilled MATCH 0.35%, random 0%.
* **Shipped rule:** accept = max(q99.75 skilled OOF, q99.9 random OOF) + margin (single 1.0,
  multi 1.5); reject = q5 genuine OOF; hard_reject = min genuine OOF. Thresholds old -> new:
  single accept 6.3645 -> **5.5123**, reject -0.0803 -> **-0.6729**, hard -4.1280 -> -5.5863;
  multi accept 8.5690 -> **8.3837**, reject 3.0381 -> **2.2261**, hard 0.5526 -> 0.0533.
  `evidence_reference.json` band reliability now holds the nested counts (signal quantiles
  unchanged). Fusion weights and features unchanged.
* **Val clearance (reported once):** 4.98 -> **5.06** (D 0.599, R 0.409, O 0.454 -> **0.479**,
  gate off). Operating point: genuine MATCH 28.2 -> 36.8 / REVIEW 58.1 / NOMATCH 8.0 -> 5.1%;
  skilled MATCH 0.1 -> 0.3 / NOMATCH 78.5 -> 69.1%; random MATCH 0.0 / NOMATCH 99.4%. No
  condition exceeds 0.5% skilled MATCH (max faint_ink 0.4%). Genuine NOMATCH remains
  15.9-20.6% under scale / rotation.
* **Limits:** dev has only ~10 skilled pairs per 0.5%, so the skilled tail is coarse;
  hard_reject still rests on one extreme genuine (not scored); val genuine NOMATCH 5.1%
  sits just above the 5% target.

## EXP-019 — Symmetric stroke alignment; cohort score normalisation rejected

* **Question:** `compare(a, b)` aligns specimen -> query (keypoint RANSAC seed + ICP). Does a
  symmetric comparison help, and does cohort normalisation against a fixed set of dev
  genuines (no training) remove "generic signature scores high against everyone"?
* **Method (dev only):** `compare(i, j)` signals for all 660 x 659 ordered harmonized dev
  image pairs, cached once; every variant refits the fusion (`fit_fusion._fit`, same
  `--max-random 6000` pair set) on one writer fold and scores the other. Baseline
  reproduces fit_fusion: CV skilled EER **10.58%**, random 2.18%.
* **Symmetric variants** (signals of compare(a, b) and compare(b, a), refit):

  | variant | skilled EER | AUC | random EER |
  |---|---|---|---|
  | forward (production) | 10.58 | 0.9582 | 2.18 |
  | backward only | 10.41 | 0.9533 | 2.42 |
  | mean, all signals | 10.06 | 0.9596 | 2.07 |
  | min, all signals | 11.27 | 0.9549 | 2.31 |
  | concat (16 signals) | 10.19 | 0.9587 | 2.04 |
  | max, all signals | 9.70 | 0.9611 | 1.80 |
  | **max, stroke_direction + pressure_pattern only** | **9.59** | **0.9618** | **1.70** |
  | mean, stroke_direction + pressure_pattern | 9.81 | 0.9607 | 2.05 |

  Max on one signal at a time: pressure 9.34, direction 10.30, keypoint 10.91; slant,
  profiles, width, curvature are already symmetric (identical numbers). The two ICP-read
  signals are the asymmetric ones: the better of two alignments is a better alignment.
  Per fold (forward -> shipped): 10.23 -> 8.38, 9.65 -> 9.15.
* **Cohort normalisation** (honest: cohort = K genuines of the *other* fold's writers,
  round-robin by writer; logits from the fold's own fusion). Skilled EER %:

  | base / norm | K=20 | K=50 | K=100 |
  |---|---|---|---|
  | forward, Z-norm by reference | 12.95 | 12.37 | 12.62 |
  | forward, Z-norm by query | 12.37 | 12.23 | 12.48 |
  | forward, S-norm (Z, both sides) | 12.12 | 12.12 | 12.01 |
  | forward, minus reference cohort mean | 11.16 | 11.27 | 11.40 |
  | forward, minus mean of both cohort means | 9.81 | 10.06 | 10.06 |
  | forward, minus query cohort mean | 9.56 | 9.59 | 9.59 |
  | max-all, minus query cohort mean | 8.98 | 9.12 | 9.34 |
  | max-all, minus both means | 9.20 | 9.31 | 9.34 |
  | **shipped**, minus query cohort mean | 8.84 | 9.20 | not run |

  Dividing by the cohort sigma always hurts (+1.4 to +2.8 pt). Mean offsets help the
  forward score but add only 0.4-0.75 pt on top of the symmetric signals, not monotone in
  K. Caveat variants (forward / max-all, minus both means): cohort from *all* other
  dev writers (including test-fold ones) 10.80 / 10.17 at K=20, 10.17 / 9.70 at K=50;
  "val-like" (same writers allowed, only the pair's images excluded; with K <= 55 only
  the first K sorted writers can contribute) 10.69 / 9.81 at K=20, 10.08 / 9.34 at K=50,
  i.e. no gain over the symmetric signals. Cost: K compares per side per request (K=20,
  query side: ~200 ms, +55% of a request), a shipped cohort, and a logit that is no
  longer the fused log-odds the thresholds and inspect breakdown assume. **Rejected:**
  not robust above the 0.5 pt bar.
* **Shipped:** `similarity.compare` computes stroke signals a second time with the roles
  swapped (own b -> a keypoint seed) and keeps max(forward, reverse) for stroke_direction
  and pressure_pattern. All consumers (1:1, multi-specimen max, fitting scripts,
  clearance score, inspect) read `pair.signals`, so the fused log-odds is still
  bias + sum(weight x signal); inspect's breakdown is unchanged and still sums exactly.
  Refit (dev): CV skilled AUC 0.9618, EER **9.59%**, random 1.70%; all weights positive.
  Thresholds (EXP-018 rule): single accept 5.5123 -> 6.3685, reject -0.6729 -> -0.6514,
  hard -5.5863 -> -5.5909; multi accept 8.3837 -> 8.7599, reject 2.2261 -> 2.5911, hard
  0.0533 -> -1.0847. Nested held-out single: genuine accept 53.2% / reject 4.5%, skilled
  accept 0.35%, random accept 0%; multi: genuine 67.3% / 3.0%, skilled accept 0.56%.
* **Dev clearance (in-sample):** 6.29 (D 0.725, R 0.580, O 0.549); clean skilled EER 8.73%.
* **Val clearance (reported once):** 5.06 -> **5.24** (D 0.599 -> 0.622, R 0.409 -> 0.456,
  O 0.479 -> 0.461, gate off). Clean skilled EER 12.48% -> **11.76%** (AUC 0.946), random
  2.67% -> 2.54%. Operating point: genuine MATCH 28.8 / REVIEW 66.1 / NOMATCH 5.1%; skilled
  MATCH 0.2 / NOMATCH 73.5%; random MATCH 0.0 / NOMATCH 99.7%. Condition EERs: tinted 11.4,
  shadow 12.0, ruled 12.5, large canvas 11.6, distractor 11.7, faint 11.7, scale 0.5 18.4,
  scale 2.0 17.2, rotate +20 14.3, rotate -15 16.3, phone 12.6. Max skilled MATCH over
  conditions 0.2%.
* **Cost:** compare 5.4 -> 10.2 ms per pair (serial, 200 dev pairs); `compare_signatures`
  339 -> 375 ms per request (12 dev pairs, serial; the 1:1 path calls compare() more than
  once).
* **Limits:** the higher accept threshold (larger weight norm) lowered val genuine MATCH
  36.8% -> 28.8%, so O fell slightly; multi-specimen nested skilled accept rose 0.35% ->
  0.56% (rule unchanged, not re-tuned). Genuine NOMATCH under scale / rotation is still
  12.8-18.3%.
