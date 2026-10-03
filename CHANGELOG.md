# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Stroke-level comparison signals (`src/verification/stroke_geometry.py`): stroke direction along ICP-aligned strokes, writing slant, horizontal / vertical ink profiles (banded DTW) and pen width, fused with the keypoint signal. 1:1 skilled-forgery EER on an untouched 55-writer test slice 16.5% -> 9.3% (EXP-015).
- `benchmark/build_cedar_eval.py` to build dev / test slices of full CEDAR; `--data-dir`, `--max-random` and `--max-random-per-query` options for the benchmark, fit and threshold scripts.
- Explanation evidence for stroke direction, slant, horizontal / vertical rhythm and stroke width.
- Hermetic tests with synthetic signatures (`tests/test_stroke_geometry.py`), including cross-process determinism.
- v3 deterministic verifier: photometric normalisation (polarity, NLM denoise, 2-pass flat-field), SIFT + RANSAC keypoint similarity, gradient-grid layout similarity with RANSAC alignment, and a fitted logistic fusion.
- Multi-specimen verification (`verify_against_references`, feature-wise max aggregation).
- Validated ACCEPT / REVIEW / REJECT log-odds bands and a signature quality gate (INCONCLUSIVE).
- Evidence-based explanations derived only from measured signals.
- `ChequeVerificationPipeline` orchestrator; composed-cheque judge demo (`scripts/demo_showcase.py`, `demo_cli.py demo`).
- Signature-only comparison mode: `POST /api/v1/signature/compare`, `demo_cli.py compare`.
- Leak-free benchmark suite (1:1, 3-specimen, robustness, cheque path), experiment log and final report.
- `DETERMINISM.md` and `src/core/determinism.py` (single determinism control point).

### Changed
- `FusionModel` now holds one weight per signal (`signal_weights`, validated); fusion weights, ACCEPT / REVIEW / REJECT thresholds and the evidence reference were refitted on a 55-writer dev split. With three specimens genuine auto-accept falls (73% -> 29%) while skilled forgeries accepted fall 1.5% -> 0% on the untouched test slice.
- Declared `scipy` (runtime), `scikit-learn` (dev, fitting scripts) and Python >= 3.12 in the requirements files.
- `DecisionEngine` routes logit bands; legacy probability thresholds only apply to results without a logit.
- API and CLI cheque processing run through the single pipeline and accept several specimens.
- `is_match` is true only for ACCEPT.

### Security
- The API no longer reads server filesystem paths supplied by clients.
- Header-only image-size checks (decompression-bomb guard), request size cap, strict Pydantic input validation, CORS allow-list, and account numbers masked in the audit chain.
