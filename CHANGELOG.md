# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- v3 deterministic verifier: photometric normalisation (polarity, NLM denoise, 2-pass flat-field), SIFT + RANSAC keypoint similarity, gradient-grid layout similarity with RANSAC alignment, and a fitted logistic fusion.
- Multi-specimen verification (`verify_against_references`, feature-wise max aggregation).
- Validated ACCEPT / REVIEW / REJECT log-odds bands and a signature quality gate (INCONCLUSIVE).
- Evidence-based explanations derived only from measured signals.
- `ChequeVerificationPipeline` orchestrator; composed-cheque judge demo (`scripts/demo_showcase.py`, `demo_cli.py demo`).
- Signature-only comparison mode: `POST /api/v1/signature/compare`, `demo_cli.py compare`.
- Leak-free benchmark suite (1:1, 3-specimen, robustness, cheque path), experiment log and final report.
- `DETERMINISM.md` and `src/core/determinism.py` (single determinism control point).

### Changed
- `DecisionEngine` routes logit bands; legacy probability thresholds only apply to results without a logit.
- API and CLI cheque processing run through the single pipeline and accept several specimens.
- `is_match` is true only for ACCEPT.

### Security
- The API no longer reads server filesystem paths supplied by clients.
- Header-only image-size checks (decompression-bomb guard), request size cap, strict Pydantic input validation, CORS allow-list, and account numbers masked in the audit chain.
