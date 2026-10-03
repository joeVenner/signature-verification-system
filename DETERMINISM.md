# Determinism

Identical inputs produce bit-identical scores, bands and explanations. This is
verified by `tests/test_pipeline.py::TestCrossProcessDeterminism` (3 fresh
interpreters) and by byte-comparing benchmark JSON across processes (see
`benchmark/experiments.md`).

Single control point: `src/core/determinism.py` (`GLOBAL_SEED`, `OPENCV_THREADS`,
`configure_determinism()`, `seeded_rng()`).

## Every source of potential nondeterminism, and how it is handled

| Source | Where | Handling |
|---|---|---|
| Random numbers | `benchmark/robustness.py` additive noise (the **only** stochastic code) | `seeded_rng(offset)` = PCG64(GLOBAL_SEED + image SHA prefix); order-independent |
| RANSAC sampling | `cv2.estimateAffinePartial2D` in `similarity.py` | OpenCV re-seeds its RNG with a fixed constant on every call; pinned by tests |
| Multithreading | OpenCV parallel loops (NLM, filters, SIFT) | `cv2.setNumThreads(1)` applied once at import via `configure_determinism()` |
| Shared mutable state | SIFT detector | created per call (no singleton) |
| Keypoint order | SIFT output | lexicographically sorted by (y, x, descriptor sum) |
| Filesystem order | dataset loading | images de-duplicated by SHA-256, explicitly sorted; no `glob` order dependence |
| Dict / set order | protocol, metrics | Python dicts are insertion-ordered; all set results are `sorted()` |
| Model fitting | `fit_fusion.py` (sklearn `LogisticRegression`, lbfgs) | deterministic solver on a fixed, sorted design matrix; coefficients stored in `config.py` |
| Threshold selection | `select_thresholds.py` | order statistics only (no interpolation randomness) |
| Metric computation | `benchmark/metrics.py` | exact AUC (Mann-Whitney) and EER over all unique scores, so no grid resolution |
| Float rounding of display | `VerificationResult` | decisions use the unrounded `match_logit`; rounding affects display only |
| GPU | none | no GPU code paths |
| Time / IDs | `audit_logger.py` (`timestamp_utc`, `uuid4` record id), API `format_document_id` when no id given | **intentionally non-deterministic audit metadata**; never feeds scores, bands or decisions |
| Benchmark wall-clock | `run_benchmark.py` | written to a separate `*.runtime.json`; the results JSON has no timestamps |

## Reproducibility commands (from the workspace parent directory)

```bash
python -m signature_verification_system.benchmark.run_benchmark --system current --condition raw --repeats 3 \
       --out signature_verification_system/benchmark/results/check.json      # exit code 2 if repeats differ
python -m signature_verification_system.benchmark.robustness
python -m signature_verification_system.benchmark.cheque_path
python -m pytest signature_verification_system/tests -q
```

Environment pins that matter: `numpy<2`, `opencv-python-headless==4.10.0.84`,
`scikit-image 0.24` (see `requirements.txt`). Different OpenCV builds may produce
different (but internally deterministic) SIFT/NLM outputs, so re-run
`fit_fusion.py` / `select_thresholds.py` after changing them.
