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
| Stroke alignment (ICP) | `stroke_geometry.py` | KD-tree nearest neighbours (`cKDTree`, single worker), 2x2 SVD, fixed iteration count, no random restarts; the two starting transforms are tried in a fixed order and ties keep the first |
| Skeletonisation | `skimage.morphology.skeletonize` | pure deterministic thinning on a fixed 512x256 canvas; points taken in raster order (`argwhere`), capped by an even stride when above 20,000 |
| Banded DTW, slant EMD | `stroke_geometry.py` | fixed-size dynamic programming / cumulative sums; no data-dependent ordering |
| Cross-machine floats | all of the above | bit-identical on one pinned build; a different CPU/BLAS/OpenCV/scikit-image build may differ in the last digits, so decisions use margins (thresholds are not at exact float ties) and display values are rounded |
| BLAS threads (numpy matmul / SVD) | `stroke_geometry.py` and fitting scripts | results are identical across runs on one machine (100-run test), but thread oversubscription made small matrix products ~500 ms under parallel load (EXP-020). Launch services and benchmarks with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; these must be set before numpy is imported, so they belong in the launch environment, not in code |
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
`scikit-image 0.24`, `scipy` (see `requirements.txt`; Python >= 3.12). Different OpenCV builds may produce
different (but internally deterministic) SIFT/NLM outputs, so re-run
`fit_fusion.py` / `select_thresholds.py` after changing them.
