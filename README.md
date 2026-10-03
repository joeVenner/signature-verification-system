# Deterministic Signature Verification

Offline signature verification and cheque-clearing adjudication, built for a
UAE/CBUAE banking context. It is classical computer vision with no trained
neural weights, and it is **bit-for-bit deterministic**: the same input always
gives the same score, verdict and explanation.

```
cheque → signature located → quality gate → normalise → compare with enrolled specimens
       → ACCEPT / REVIEW / REJECT (validated thresholds) → clearing policy → explanation + audit
```

## Results (held-out writers, raw scans; see [`benchmark/REPORT.md`](benchmark/REPORT.md))

| metric | legacy system | this system |
|---|---|---|
| Skilled-forgery EER (1 specimen) | 19.3% (27.5% once a dataset shortcut is removed) | **11.1%** |
| Skilled-forgery AUC | 0.877 | **0.934** |
| Different-writer EER | 9.6% | **2.8%** |
| 3 specimens: forgeries auto-accepted | 3 / 96 | **0 / 96** (and 0 / 2112 other writers) |
| 3 specimens: genuine auto-accepted | 65% | **88%** |
| Determinism | within a run | byte-identical across processes |

Measured on 12 CEDAR writers, which is a small sample. Read the report's
limitations section before relying on these numbers.

## Quick start

The code imports itself as `signature_verification_system.*`, so **clone into a
folder with that exact name** and run everything from its parent directory:

```bash
git clone https://github.com/joeVenner/signature-verification-system.git signature_verification_system
pip install -r signature_verification_system/requirements-dev.txt
python -m pytest signature_verification_system/tests -q
```

### Signature-only comparison (original signature(s) vs one to check)

```bash
python signature_verification_system/scripts/demo_cli.py compare \
    --reference original1.png original2.png original3.png \
    --questioned to_check.png        # --json for machine-readable output
```

It returns a verdict (MATCH / UNCERTAIN - MANUAL REVIEW / NO MATCH /
INCONCLUSIVE - IMAGE QUALITY) and a 0–100 similarity score aligned with it
(≥ 70 match, < 40 no match, otherwise review), plus the evidence. Supply 3 or
more originals: with a single original, genuine signatures go to manual review
by design.

### API

```bash
uvicorn signature_verification_system.src.api.app:app --port 8000   # docs at /docs
```

* `POST /api/v1/signature/compare`: signature only (`reference_images[]`, `questioned_image`)
* `POST /api/v1/cheque/process`: full cheque pipeline with clearing policy and audit record
* `GET /api/v1/audit/verify-chain`: audit hash-chain integrity

> Security posture is **demo-grade**: there is no authentication or rate limiting,
> and the audit chain is unkeyed SHA-256. Run on localhost or a trusted network only.

## Data (not included)

The sample images are **not redistributed**: CEDAR is an academic dataset, and
the cheque scans are third-party. `data/samples/*/manifest.json` and
`data/samples/README.md` describe the expected layout. Place the images there to
run the benchmark, the demo and the data-dependent tests; without them those
tests are skipped automatically.

## Documentation

* [`benchmark/REPORT.md`](benchmark/REPORT.md): final benchmark (baseline vs final, FAR/FRR/AUC/EER, thresholds, robustness, runtime)
* [`benchmark/experiments.md`](benchmark/experiments.md): every experiment, including rejected ones
* [`DETERMINISM.md`](DETERMINISM.md): every source of nondeterminism and how it is controlled
* `00_INDEX.md` … `14_*.md`: background research notes
