# Deterministic Signature Verification

Offline signature verification and cheque-clearing adjudication, built for a
UAE/CBUAE banking context. It is classical computer vision with no trained
neural weights, and it is **bit-for-bit deterministic**: the same input always
gives the same score, verdict and explanation.

```
cheque → signature located → quality gate → normalise → compare with enrolled specimens
       → ACCEPT / REVIEW / REJECT (validated thresholds) → clearing policy → explanation + audit
```

## Results (frozen 0–10 clearance score, one reference vs one questioned; see [`benchmark/REPORT.md`](benchmark/REPORT.md))

| CEDAR slice (55 writers) | before | after |
|---|---|---|
| **Holdout, images 19–24 (scored once)** | 4.93 | **5.78** |
| Test, images 13–18 (never tuned on) | 5.26 | **6.00** |
| Validation, images 7–12 (drove decisions) | 3.88 | **5.24** |

On the holdout: skilled-forgery EER 11.0% → **9.7%**, different-writer EER 3.9% → **2.9%**,
genuine signatures auto-matched 14.3% → **28.5%**, forgeries auto-matched **0%** in every
condition, printed text next to the signature 31.6% → **9.8%** EER, rotation ±15–20°
17–20% → **13–14%** EER. Same input → byte-identical output over 100 runs.

**Honest limits:** the 7/10 target was not reached; with one reference about two in three
genuine signatures still go to manual review; benchmarked on lab scans (CEDAR), not on real
counter photos. Read the report's limitations before relying on these numbers.

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

### Live verification console + API

```bash
cp signature_verification_system/.env.example signature_verification_system/.env   # edit as needed
python -m signature_verification_system.serve        # http://127.0.0.1:8765  (API docs at /docs)
```

All settings (host, port, CORS, sample gallery folder, audit ledger paths, OpenCV pixel
limit, CPU thread pinning) come from `.env`; variables already set in the shell win.
Set `SIGVERIFY_SAMPLES_DIR` to a folder laid out like `data/samples` to enable the sample picker.

* `/`: verification console (upload two signatures, see every pipeline stage, the signal contributions, score, confidence margin and verdict)
* `POST /api/v1/signature/compare`: signature only (`reference_images[]`, `questioned_image`)
* `POST /api/v1/signature/inspect`: the same result plus pipeline images and signal contributions
* `POST /api/v1/cheque/process`: full cheque pipeline with clearing policy and audit record
* `GET /api/v1/audit/verify-chain`: audit hash-chain integrity

> Security posture is **demo-grade**: there is no authentication or rate limiting,
> and the audit chain is unkeyed SHA-256. Run on localhost or a trusted network only.

### Slides

`slides/signature-verification/` is an [open-slide](https://www.npmjs.com/package/@open-slide/core)
deck about the system: `npm install && npm run dev`, then open `/s/signature-verification`.

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
* [`benchmark/clearance_score.py`](benchmark/clearance_score.py): the frozen 0–10 score used for every decision
* `00_INDEX.md` … `14_*.md`: background research notes
