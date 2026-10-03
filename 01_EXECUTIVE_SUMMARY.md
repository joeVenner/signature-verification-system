# Executive Summary — Signature Verification System

## What We Are Building
A system that verifies handwritten signatures on cheques and documents: detect the signature on the page, verify it is genuine vs forged against a reference specimen, and adjudicate — route the decision with a full audit trail.

## The Three-Problem Framing
1. **Detection** — "Is there a signature here?" + localize it. Solved with modern object detectors (~94%+ mAP, <10 ms inference).
2. **Verification** — "Is this signature genuine or forged?" The hard biometric core. Skilled forgeries are the only benchmark that matters.
3. **Adjudication** — "What do we do about it?" Calibrated thresholds, routing, human-in-the-loop, immutable audit. This is the product.

## Key Numbers (2024-2026 SOTA, skilled forgeries)
- Best published skilled-forgery EER: DetailSemNet 0.58% (CEDAR), 2.07% (BHSig-H), 2.11% (BHSig-B)
- ProtoSig (ICPR 2026 award): 1.29% EER (CEDAR) with only 1-3 reference signatures; cuts verification FLOPs by ~99.99% (2.02e12 -> 1.10e5)
- Random forgeries: 0.1-0.3% EER on all modern models (trivial)
- 1-shot enrollment penalty: 4-8% EER without augmentation/calibration
- VLMs zero-shot: 0.32% EER on random forgeries but 32.2-48.08% EER on skilled forgeries — coin-flip level. DO NOT use for core decisions.

## Production Architecture Decision
- Writer-independent metric extractor (frozen) + per-user adaptive threshold calibration (hybrid WI/WD)
- Three-tier routing: Green >=85% straight-through processing; Amber 65-84% operator queue; Red <65% or high-value four-eyes dual verification
- ML produces a calibrated similarity score; deterministic code owns every accept/reject/route decision + the audit trail

## Build vs Buy (UAE/GCC)
- Regulated bank in CBUAE clearing: BUY — ProgressSoft (runs CBUAE ICCS national clearing), Parascript, or Tungsten FraudOne
- Fintech / neobank / PDC management: HYBRID — build the app layer, OEM-license the verification engine (Parascript SignatureXpert SDK / Mitek) as containerized microservices
- Auxiliary non-clearing (real estate PDC tracking, corporate AP, back-office triage): BUILD fully open-source
- Pricing reality: commercial engines $40k-150k base license + 18-22%/yr maintenance + $0.02-0.12 per-cheque royalty at 1M-10M/yr volume

## Critical Warnings
1. The Rationalization Trap: VLMs + chain-of-thought hallucinate justifications for forged stroke artifacts ("normal intra-writer variation caused by fatigue") — CoT makes VLM forgery detection WORSE, not better.
2. Naive Otsu binarization merges cheque guilloche security patterns with thin ink strokes — use Sauvola/Wolf adaptive binarization or a DIBCO-trained U-Net.
3. Arabic signature datasets are scarce — plan proprietary data collection (with consent + quality specs) from week one.
4. VLMs give natural-language verdicts, not calibrated continuous scores — you cannot set defensible FAR/FRR thresholds on them.
5. Legal weight: under UAE Federal Decree-Law No. 14 of 2020, disputed cheques go to direct civil execution — every decision must log both images, the score, the threshold policy version, and model version.

## Recommended Core Stack (Build Path)
YOLO11x (detection) -> CIE-Lab ink extraction -> DetailSemNet or ProtoSig (verification) -> per-user calibrated thresholds -> three-tier routing -> FastAPI + PostgreSQL immutable audit log -> React review console.
