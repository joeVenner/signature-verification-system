# Signature Verification System — Knowledge Base

**Purpose:** Consolidated research base for designing and shipping a handwritten signature verification system for cheques and documents (UAE/GCC context). This folder is the single source of truth for all findings, benchmarks, vendor options, and architecture decisions.

**Compiled:** 29 September 2026
**Status:** Research complete — ready for build planning (v1.0)

## File Map

| # | File | Contents |
|---|------|----------|
| 01 | EXECUTIVE_SUMMARY.md | Headline findings, key numbers, critical warnings, recommended stack |
| 02 | SYSTEM_ARCHITECTURE.md | 3-problem framing, 7-stage production pipeline, decision routing |
| 03 | DETECTION_LOCALIZATION.md | Stage 1: finding and cleaning signatures on documents |
| 04 | VERIFICATION_SOTA.md | Stage 2: genuine-vs-forged models, losses, benchmark tables |
| 05 | VLM_LIMITATIONS.md | Why GPT/Claude/Gemini cannot do core verification (and where they help) |
| 06 | DATASETS_BENCHMARKS.md | Public training/eval datasets and EER benchmarks |
| 07 | OPEN_SOURCE_STACK.md | Build-it-yourself components, repos, serving stack |
| 08 | VENDOR_LANDSCAPE.md | Commercial products comparison + real pricing |
| 09 | CHEQUE_PROCESSING_PIPELINE.md | MICR, CAR/LAR, mandates, positive pay, UAE clearing |
| 10 | BUILD_VS_BUY.md | Strategic build-vs-buy decision matrix (UAE/GCC) |
| 11 | IMPLEMENTATION_ROADMAP.md | Phased plan to ship a solution |
| 12 | RESOURCES_LINKS.md | All vendors, GitHub repos, datasets, standards, regulations |
| 13 | FIELD_EXTRACTION_OCR.md | Handwriting OCR for cheque fields (supporting research) |
| 14 | HANDWRITTEN_OCR_PLAYBOOK.md | Earlier deep-dive OCR research (copied in) |

## Recommended Reading Order
1. 01 then 02 — the big picture (~30 min)
2. 10 then 11 — the decision and the plan
3. 03-09 — deep reference during build

## Headline Conclusions
1. Signature verification = detection (solved, 94%+ mAP, <10ms) + verification (hard biometric core) + adjudication (routing/thresholds/audit — the actual product).
2. SOTA skilled-forgery EER on CEDAR: 0.58% (DetailSemNet, ECCV 2024), down from ~16% (GPDS) for classic baselines.
3. Never use zero-shot VLMs for skilled-forgery decisions — they score 32-48% EER (coin-flip). They are near-flawless only on random forgeries (0.32% EER).
4. Production standard = writer-independent metric model + per-user adaptive threshold calibration + three-tier deterministic routing.
5. UAE: banks clearing via CBUAE ICCS -> buy (ProgressSoft embedded in national clearing). Fintech/PDC apps -> hybrid. Auxiliary tools -> build open-source.

## Source Confidence Note
Academic figures (EER tables, venues, arXiv IDs) come from the research pass — re-verify against linked primary sources before final model selection. Vendor pricing reflects published structures, not quotes.
