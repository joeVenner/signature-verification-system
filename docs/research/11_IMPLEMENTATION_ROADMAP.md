# Implementation Roadmap — Shipping Plan

## Phase 0 — Scope & Compliance (Weeks 0-2)
- Decide scenario (bank clearing / fintech hybrid / auxiliary build) per file 10 — this gates everything else
- Define FAR/FRR targets per amount tier (e.g., FAR <0.1% above AED 100k)
- Regulatory review: if touching CBUAE clearing -> stop, engage vendor; if auxiliary -> confirm data-consent basis for signature collection
- Inventory reference signature sources (core banking, signature cards, historical cheques)

## Phase 1 — Data & Models (Weeks 2-8)
- Pretrain/benchmark: GPDS-Synthetic pretraining; validate chosen model class on CEDAR + BHSig260 against published EERs
- Collect local signatures: consent, 300 DPI, pen/ink diversity, 10+ samples per signer; build controlled skilled-forgery eval set (ethical safeguards)
- Baseline: SigNet (sanity check) -> target model: DetailSemNet (best EER) or ProtoSig (best few-shot efficiency)
- Fine-tune YOLO11 on cheque field layout: [MICR band, CAR, LAR, date, payee, signature zones]
- Freeze a held-out golden test set for regression testing

## Phase 2 — MVP Pipeline (Weeks 8-14)
- Wire: capture -> IQA -> YOLO detection -> ink extraction -> embedding -> distance vs reference vectors -> score
- FastAPI service + PostgreSQL immutable audit schema (both images, score, policy version, model version, operator, timestamps)
- Three-tier routing v1 (Green/Amber/Red) with versioned config thresholds
- Simple review console: side-by-side reference vs presented signature, approve/escalate buttons
- End-to-end eval on golden set: measure EER, STP rate, operator override rate

## Phase 3 — Calibration & HITL Hardening (Weeks 14-20)
- Per-user adaptive thresholds from accumulated genuine history
- Operator SLA timers + escalation chains; override feedback loop into threshold recalibration
- Regression harness: every model/threshold change re-scored against golden set; routing must be bit-stable (same input -> same route)
- Security review: access controls, image encryption at rest, PII handling

## Phase 4 — Risk Controls & Scale (Weeks 20-30)
- Mandate matrix engine (joint signers, amount tiers)
- Positive pay / reverse positive pay integration
- Duplicate presentment + alteration forensics (UV/color)
- Load testing: p95 decision latency, queue throughput, GPU serving costs (vLLM/batching for any VLM auxiliary tier)
- UAT with real cheque images from the target domain

## Phase 5 — Pilot & OEM Decision (Weeks 30+)
- Shadow-mode pilot: run alongside manual process; compare decisions; tune thresholds
- Go/no-go: if skilled-forgery performance or regulatory burden exceeds build capacity -> OEM-license Parascript/Mitek engine and keep the app layer

## Risk Register
| Risk | Impact | Mitigation |
|---|---|---|
| Arabic/local signature data scarcity | High — model generalizes poorly | Early collection drive; UTSig-style augmentation; synthetic writers |
| Skilled-forgery evaluation difficulty | High — can't ship on random-forgery metrics | Controlled forgery eval set; published-benchmark parity checks |
| VLM temptation (zero-shot "good enough") | Severe — coin-flip on skilled forgeries | Hard rule: VLMs never make accept/reject decisions (file 05) |
| Regulatory / clearing compliance | Severe if in scope | Phase 0 gate; vendor engagement |
| Model drift / signature aging | Medium | ProtoSig prototypical updates; periodic recalibration |
| Operator over-trust in auto-decisions | Medium | Blind spot-audits; sampled forced reviews; override tracking |

## Team (minimum viable)
1 CV/ML engineer (models, calibration), 1 backend engineer (FastAPI/PostgreSQL/queues), 1 frontend (React review console), part-time QA/ops + a banking-operations domain advisor for HITL workflow design.

## KPIs
- Skilled-forgery EER on held-out local set (target: match published class, <2-4%)
- STP rate (target 60-80% of volume) and operator override rate (target <10% of reviewed items)
- p95 end-to-end decision latency; audit-log completeness (100%); routing stability (100% deterministic given same inputs+policy version)
