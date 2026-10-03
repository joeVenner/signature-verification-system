# System Architecture — Signature Verification for Cheques & Documents

## Top Level: Two ML Tasks + A Decision Layer

1. DETECTION — "Is there a signature here?" + localize it (YOLO-class object detection, ~94%+ mAP, <10ms)
2. VERIFICATION — "Is this signature genuine or forged?" (biometric metric learning — where systems live or die)
3. ADJUDICATION — "What do we do about it?" (calibrated thresholds, routing, human-in-the-loop, audit)

## Full Production Pipeline (7 Stages)

[Deposit channels: Branch scanner | Smart ATM/CCDM | Remote deposit | Mobile camera]
  -> STAGE 1 — CAPTURE & IMAGE QA (IQA/IUA per ANSI X9.100-181 / CBUAE specs): 200-300 DPI; grayscale/bi-level/color/UV; skew, darkness, folded-corner checks; fail -> reject at device, prompt re-scan
  -> STAGE 2 — MICR CODE-LINE READING: E-13B font (GCC/US) or CMC-7 (Europe/LatAm); magnetic + optical dual-read; parse [Cheque No.][Routing/Sort][Account][Transaction Code]; modulo-10/11 checksums
  -> STAGE 3 — FIELD EXTRACTION (deep layout segmentation): CAR (courtesy/numeric amount box) | LAR (legal amount in words) | Payee name | Date | Endorsement zone (rear)
  -> STAGE 4 — ARBITRATION & LEGAL CHECKS: CAR vs LAR "words govern figures" (UAE Commercial Transactions Law); Arabic tafqeet parsing for written amounts; mismatch -> exception queue; Date: stale (>6 months -> return) | post-dated -> PDC hold vault
  -> STAGE 5 — AUTOMATED SIGNATURE VERIFICATION (ASV): Crop signature zone -> remove baseline/pantograph/stamps -> embedding -> distance vs specimen card vector (vector DB via account number) -> mandate rule engine (single / joint / any-2-of-N; amount-tier limits)
  -> STAGE 6 — RISK CONTROLS: Positive pay 4-way match (account, cheque#, amount, payee) | duplicate presentment (internal + central clearing index) | alteration forensics (UV/chemical washing)
  -> STAGE 7 — THREE-TIER DECISION ROUTING: GREEN -> auto-clear -> transmit to central clearing (e.g., UAE ICCS); AMBER -> L1 operator queue (side-by-side compare, SLA-bounded); RED -> four-eyes dual verification -> approve or return code ("Signature Differs" / "Altered Cheque")

## Three-Tier Decision Routing (Policy Matrix)

| Tier | Trigger conditions | Action | Expected volume |
|------|--------------------|--------|-----------------|
| Green (STP) | Score >=85% AND CAR=LAR AND positive-pay match AND amount < floor limit (e.g., AED 20k) | Auto-clear, zero human touch | 60-80% of daily volume |
| Amber (L1) | Score 65-84%, OR low CAR/LAR confidence, OR amount AED 20k-100k | Back-office operator queue: side-by-side image comparison UI, approve/escalate within 30-60 min SLA | ~15-30% |
| Red (L2) | Score <65%, OR >= AED 100k high-value, OR mandate violation (missing joint signer) | Mandatory four-eyes: Senior Clearing Officer + Operations Manager dual sign-off; rejection emits standardized return codes | ~5% |

## The Determinism Principle
- The ML system outputs a calibrated similarity score s in [0,1] — never a final yes/no.
- Acceptance thresholds (theta_STP, theta_reject) are set per amount tier and versioned in configuration, with model version pinned alongside.
- Every decision is logged: both images, score, threshold policy version, model version, operator ID (for overrides).
- A borderline cheque must route to the same queue every single time (same input -> same route). The system must be consistently correct or consistently conservative — never randomly confident.
- Operator overrides feed back into threshold recalibration (feedback loop).

## Hybrid WI/WD Verification Design
- A frozen writer-independent (WI) feature extractor encodes each new customer's signature card into a vector -> stored in a vector DB. Zero retraining at onboarding (scales to 10M+ accounts).
- Per-user adaptive calibration tightens decision margins as genuine transaction history accumulates (writer-dependent behavior without per-user models).
- Prototypical reference representation (ProtoSig pattern): summarize 1-N reference signatures into a prototype to cut comparison FLOPs by ~99.99% and resist natural signature drift (template aging).
