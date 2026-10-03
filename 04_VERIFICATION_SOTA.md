# Stage 2 — Signature Verification (Genuine vs Forged)

## Two Paradigms + the Production Hybrid

| | Writer-Independent (WI) | Writer-Dependent (WD) |
|---|---|---|
| Model | One universal metric network (SigNet / DetailSemNet) | Per-signer classifier (e.g., One-Class SVM) |
| Onboarding | Encode signature card -> store vector; zero retraining | Requires 10-20 reference signatures per user |
| Scalability | Yes — 10M+ accounts | No — operationally impossible in retail banking |
| EER | Slightly higher | Lower (tailored decision boundary) |

Production standard = hybrid: frozen WI deep-metric extractor (ResNet152 / DetailSemNet class) + user-specific adaptive thresholding or a lightweight per-user head calibrated dynamically as genuine transaction history accumulates.

## Metric Learning Losses
- Siamese + Contrastive loss: minimize genuine-pair distance; margin m against forged pairs.
- Triplet / Co-Tuplet loss (MS-SigNet, Huang & Lu): anchor-positive-negative with multiple genuine and multiple negative samples simultaneously — prevents embedding collapse for writers with wide intra-personal variation.
- Angular margin losses (ArcFace / CosFace): adapted from face recognition (2025 works) — tight angular clusters for genuine signatures, separating skilled forgeries.
- Architecture trend: hybrid CNN feature pyramids (micro-stroke curvature, line terminals, ink blobbing) + Transformer cross-attention (global layout, aspect proportions). Pure ViTs over-smooth local stroke details via patch tokenization.

## Few-Shot Enrollment (The Banking Reality)
- 1-shot (single reference card): EER penalty 4-8%. Mitigate with synthetic augmentation (elastic distortions, affine shear, stroke thinning) simulating intra-writer variance.
- Few-shot (r in [3,12]): ProtoSig / proto_hsv (ICPR 2026 award, Moura/Sabourin/Cruz): build a prototypical signature representation instead of all pairwise distances — verification FLOPs drop 2.02e12 -> 1.10e5 (>99.99% reduction) with near-optimal EER, plus resistance to template aging.

## SOTA Benchmark Table (Skilled Forgery EER %, lower = better)

| Model | Backbone | CEDAR | BHSig-H | BHSig-B | GPDS | MCYT-75 | Source |
|---|---|---|---|---|---|---|---|
| SigNet (baseline) | CNN/AlexNet | 4.63 | 15.36 | 13.89 | 16.44 | 2.87 | Dey/Hafemann |
| IDN | Dual-stream CNN | 3.62 | 6.96 | 4.68 | 8.20 | — | PR 2021 |
| TransOSV | ViT+CNN hybrid | — | 3.39 | 9.95 | — | — | IEEE TIFS |
| MS-SigNet | Multiscale + Co-Tuplet | 3.51 | 6.68 | 6.12 | — | — | arXiv 2308.00428 |
| FD-VAE | Disentangling VAE | — | — | — | 2.22 | 1.35 | arXiv 2024 |
| DetailSemNet | ResNet/Transformer + StructMatch | 0.58 | 2.07 | 2.11 | — | — | ECCV 2024 |
| ProtoSig (proto_hsv) | Continual ResNet + SVM | 1.29 | — | — | 4.02 | 2.68 | ICPR 2026 |
| DualSigVL | Dual Qwen-VL + DECA-MoE | 1.15 | 2.45 | 2.38 | — | — | Springer 2026 |

Random forgeries: 0.1-0.3% EER on all modern architectures (near-trivial).

## Forgery Typology & Defenses

| Forgery type | What it is | Failure risk | Defense |
|---|---|---|---|
| Random | Forger signs own name / unrelated name | Trivial | Any Siamese baseline (>99% detection) |
| Simple | Knows the name, never saw the signature design | Low-moderate | Global structural matching |
| Skilled | Practiced forgery copying flourishes/slant/proportion | High — the real benchmark | Micro-detail structural matching (DetailSemNet): localized curvature discrepancies, line continuity, stroke connection angles |
| Traced | Lightbox/carbon/digital tracing — geometry identical | Highest acceptance risk | Kinematic anomaly detection: tracing tremors, hesitation blobs, uniform stroke thickness, absent ballistic acceleration |
| Disguised | Genuine writer deliberately alters signature to repudiate later | Causes false rejections | Invariant micro-feature modeling (involuntary motor habits survive disguise) |

## Threshold Calibration & FAR/FRR
- EER = operating point where False Acceptance Rate = False Rejection Rate. For cheques, calibrate asymmetrically: FAR must be near-zero for high-value tiers (accept FAR < 0.1% => raise threshold => accept more human review).
- Use value-weighted thresholds: per-amount-tier decision boundaries (e.g., AED <20k lenient-STP, 20k-100k operator review, >100k four-eyes).
- Per-user adaptive margins tighten as genuine history accumulates (hybrid WI/WD).

## Offline vs Online
- Offline (scanned paper) = 2D integrated ink deposit only — the cheque reality.
- Online (tablet/POS) captures (x, y, pressure, azimuth, t) at ~200 Hz — far easier forgery detection, but only applies to digital-capture enrollment (KYC signature pads) or POS e-sign.
- Bridge: pseudo-dynamic stroke trajectory recovery from ink density gradients (see file 03) recovers partial dynamics from offline scans.
