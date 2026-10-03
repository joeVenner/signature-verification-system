# VLMs (GPT/Claude/Gemini/Qwen-VL) & Signature Verification

## Empirical Result (Zero-Shot Capability Study — BiDA Lab, arXiv:2605.14845)

| Task | Zero-shot frontier VLM EER | Supervised domain models |
|---|---|---|
| Random forgeries | 0.32% (near-flawless — matches supervised SOTA) | 0.1-0.3% |
| Skilled forgeries | 32.2%-48.08% (coin-flip territory) | 1-4% |

## The Rationalization Trap
Chain-of-thought prompting ("compare stroke dynamics, hesitation, loops step-by-step") makes VLM forgery detection WORSE: the model hallucinates plausible motor-execution justifications for forged stroke artifacts — describing genuine forgery discrepancies (tremor, pen lifts, differing slant) as "normal intra-writer biological variation caused by fatigue or signing angle." It narrates itself into accepting forgeries.

## Three Structural Failure Modes
1. Patch tokenization destroys forensic evidence. Hesitation tremors and pen lifts span 1-3 pixels (<0.5 mm of ink). ViT 14x14/16x16 patch projection aggregates high-frequency stroke edges into broad tokens — micro-evidence is wiped before reasoning even starts.
2. No calibrated continuous distance metric. Biometrics need a scalar similarity mappable to an exact FAR for Neyman-Pearson thresholding. VLMs emit natural-language probability verdicts over text tokens — statistically uncalibrated, so defensible FAR/FRR thresholds are impossible.
3. Prompt brittleness & confirmation bias. "Is this signature authentic?" vs "examine these two signatures for forgery" shifts the posterior wildly; decisions bias toward whichever hypothesis was mentioned first.

## Where VLMs DO Add Value
- Post-hoc audit explanations — narrating why a metric model flagged a pair (for case files, never as the decision).
- Fraud investigation summaries across documents (stamps, alterations, payee handwriting style).
- Surrounding handwriting OCR — payee names, amounts (CAR/LAR), dates: VLMs excel at handwriting recognition.
- DualSigVL pattern (2026): if a VLM must be in the decision path, it must be (a) LoRA/PEFT fine-tuned on verified corpora, (b) paired with an explicit low-level morphological expert (HOG/SIFT/FAST/Canny), and (c) gated by a Mixture-of-Experts confidence head. Never zero-shot.

## Bottom Line
Handwriting OCR != signature verification. VLMs solved the first (see file 13) and catastrophically fail the second. In our system the VLM is an adjutant, never the judge: the core skilled-forgery decision belongs to a metric-learning model with a calibrated score.
