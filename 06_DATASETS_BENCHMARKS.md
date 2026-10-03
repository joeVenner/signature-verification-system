# Datasets & Benchmarks

## Public Datasets

| Dataset | Writers | Genuine / skilled forgery per writer | Script / notes |
|---|---|---|---|
| CEDAR | 55 | 24 / 24 (2,640 images) | English cursive; the classic benchmark |
| GPDS-Synthetic (GPDS-300/960) | 10,000 synthetic | 24 / 30 (540,000 images) | Simulates pen types, nibs, line thicknesses — best pretraining corpus |
| BHSig260-B (Bengali) | 100 | 24 / 30 | Indic script; cross-script generalization test |
| BHSig260-H (Hindi) | 160 | 24 / 30 | Devanagari |
| MCYT-75/100 | 75-100 | 15 / 15 | Spanish; includes online dynamic (time-series) data |
| UTSig | 820 | — | Persian; right-to-left script — closest proxy to Arabic-style directionality |
| ICDAR SigComp 2009/2011/2013 | varies | competition sets | SigComp2011: Dutch + Chinese; SigComp2013: Bengali + forensic simulated cases |

## Metric Conventions
- EER (Equal Error Rate): operating point where FAR = FRR. The universal metric in forensic/financial ASV literature.
- Always report skilled-forgery EER. Random-forgery numbers (~0.1-0.3% everywhere) are meaningless for fraud defense.
- For cheques: calibrate to an FAR target per amount tier (not EER) — acceptance errors cost more than review queues.

## Training Strategy for Our Build
1. Pretrain on GPDS-Synthetic (scale + ink/nib diversity).
2. Benchmark on CEDAR + BHSig260 to verify the model class works as published.
3. Collect proprietary Arabic/local signatures early — public Arabic signature datasets are scarce; UTSig (Persian) is the nearest directional proxy. Collection needs: written consent, 300 DPI scans, pen/ink diversity, 10+ samples per consenting signer (genuine only), plus controlled skilled-forgery attempts for evaluation (with ethical safeguards).
4. Augmentation for 1-shot enrollment: elastic distortions, affine shearing, stroke thinning/thickening, mild rotation — simulate intra-writer variance from a single reference.
5. Keep a held-out golden test set frozen for regression testing every model/threshold change.

See file 04 for the model-vs-benchmark EER table.
