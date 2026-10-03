# Open-Source Build Stack

## Verification Models (all with public code)

| Model | GitHub | Key innovation |
|---|---|---|
| DetailSemNet (ECCV 2024) | https://github.com/nycu-acm/DetailSemNet_OSV | PyTorch/Lightning. Disentangles global semantic features (layout, aspect ratio) from high-frequency local details (stroke turns, intersections) via explicit local structural matching. Best published EER (0.58% CEDAR). |
| ProtoSig / proto_hsv (ICPR 2026 award) | https://github.com/kdmoura/proto_hsv | Moura, Sabourin, Cruz (ETS Montreal / LIVIA). Python/PyTorch/scikit-learn (SGD Linear SVM). Prototypical signature representations — solves few-shot enrollment; FLOPs 2.02e12 -> 1.10e5. |
| MS-SigNet | https://github.com/ashleyfhh/MS-SigNet | Huang & Lu (arXiv 2308.00428v5). Multi-scale feature branch + Co-Tuplet loss. 3.51% CEDAR; supports HanSig Chinese. |
| DualSigVL | https://github.com/BEFINE-code/DualsigVL | Huang, Jiang et al. (Springer 2026). Raw-image VLM expert + hand-crafted feature expert (HOG, SIFT, FAST, Canny) fused via DECA-MoE confidence routing. PyTorch, LLaMA-Factory, Qwen-VL. |
| SigNet (baseline) | https://github.com/sounakdey/SigNet | Landmark Siamese CNN for offline forgery detection. Use as the sanity-check baseline. |

## Detection & Layout
- YOLO11 / YOLOv8 / Ultralytics (signature detection fine-tunes exist on Hugging Face and GitHub, e.g., samuellimabraz/signature-detection; YOLOS variants)
- DocLayout-YOLO — document layout detection
- LayoutLMv3 (Microsoft) — anchor-text semantic extraction for forms
- Cheque field segmentation: https://github.com/GreatClasher/Cheque-Detection (bounding box extraction for cheque fields)

## MICR & Cheque Reading
- https://github.com/sclaflin/MICR-scanner — E-13B MICR line decoding pipeline
- Tesseract with E-13B traineddata, or custom CRNN/CTC model for MICR digits

## OCR Engines for Cheque Fields
- TrOCR-Handwritten (Microsoft, Hugging Face) — fine-tune on bank digit styles for CAR (numeric amounts)
- EasyOCR / PaddleOCR — multilingual handwriting incl. Arabic for LAR words + payee
- Arabic tafqeet parser (number-to-words normalization) for CAR/LAR arbitration

## Reference DIY Pipeline
1. Ingestion & preprocessing: OpenCV/Pillow/SciPy — Hough-line or min-area-rect deskew; Sauvola adaptive binarization (drops pantograph, keeps ink)
2. Layout detection: YOLOv11-OBB (oriented bounding boxes) or LayoutLMv3, fine-tuned on cheque fields [MICR band, CAR box, LAR line, date, payee, signatures]
3. Field recognition: MICR reader (E-13B), TrOCR-Handwritten for CAR, TrOCR + EasyOCR + tafqeet for LAR
4. Signature: YOLO11 crop -> morphological thinning + ink stroke extraction -> Siamese/DetailSemNet metric model vs core-banking specimen vectors
5. Serving & audit: FastAPI (Python) + Celery/Redis queue + PostgreSQL for immutable audit logs; React/Vite dashboard with side-by-side verification and manual override controls

## Curated Research Index
- awesome-signature-verification: https://github.com/soumitri2001/awesome-signature-verification — papers, datasets (CEDAR, GPDS, BHSig260), deep learning code
