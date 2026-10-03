# Supporting Research — Handwriting OCR for Cheque Fields (CAR/LAR, Payee, Date)

## Why This Is a Different Problem From Signature Verification
Signature verification is biometric authentication (is this the same hand, with micro-stroke forensics) — VLMs fail it (file 05). Field extraction is text recognition (what does it say) — and here modern VLMs are SOTA. Never conflate the two: use a metric model for the signature, use document VLMs/OCR for the fields.

## The 2025-2026 Paradigm: End-to-End VLM Document Parsing
The 2023-2024 multi-stage pipeline (binarize -> segment lines -> per-line OCR -> stitch with regex) is obsolete. Dynamic-resolution Vision-Language Models process the full page in a single forward pass with native spatial grounding:

| Dimension | Legacy 2024 | Modern 2025-2026 |
|---|---|---|
| Pipeline | Binarize -> segment -> CRNN/TrOCR -> LLM | Direct end-to-end multimodal VLM |
| Resolution | Static crops | Dynamic 2D-RoPE / native tiling (no downscaling) |
| Layout | Heuristic sorting | Native visual layout attention |
| Ambiguity | Uncalibrated OCR confidence | Deep language priors resolve sloppy cursive |
| Output | Raw lines stitched with regex | Direct schema generation (Markdown/JSON) |
| Latency | 100+ sequential GPU calls | Single batched pass (vLLM/SGLang) |

## Best Models
- Qwen2.5-VL (7B/72B): leading open model — dynamic-resolution ViT with 2D-RoPE; natively outputs text + bounding boxes
- GOT-OCR 2.0: 580M-param end-to-end OCR engine (CAS/Megvii); unifies handwriting, formulas, dense documents into Markdown/LaTeX single-pass
- InternVL 2.5 / DeepSeek-VL2: dynamic tiling + MoE; high-throughput page transcription
- Claude 3.5/3.7 Sonnet (high-res doc mode): top zero-shot performer for illegible cursive/historical handwriting
- Gemini 2.x Flash/Pro: ultra-long context; multi-hundred-page ledgers without intermediate OCR

## Document Frameworks
- Docling (IBM): unified DocTags representation -> clean Markdown/JSON
- MinerU 2.5/4.0 (PDF-Extract-Kit): end-to-end visual parsing for RAG/agentic workflows
- Marker: modern replacement for Tesseract/PyMuPDF stacks (Surya layout engine)

## Tiered Routing Pattern (Recommended for Our Build)
Tier 1 — fast edge VLM (GOT-OCR 2.0 / Qwen2.5-VL-7B): emits structured Markdown/fields + confidence; auto-accept when high confidence.
Tier 2 — frontier VLM fallback (Claude/Gemini) for messy cursive/low confidence: schema-constrained decoding.
Never: destructive binarization (destroys grayscale gradients), free-form LLM guessing (constrain output space to a JSON schema).

## Cheque Field Specifics
- CAR (numeric box): TrOCR-Handwritten fine-tuned on bank digit styles — or VLM with schema-constrained numeric output
- LAR (words): OCR + number-to-words normalization incl. Arabic tafqeet (تفقيط); then CAR/LAR arbitration — words govern figures (file 09)
- MICR band: E-13B via Tesseract traineddata or custom CRNN — VLMs are NOT the tool here (specialized font, checksums)
- Date: OCR + format normalization (DD/MM/YYYY vs MM/DD/YYYY ambiguity)
- Payee: fuzzy/phonetic match vs depositing account title (Levenshtein)

## Full Deep-Dive
See 14_HANDWRITTEN_OCR_PLAYBOOK.md (earlier research on complex handwritten OCR, expert systems, and HITL patterns).
