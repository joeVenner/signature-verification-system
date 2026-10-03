# Extracting Valid Data from Complex Handwritten Documents
### An expert playbook for OCR + LLM pipelines (Azure Document Intelligence + GPT-class LLM)

*Compiled from Microsoft Learn docs, arXiv papers, engineering blogs (LlamaIndex, Vellum, Labelbox, Scale), and practitioner forums. Vendor-agnostic where possible, with Azure-specific sections.*

---

## 0. The core mental model

Handwriting is fundamentally different from print: extreme intra-author variance, irregular baselines, touching ascenders/descenders, ink bleed-through, strikethroughs, and archaic/ambiguous characters. **No single model solves it.** Every expert recommendation reduces to the same five-stage architecture:

```
Image → Preprocess → Segment/Layout → Multi-engine OCR → LLM post-process → Validate → HITL
```

The logic that separates good systems from bad ones is **not** "use a better OCR engine" — it's:

1. **Disagree-and-gate**: run multiple diverse engines, trust only where they agree, route the rest to a smarter model or a human.
2. **Confidence is a signal, not a guarantee** — calibrate it, then make routing decisions on it.
3. **Never let an LLM freely guess** — constrain its output space and verify against evidence.
4. **Deterministic rules catch what neural models miss** (check digits, totals, regex, cross-field consistency).

---

## 1. Image preprocessing (do this first — it's the highest-ROI step)

Raw low-contrast, skewed, warped scans destroy accuracy before OCR even starts.

| Step | Technique | Why |
| :--- | :--- | :--- |
| **Denoising** | `cv2.bilateralFilter` (d=9, sigma=75/75) or Fast Non-Local Means | Preserve stroke edges while removing paper grain. Median filter (ksize=3) only for salt-and-pepper. |
| **Dewarping** | DocTr / DewarpNet / page-dewarp | Flatten page curl from bound ledgers / phone captures. |
| **Deskewing** | Hough lines / `cv2.minAreaRect` / projection-profile variance | Correct rotation. For handwriting, do it after coarse line extraction. |
| **Super-resolution** | Real-ESRGAN (RealESRGAN_x4plus) | For scans <300 DPI where 'e/a/o' loops merge into blobs. |
| **Binarization** | Sauvola / Nick adaptive thresholding, or a U-Net binarizer | Convert to clean black/white. NOT global Otsu on uneven illumination. |
| **Upscale check** | Validate scan quality | Text must be ≥12 px height (~8pt @ 150 DPI) for Azure; below that it's skipped. |

**Rule of thumb:** reject or re-request the scan at intake if blur/contrast/skew are unreadable, rather than pouring bad images through an expensive pipeline.

---

## 2. Layout & line segmentation

Transcription fails if line extraction clips ascenders/descenders or merges loops.

- **Surya** — best modern open-source document segmenter for messy, non-linear handwritten layouts and reading order.
- **DocLayout-YOLO / YOLOv11** — macro layout: isolate handwritten marginalia, signatures, stamps vs printed body text at high FPS.
- **DBNet / DBNet++** — standard baseline (PaddleOCR/docTR) but rigid hulls can bisect overlapping handwritten loops.
- **CRAFT / EasyOCR's detector** — **avoid** for cursive: character-center isolation catastrophically breaks on connected script.
- **Detectron2 (Mask R-CNN / LayoutLMv3)** — high-precision but heavy.

---

## 3. OCR engines — what actually works on handwriting

Benchmark figures (IAM / unconstrained handwriting, line-level CER):

| Engine | Handwriting fit | Typical CER | Verdict |
| :--- | :--- | :--- | :--- |
| **TrOCR-Large** (fine-tuned) | Exceptional | **~2.9%** | Best specialized open-source engine; seq2seq avoids segmentation failures. |
| **Google Cloud Vision** | Excellent | ~8–12% | Best general commercial API for messy slanted cursive. |
| **Frontier VLM (GPT-4o / GPT-5 / Gemini / Qwen2.5-VL)** | Excellent (with fidelity prompting) | ~1.2–11% | Best overall when you can afford it; watch for "autocorrect" hallucination. |
| **AWS Textract** | Good | ~10–15% | Strong for form fields/tables; lags Google on freeform cursive. |
| **PaddleOCR (v4/v5)** | Moderate | ~25–35% | High throughput; DBNet clips ascenders unless retrained. |
| **docTR** | Moderate | ~30–40% | Printed-oriented default weights. |
| **EasyOCR** | Poor | ~50–60% | CRAFT breaks on connected cursive. |
| **Tesseract 5** | Very poor | ~55–68% | Unusable for degraded cursive. |

**Key insight:** CTC/character-isolation engines (Tesseract, EasyOCR) fail on cursive. Sequence-to-sequence transformers (TrOCR) and large multimodal models (VLM) dominate.

**Recommended open-source stack for handwriting:**
`DocLayout-YOLO (zones) → Surya (lines) → TrOCR-Large-Handwritten (transcribe) or Qwen2.5-VL (heavy degradation) → local LLM post-correction`

---

## 4. Azure Document Intelligence — specific guidance (your current stack)

### Model selection
- **`prebuilt-read`** — base OCR, handles print + handwriting, returns `styles[].isHandwritten` flag with confidence and offsets. **Use this flag to programmatically isolate handwritten regions from printed boilerplate.**
- **`prebuilt-layout`** — adds tables, checkboxes, reading order, and (v4) markdown output.
- **Custom models:** use **`buildMode: "neural"`**, not template. Template models rely on fixed pixel geometry and fail on variable handwriting. Neural models use transformers + semantic context.
  - Minimum 5 labeled docs; **experts recommend 10–15+ samples per layout variation AND per penmanship style** (neat + cursive + messy).
  - Label field tokens contiguously in reading order, or define an explicit region bbox.
- **Version:** target **REST API `2024-11-30` GA (v4.0)** — adds signature detection in custom neural, table cell-level confidence, overlapping fields support, Batch API, query fields, and markdown output. (v2.1 retires 2027-09-15; v3.0 retires 2029-03-30.)

### Confidence scores — read them correctly
- `documents[].fields[].confidence` = probability the tokens map to the *target field*.
- `pages[].words[].confidence` = probability the *character transcription* is correct.
- These are **different things** — a field can be perfectly located but mis-read (and vice versa). Don't conflate them.

### Known failure modes to plan around
- `0`↔`O`, `1`↔`l`↔`I`, `5`↔`S`, `8`↔`B`, `rn`↔`m`, `cl`↔`d`
- Strikethroughs and between-line marginalia corrupt reading order and neighboring key-values
- Low-contrast / carbon copies / bleed-through from double-sided paper
- No cross-page reading order reconstruction (handle multi-page yourself)

---

## 5. The LLM post-processing layer (your GPT setup)

This is where you recover from OCR errors. The single most important rule:

> **Give the LLM the IMAGE CROP + the raw OCR drafts + a strict JSON schema, and forbid it from "fixing" what it can't see.**

### 5.1 Multi-engine voting (before the LLM)
- Run Azure + a second diverse engine (PaddleOCR/TrOCR/Google Vision). **Agreement = trust.** Disagreement = escalate.
- Merge bounding boxes with **Intersection-over-Minimum (IoM)** instead of IoU (handles line-vs-word segmentation mismatch).
- Align hypotheses with **ROVER / Needleman-Wunsch** multiple-sequence alignment; build consensus column-by-column by plurality or confidence-weighted vote.
- Track an `engines_agreed` counter per field. If all engines agree, skip the LLM; if not, route the crop to a VLM arbiter.

### 5.2 Constrained / structured decoding
- Force JSON schema (function calling / JSON mode / pydantic). Use **regex/FSM-constrained generation** (Outlines, Guidance, or native structured outputs) so critical fields (IDs, dates, amounts) can only emit valid token patterns.
- Use **closed-choice decoding**: for fields that must come from a reference set (product codes, taxonomies), constrain the model to only emit tokens from that set — prevents out-of-vocabulary hallucination.

### 5.3 Self-consistency (multi-sample voting)
- Ask the LLM the same question N times (temperature 0.5–0.7), vote. Especially effective for ambiguous cursive.
- **Test-time perturbation**: apply label-preserving transforms (padding, Gaussian noise, blur, grid-warp) and re-query. Grid-warping decorrelates errors and gives a strong agreement↔accuracy signal (r > 0.64).

### 5.4 Prompting patterns for transcription
- **Verbatim-first:** "Transcribe EXACTLY as written. Do not modernize spelling, expand abbreviations, or correct grammar. Output the exact text or mark UNREADABLE." (Prevents "autocorrect hallucination" where the model turns *Mrytle* into *Myrtle*.)
- **Evidence-anchored:** require the model to return `{value, verbatim_substring, bbox}` and confirm the verbatim substring exists in the OCR text layer within that bbox.
- **Isolated-crop verification:** crop the field (with ~10–15% padding), send ONLY the crop (not the whole page) with: *"Does this snippet read '{candidate}'? → {verified, confidence, alternate_text}"*. Removing page-level context stops the model from "plausibly guessing" from neighbors.
- **Glass-box reasoning field:** ask for a one-line justification; it measurably reduces hallucination.

### 5.5 Region-of-interest / chunking (long docs)
- Don't feed a whole multipage doc and ask for one field. Crop regions, or process page-by-page with overlap (20% sliding window).
- Multi-page reading order must be reconstructed by you (Azure doesn't do it).

---

## 6. Confidence & calibration (knowing when to trust output)

- **Geometric-mean field confidence:** combine OCR character probability with LLM token likelihood: `C_field = sqrt(P_OCR · P_LLM)`.
- **Token entropy / normalized perplexity** (if you have logprobs): sharp entropy spikes pinpoint unreadable strokes.
- **Split conformal prediction:** on a held-out calibration set, compute a non-conformity score (1 − Levenshtein similarity); the (1−α) quantile gives a *mathematically guaranteed* error bound. Escalate any field whose uncertainty set has >1 candidate.
- **Tiered routing (example thresholds — calibrate to your data):**
  - `C ≥ 0.92` → auto-approve (STP) if deterministic tests pass
  - `0.70 ≤ C < 0.92` → micro-task human review
  - `C < 0.70` → full manual review / re-scan request

---

## 7. Deterministic validation (catches what neural models miss)

Always run these BEFORE trusting a field. This is cheap, exact, and hugely reduces errors:

- **Regex schemas** per field (dates, phone, SSN, IBAN, postcodes).
- **Number/date ranges** (e.g. DOB in the past, invoice date ≤ today).
- **Check digits:** Luhn (cards), Modulo 97/11 (IBAN), etc.
- **Arithmetic consistency:** `Σ line items == subtotal`, `subtotal + tax == total`.
- **Cross-field consistency:** age vs DOB, gender vs name/title, address vs postcode, page totals vs summary totals.
- **Reference reconciliation:** fuzzy-match names/addresses/catalog items against master data with RapidFuzz (Jaro-Winkler / token-set ratio) + phonetic matching (libpostal for addresses).

---

## 8. Hallucination verification (second-pass grounding)

- **Grounded coordinate + quote check:** confirm the verbatim string exists in the OCR text layer enclosed by the predicted bbox.
- **Second-pass verifier / critic agent:** a small, isolated model prompted for binary entailment — *"Does the snippet explicitly contain [Value] for [Field]? YES/NO."*
- **Cycle-consistency (back-projection):** reconstruct source text from the extracted record and compare (BERTScore/BLEU) to the original OCR transcript. Large deviation = hallucinated interpolation.
- **Multi-model consensus entropy:** query 2–3 diverse models, compute entropy of pairwise Levenshtein distances. Low entropy → auto-accept; high entropy → human review. (Reported to improve hallucination-detection F1 by >40% vs a single judge.)

---

## 9. Human-in-the-loop & continuous quality

- **Micro-reviews, not full documents:** deconstruct into single-field tasks. Never serve a whole PDF for one bad field.
- **Field-type batching:** review 50–100 of the same field type in sequence to cut context-switching.
- **Side-by-side UI:** tight crop (~20px padding) next to the prefilled OCR/LLM hypothesis; anti-anchoring blank inputs for critical financial/PII fields.
- **Keyboard-first:** Space=confirm, Tab=edit, 1–9=fuzzy-match options, Esc=illegible.
- **Canary / gold documents:** silently inject pre-verified documents at 2–5% of the queue to detect reviewer drift/fatigue.
- **Metrics to track:** CER/WER for text; **field-level** precision/recall/F1 per schema key (document-level accuracy hides the field errors that matter); **STP rate** = % zero-human-touch-and-valid, targeted at a fixed field error rate (<0.5%).

---

## 10. Reference architecture (summary)

```
[Scanned handwritten doc]
        │
        ▼
[Intake QC]  blur/contrast/skew check → reject or re-request bad scans
        │
        ▼
[Preprocess]  denoise → dewarp → deskew → super-res → binarize
        │
        ▼
[Layout + segment]  DocLayout-YOLO (zones) → Surya (lines)
        │
        ▼
[Multi-engine OCR]  Azure DI (styles.isHandwritten) + PaddleOCR/TrOCR/Google
        │   → IoM merge → ROVER/NW alignment → engines_agreed counter
        ▼
[Agreement / confidence gate]
        │
   ┌────┴────────────────────────────┐
   ▼ (agree)                         ▼ (disagree / low conf)
[Deterministic path]           [VLM path: crop + OCR drafts]
   - regex/checkdigit/totals       - strict JSON schema / FSM decoding
   - master-data reconciliation     - self-consistency (N samples)
   - direct structured output       - evidence-anchored (verbatim + bbox)
        │                              │
        └──────────────┬───────────────┘
                       ▼
              [Validation & guardrails]
   - coordinate/quote grounding · critic agent · cycle-consistency
                       ▼
              [Calibrated routing]
   high → STP · mid → micro-review · low → manual / re-scan
                       ▼
              [Continuous eval]  canaries · field-level F1 · STP rate
```

---

## 11. Concrete next steps for YOUR stack (Azure DI + GPT)

1. **Turn on `styles.isHandwritten`** and treat handwritten regions as high-risk — route them through the VLM path, not the straight path.
2. **Add a second OCR engine** (PaddleOCR or TrOCR, or Google Vision) and gate on cross-engine agreement before calling the LLM.
3. **Use structured outputs / constrained decoding** for IDs, dates, and amounts — never free-text for critical fields.
4. **Prompt the GPT model verbatim-first** with image crops + raw OCR + JSON schema + a "transcribe exactly, else UNREADABLE" instruction.
5. **Build the deterministic validator layer** (regex, check digits, totals, cross-field rules, master-data fuzzy match) — it's cheap and catches the most expensive errors.
6. **Calibrate thresholds on your own gold set** (don't copy the 0.92/0.70 numbers blindly) and track field-level F1 + STP rate.
7. **Use `2024-11-30` (v4.0 GA)** API if you aren't already (signature detection, table cell confidence, batch, markdown).

---

*Sources: Microsoft Learn (Custom Neural Model, Accuracy & Confidence, Add-on Capabilities, What's New 2024-11-30); Li et al., TrOCR (arXiv:2109.10282); Archibald & Martinez (arXiv:2509.09722); Papadopoulos et al., Conformal Prediction (NeurIPS 2024); Yue et al., self-consistency/hallucination (arXiv:2305.14558); Surya/DocLayout-YOLO/Inksight benchmarks; Labelbox & Scale HITL guidelines; LlamaIndex/Vellum OCR+LLM engineering posts.*
