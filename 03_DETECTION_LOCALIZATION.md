# Stage 1 — Signature Detection & Localization

## Why It Is Non-Trivial
Signatures never exist in isolation. On cheques and forms they are embedded in guilloche/pantograph security patterns, pre-printed labels ("Authorized Signatory"), baseline rules, corporate logos, and overlapping rubber stamps. Naive cropping corrupts the verification input.

## Detection Tool Comparison (2025-2026)

| Component | Best tools | Performance |
|---|---|---|
| Coarse detection | YOLO11x / DocLayout-YOLO (fine-tuned on DocLayNet, Tobacco-800, RVL-CDIP) | >94.2% mAP50:95 on signature localization, <8 ms/page on modern GPU |
| Semantic form parsing | LayoutLMv3 / DiT — uses OCR text + coordinates + image patches; finds the box preceded by anchor text (e.g., "Signature of Drawer") | >97% F1; ignores decorative flourishes and logos |
| Legacy backends | Faster R-CNN / Mask R-CNN (ResNet-50/101-FPN) | 40-60 ms/page; instance pixel masks separate signature from adjacent print |

Production choice: YOLO11x for speed, cross-referenced with LayoutLMv3 when multiple textual candidate fields exist.

## Cheque-Specific Preprocessing (Ink Extraction)
1. Guilloche/safety-tint removal — Naive Otsu binarization merges fine geometric patterns with thin pen strokes. Use adaptive Sauvola/Wolf binarization or a deep semantic binarization U-Net trained on DIBCO benchmarks to isolate high-frequency ink strokes from low-contrast background waves.
2. Ink vs. pre-printed stock separation — Convert RGB to CIE-Lab or HSV. Blue-ink signatures separate cleanly in the Lab b* chromaticity channel; pastel pantograph colors collapse into predictable luminance bands; K-Means/GMM chromatic clustering masks non-signature print.
3. Baseline removal — Morphological 1xW horizontal structuring elements detect guide lines, then directional inpainting removes the line while preserving vertical stroke descenders (g, y, f loops) crossing it.
4. Stamp elimination — Corporate stamps (red/violet/green) overlap genuine signatures. Deploy deep color-unmixing networks or Pix2Pix/CycleGAN segmentation to subtract the stamp ink layer before verification.
5. Pseudo-dynamic recovery — Ballpoint/fountain pens deposit variable ink by pressure & speed: fast ballistic strokes = lighter ink + feathered edges; hesitation = ink blobbing + wider tracks. Preserve grayscale gradients — do not binarize away the dynamism evidence needed for traced-forgery detection.

## Pitfalls (Rookie Mistakes)
- Otsu binarization on security-patterned cheque stock (merges pattern with strokes).
- Tight box cropping that clips descenders/ascenders — pad the crop.
- Single-signature assumption — corporate cheques have multiple signature zones (joint mandates); detect all zones then resolve via the mandate matrix.
- Forgetting the endorsement zone on the rear image.
