# Signature & Cheque Verification Dataset

Curated benchmark dataset for offline signature verification, forgery detection, and full cheque document analysis.

## Dataset Structure

```text
signature_verification_system/data/samples/
├── dataset_summary.json         # High-level dataset metadata & counts
├── genuine_pairs/               # Intra-writer genuine variations (50 images / 25 pairs)
│   ├── manifest.json            # Pair mapping (author ID, turns, filenames)
│   ├── pair_00_synthetic_*      # Synthetic reference + jittered variant
│   └── pair_01..24_cedar_*      # CEDAR benchmark intra-writer signing turns
├── skilled_forgeries/           # Practiced forgeries imitating target author (50 images / 25 pairs)
│   ├── manifest.json            # Target author, forgery turn, style details
│   ├── pair_00_synthetic_*      # Synthetic envelope-mimic forgery
│   └── pair_01..24_cedar_*      # CEDAR benchmark skilled imitations
├── random_forgeries/            # Cross-author unskilled forgeries (50 images / 25 pairs)
│   ├── manifest.json            # Disparate author pairs
│   ├── pair_00_synthetic_*      # Different synthetic font/stroke authors
│   └── pair_01..24_cedar_*      # CEDAR cross-writer random pairs
└── cheques/                     # Full bank cheque documents (26 images)
    ├── manifest.json            # Cheque resolution, category, bank, payee, amount
    ├── cheque_real_scanned_*.jpg       # 10 scanned real bank cheques with security pantographs
    ├── cheque_benchmark_synth_*.jpg   # 10 alpha-brain high-resolution synthetic cheques
    ├── cheque_procedural_pantograph_* # 5 procedural cheques with multi-wave guilloche security
    └── cheque_workspace_example.png   # Full check specimen from workspace root
```

## Summary Statistics

| Category | Pairs | Total Images | Image Types / Resolutions | Sources |
|---|---|---|---|---|
| **genuine_pairs** | 25 | 50 | 256×256 Grayscale PNG, 900×300 PNG | CEDAR Benchmark (12 writers, multiple turns), Synthetic vector baseline |
| **skilled_forgeries** | 25 | 50 | 256×256 Grayscale PNG, 900×300 PNG | CEDAR Skilled Forgery Benchmark, Synthetic structural envelope forgery |
| **random_forgeries** | 25 | 50 | 256×256 Grayscale PNG, 900×300 PNG | CEDAR Cross-Author Genuine pairings, Synthetic distinct-author vectors |
| **cheques** | N/A | 26 | 1584×712 to 3024×1724 RGB JPG/PNG | Scanned bank cheques, Alpha-Brain dataset, Mathematical Guilloche procedural engine |
| **TOTAL** | **75 pairs** | **176 images** | | |

## Data Sources & Benchmark Provenance
1. **CEDAR Signature Database**: Center of Excellence for Document Analysis and Recognition, SUNY Buffalo. Standard offline signature verification benchmark with verified genuine signatures and practiced skilled forgeries.
2. **Bank Cheque Validation Dataset**: Real scanned banking cheques with handwritten field segmentation, bank watermarks, security background tints, and MICR code bands.
3. **Alpha-Brain OCR Synthetic Cheque Dataset**: Ultra-high-resolution (2365×1100) bank cheques with structured JSON metadata for payee, legal amount, courtesy amount, date, and signature field.
4. **Guilloche Security Pantograph Generator**: Procedural generator utilizing multi-harmonic sinusoidal wave interference ($y = A \sin(\omega_1 x) \cos(\omega_2 y) + \sin(\omega_3 (x+y))$) producing anti-copy security pantographs, microprint security text, and embedded signatures for cheque localization tests.
