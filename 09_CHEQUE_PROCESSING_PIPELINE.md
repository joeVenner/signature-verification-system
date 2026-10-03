# Cheque Processing Pipeline, Risk Controls & UAE Regulation

## End-to-End Flow
Capture (IQA/IUA quality gate) -> MICR code-line -> Field extraction (CAR/LAR/payee/date/endorsement) -> Arbitration (CAR vs LAR, date logic) -> Signature verification + mandate matrix -> Risk controls (positive pay, duplicates, alteration forensics) -> Three-tier routing (STP / L1 review / four-eyes) -> Clearing transmission or return.

## Capture & Image Quality
- Front & rear capture at 200-300 DPI; grayscale, bi-level, color and UV channels
- IQA/IUA per ANSI X9.100-181 (US) and CBUAE image specifications (UAE): skew, darkness, brightness, folded corners, torn edges, framing
- Failed QA -> reject at the capture device with a re-scan prompt (never propagate bad images downstream)

## MICR Code-Line
- E-13B font in GCC/US; CMC-7 in parts of Europe/LatAm
- Magnetic read head + optical OCR dual-read ensures >99.9% read rates; mismatch triggers optical re-read
- Parse: [Cheque No.][Bank/Branch Routing Sort Code][Account No.][Transaction Code]
- Modulo-10/modulo-11 checksum validation on routing code and account

## CAR/LAR Arbitration (Amount Verification)
- CAR = Courtesy Amount Recognition (numeric box); LAR = Legal Amount Recognition (written words line)
- Rule of law (Bills of Exchange / GCC Commercial Code): "WORDS GOVERN FIGURES" — the written legal amount supersedes the numeric figure in any discrepancy
- Automated parser converts LAR text to numbers, including Arabic tafqeet (e.g., "فقط مائة وعشرون ألف درهم لا غير")
- CAR == LAR -> auto-confirmed; CAR != LAR -> exception queue (material alteration risk)

## Date Logic
- Stale cheque: date older than 6 months -> return "Stale Cheque"
- Post-dated cheque (PDC): future date -> divert to automated PDC vault/hold queue until maturity (critical for UAE real-estate rent cheque management)

## Endorsement Checking (rear image)
- Detect endorsement signature presence
- Detect restrictive endorsement stamps ("For Mobile Deposit Only at Bank X") — prevents cross-bank double cashing
- Detect collecting-bank endorsement stamps

## Account Mandate Matrix
- Single signer vs joint signers; "Any 2 of N" corporate rules
- Financial limit tiering: e.g., Signatory A alone up to AED 50k; Signatory A+B required above AED 50k
- Engine verifies: correct number of signatures + each signature genuine + each signer within limit tier

## Fraud Controls
- Positive Pay / Payee Positive Pay: corporates upload issued-cheque batch files; engine runs 4-way match (Account, Cheque No., Amount, Payee). Payee mismatch -> freeze (check-washing/payee-alteration signal). Exact match can auto-bypass manual review
- Reverse Positive Pay: daily morning alerts to corporate treasury listing presented cheques above thresholds; client approves/returns via online banking before clearing cut-off
- Duplicate presentment check: internal bank DB + central truncation system index (same cheque image presented twice)
- Alteration & anti-tampering forensics: UV/color scans detect erased ink, chemical washing, erasure residue

## UAE Regulatory Context (Critical)
- CBUAE Image Cheque Clearing System (ICCS): cheques truncated at collecting bank (physical paper never moves); CBUAE transmits ANSI-compliant digital images + clearing XML between collecting and paying banks; same-day/T+1 windows
- UAE Federal Decree-Law No. 14 of 2020 (Commercial Transactions Law amendment, effective 2022): decriminalized bounced cheques (NSF) -> direct civil court execution; mandatory partial payments (drawee bank must clear whatever balance is available)
- Practical implication: automated verification speed directly feeds CBUAE clearing SLAs, and every reject/return decision has direct legal-execution consequences -> audit completeness is non-negotiable

## International Comparisons
- US: Check 21 Act + ECCHO rules; IRD (Image Replacement Document) standards; warranties against altered/forged items
- India: NPCI CTS-2010 standards (mandatory watermarks, VOID pantograph, standardized MICR band)
