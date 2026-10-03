# Build vs Buy — UAE/GCC Strategic Guidance

## The Three Scenarios

### Scenario 1 — Regulated commercial bank clearing via CBUAE: BUY
- Buy: ProgressSoft (PS-SIG + PS-ECC) or Parascript/Tungsten
- Rationale: CBUAE ICCS integration requires certified format compliance (TIFF/JPEG metadata, ANSI X9 standards, PKI digital signing, direct interface with national clearing gateways). ProgressSoft is already embedded in CBUAE and GCC central clearing infrastructures. Building from scratch introduces severe regulatory and clearing-SLA non-compliance risk — the regulatory integration is a higher barrier than the ML.
- Risks of building: certification timelines, clearing SLA penalties, legal exposure on forgery losses.

### Scenario 2 — Fintech, digital lender, real-estate PDC management, neobank: HYBRID
- Build in-house: capture flows, PDC scheduling, React HITL review interfaces, audit trail, integrations
- Buy/OEM as containerized microservices: Parascript SignatureXpert SDK or Mitek Mobile Deposit/CFD for core CAR/LAR + signature verification engines
- Rationale: fastest time-to-market with proven forgery detection, while owning the customer experience and data model.

### Scenario 3 — Auxiliary, non-clearing workflows: BUILD (open-source)
- Viable when formal bank clearing warranties are NOT at stake: real-estate property management tracking post-dated rent cheques, corporate accounts payable matching cheques to invoices, back-office preliminary triage
- Stack: YOLO11 + DetailSemNet/ProtoSig + TrOCR/EasyOCR + FastAPI/PostgreSQL (see file 07)

## Decision Matrix

| Criterion | BUY | HYBRID | BUILD |
|---|---|---|---|
| Formal clearing participation (CBUAE ICCS) | Required | Not required | Not required |
| Time-to-market | Fast | Medium | Slow |
| Forgery-detection maturity | Proven (vendor benchmarks) | Proven (OEM engine) | Must be validated yourself |
| Cost profile | $40k-150k base + $0.02-0.12/cheque + 18-22%/yr | License + build cost | Engineering time only |
| Data sovereignty / customization | Low | High | Full |
| Audit/legal defensibility | Vendor-assisted | Shared | Entirely on you |
| Best fit | Banks | Fintechs/PDC products | Internal/auxiliary tools |

## Cost Comparison (indicative)
- Commercial engine: $40k-150k upfront + 18-22%/yr maintenance + per-cheque royalty ($0.02-0.12 at 1M-10M/yr)
- Cloud IDP (Textract/Azure/Google): $1.50-10.00 per 1,000 pages — but no MICR validation, no CAR/LAR normalizer, no forensic signature verification
- Open-source build: ~2-3 engineer-months for MVP + ongoing model ops; GPU serving costs
