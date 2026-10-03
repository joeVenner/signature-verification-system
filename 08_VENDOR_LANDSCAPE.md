# Commercial Vendor Landscape (2025-2026)

## Vendor Comparison

| Vendor & Product | Primary capabilities | Signature verification | Deployment | Footprint |
|---|---|---|---|---|
| Parascript (CheckXpert.AI, SignatureXpert.AI) | Gold standard cheque automation: MICR, CAR/LAR, payee verification, alteration detection | SignatureXpert.AI: offline signature detection & verification against 1-N reference cards; stroke/shape/morphometric analysis; high forgery detection | On-prem SDK (C++, .NET, Linux/Windows), Docker, private-cloud REST API | OEM inside bank cheque sorters (Burroughs, NCR, Panini), Fiserv, Jack Henry; >2B cheques/yr globally |
| Tungsten Automation, ex-Kofax (FraudOne, SignCheck, SignTeller) | Enterprise cheque processing & IDP; Softpro-origin signature management | FraudOne/SignCheck: batch comparison against central signature DB (SignBase); SignTeller (STV): real-time teller verification | On-prem (Windows/Linux/Oracle/MSSQL), hybrid, private cloud | Tier-1 global (Deutsche Bank, HSBC, Barclays); standard across EMEA |
| Mitek Systems (A2iA CheckReader, Check Fraud Defender) | A2iA pioneer of handwriting & cheque OCR; mobile deposit capture leader; consortium fraud network | Signature presence, layout anomalies, handwriting style match, cross-bank duplicate/forgery patterns via consortium data | SaaS (CFD), on-prem SDK (legacy A2iA) | Top-10 US banks, Chase, Wells Fargo, Fiserv network |
| ProgressSoft (PS-ECC, PS-SIG, PS-PDC) | DOMINANT GCC/MENA vendor: national cheque truncation (CTS) & clearing infrastructure | PS-SIG Intelligent Signature Recognition: verifies vs core-banking signature cards; executes corporate mandate matrices (dual signers, limit tiers) | On-prem (bare metal/VMware), private banking cloud, HA active-active clusters; mandate-compliant | CBUAE ICCS (UAE), Jordan, Qatar, Oman, Kuwait central banks; 350+ commercial banks |
| Hyperscience (Platform) | Modern IDP with proprietary deep learning for handwritten forms & cheques | Signature presence/absence + bounding boxes; biometric verification handed off to metric models | Cloud SaaS, on-prem Kubernetes/OpenShift, hybrid | US/UK financial institutions, TD Bank, QBE, government revenue authorities |
| NCR Atleos / Voyix (ImageMark, APTRA Passport) | Cheque processing, ATM deposit capture, high-speed sorting, clearing archives | Embeds licensed Parascript or FraudOne engines under the hood | On-prem embedded (ATM/teller transports), central clearing servers | Thousands of banks; dominant ATM share in North America & GCC |
| SmartSoft (Cheque OCR SDK) | E-13B/CMC-7 MICR, CAR/LAR extraction, field segmentation | Basic signature detection (presence/zone); separate engine needed for verification | On-prem SDK (Windows/Linux, C++, C#, Java) | European & Latin American regional banks, scanner integrators |
| CR2 (BankWorld) | Omnichannel digital banking + ATM self-service with cheque deposit module | Image capture & field validation at ATM/kiosk; verification handed to core banking or FraudOne/ProgressSoft | Private cloud, on-prem | MENA & Africa (Mashreq, Jordan Kuwait Bank, Standard Chartered ME) |
| ID R&D / Onfido (Entrust) | Biometric identity verification (face/voice), digital KYC | Not cheque engines — used for specimen onboarding (signature pads, KYC mandate cards) | SaaS API, mobile SDK, hybrid | Revolut, Curve, tier-2 retail banks |
| Sigma AI / Sigma Software | Custom AI engineering & annotation for banking IDP | Bespoke Siamese/ViT signature pipelines built per institution | Custom, client-owned cloud/on-prem | European & GCC custom engagements |

## Pricing Realities (published structures, not quotes)
- Commercial cheque SDKs (Parascript, Tungsten): $40,000-$150,000 upfront enterprise base license/server-core fee + annual maintenance 18-22% + volume royalty $0.02-$0.12 per cheque at 1M-10M cheques/yr
- Cloud IDP APIs (AWS Textract, Azure AI Document Intelligence, Google Document AI): $1.50-$10.00 per 1,000 pages (~$0.0015-$0.01/call) — but they LACK cheque-specific features: no E-13B MICR validation, no legal-vs-courtesy normalizer, no forensic signature verification vs mandate cards
- Hyperscience/Mitek SaaS: subscription-based; quote-driven

## GCC-Specific Note
ProgressSoft is embedded in the UAE's national clearing infrastructure (CBUAE ICCS) and the central banks of Jordan, Qatar, Oman, and Kuwait. For regulated UAE clearing, integration compliance (image specs, ANSI X9 standards, PKI signing, XML clearing records) is as much a barrier as the ML itself.
