# 🛡️ xInsur AI Finance: Tier-1 Bank-Grade AI Advisor
## Enterprise Financial Intelligence Layer (EFIL)

xInsur AI is NOT a generic chatbot. It is a **hardened, auditable, and SR 11-7 compliant** financial intelligence middleware designed for direct integration with Tier-1 Retail Banks and Systemically Important NBFCs.

---

## 🏛️ Enterprise Core Pillars

### 1. **Regulatory Governance (SR 11-7)**
   - **Model Risk Guardian**: Active runtime monitoring that intercepts low-confidence responses and enforces human-in-the-loop (HITL) escalation protocols.
   - **Independent Validation**: Architected for three-lines-of-defense governance, separating engineering, risk, and internal audit.
   - **Model Registry**: Full lifecycle tracking from development to secure retirement.

### 2. **Zero-Trust Security & Privacy**
   - **PII Scrubbing Middleware**: Automatic regex/NER-based masking of sensitive data (PAN, Aadhaar, Phone, Email) *before* it leaves the bank's secure perimeter.
   - **WORM Audit Trails**: Write-Once-Read-Many logging with **Cryptographic Chaining**. Every interaction is immutably linked to the previous entry, preventing any unauthorized modification or deletion.
   - **AES-256 Encryption**: Hardware-hardened encryption (HSM/KMS compatible) for all data-at-rest.

### 3. **Compliance Readiness**
   - **GDPR/DPDP SAR Tool**: Integrated Subject Access Request (SAR) automation for 24-hour user data export.
   - **Breach Notification Protocol**: Pre-configured 72-hour regulatory workflow for CERT-In/EBA reporting.
   - **Data Lineage Map**: Complete visibility of data flows from PDF ingestion to AI generation.

---

## 🚀 Advanced RAG & Reasoning
- **Engine**: Meta Llama-4-Maverick (17B) via NVIDIA NIM.
- **Hybrid Retrieval**: Jina V4 Embeddings + FAISS + Keyword Fallback.
- **Source-Pinned Citations**: Every claim is traced back to a specific ICICI Prudential policy chunk with verified context.

---

## 🛠️ Compliance Tech Stack

| Component | Technology | Banking Standard |
| :--- | :--- | :--- |
| **Governance Hub** | `governance.py` | SR 11-7 / Model Risk |
| **Audit Storage** | `audit_log.jsonl` | WORM / SEC 17a-4 |
| **PII Protection** | `PIIScrubber` | GDPR / DPDP Article 25 |
| **SAR Tool** | `SAR_EXPORT_TOOL.py` | GDPR Article 15 |
| **Backend** | FastAPI (SSL-Pinned) | SOC 2 Security Pillar |

---

## 📂 Project Governance Map

```bash
Insurance_ChatBot/
├── PROJECT_OVERVIEW.md                 # Strategic Framework Overview
├── DATA_LINEAGE_MAP.md                  # Data Classification
├── BREACH_NOTIFICATION_WORKFLOW.md      # Incident Protocol
├── governance.py                        # Risk & Audit Middleware
├── SAR_EXPORT_TOOL.py                   # Compliance Automation
├── backend.py                           # Logic Core (Hardened)
├── index.html                           # Premium Banking Interface
└── README.md                            # Primary Documentation
```

---

## ▶️ Deployment Protocol

### 1. Provision Environment
```bash
# Securely build the production-ready container
docker-compose up --build -d
```

### 2. Verify Governance Sync
Access the **System Health Report** in the web dashboard or via:
```bash
GET /health
```
A `healthy` status indicates that the PII Scrubber and WORM Audit Link are active.

### 3. Run Compliance Audit
```bash
python SAR_EXPORT_TOOL.py --user user_session_1
```
This generates a portable JSON report of all interactions linked to the session, fulfilling SAR requirements.

---

## ⚖️ Disclaimer
*This system is an AI-powered advisory middleware. Output is based on retrieved policy documentation and is provided for informational purposes only. Direct human consultation is advised for high-impact financial decisions.*
