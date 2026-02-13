# xInsur AI: Data Lineage & Classification Map
## Regulatory Transparency Protocol

This document maps the flow of information through the xInsur AI architecture to satisfy institutional data sovereignty and governance requirements.

---

### 📥 1. Ingestion Layer
*   **Source**: `ICICI_Insurance.pdf` (Static Policy Repository).
*   **Classification**: Confidential Business Information (CBI).
*   **Process**: Extracted via `pdfplumber`, chunked for vectorization.

### 🧠 2. Processing & Vectorization
*   **Embedding Engine**: Jina V4 (Private VPC Instance).
*   **Vector Store**: FAISS (Locally Persistent).
*   **Audit**: Metadata includes original page numbers and chunk hashes.

### 🛡️ 3. Governance & Redaction
*   **Middleware**: `governance.py` / `PIIScrubber`.
*   **Action**: Real-time regex/NER screening for PAN, Aadhaar, and PII.
*   **Transformation**: Clear text -> Sanitized JSONL in `audit_log.jsonl`.

### 📤 4. Generation & Delivery
*   **LLM**: Meta Llama-4 (Execution Gateway).
*   **Citations**: Mandatory source-pinning enforced by the RAG controller.
*   **Retention**: Zero-retention policy for user identity after session termination.

---

### ⚖️ Classification Key
| Data Type | Tier | Retention | Encryption |
| :--- | :--- | :--- | :--- |
| Policy Metadata | Tier 3 (Internal) | Persistent | AES-256 |
| User Queries | Tier 1 (Highly Sensitive) | Redacted / 30 Days | AES-256 |
| Audit Logs | Tier 2 (Restricted) | 7 Years (Compliance) | Cryptographically Tied |
