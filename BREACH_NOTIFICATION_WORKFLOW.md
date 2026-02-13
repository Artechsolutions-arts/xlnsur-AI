# xInsur AI: Breach Notification Protocol
## Regulatory Incident Response (72-Hour Workflow)

In compliance with CERT-In and international banking standards, the following protocol is active for all identified security anomalies.

---

### 🚨 Phase 1: Detection & Triage (Hour 0-4)
*   **Trigger**: SHA-256 Integrity check failure in `audit_log.jsonl`.
*   **Action**: Automated session termination and institutional alert.
*   **Contact**: Chief Information Security Officer (CISO).

### 🔍 Phase 2: Impact Assessment (Hour 4-24)
*   **Tool**: `SAR_EXPORT_TOOL.py` execution for affected session buckets.
*   **Goal**: Identify if PII was exposed or if masking protocols were bypassed.
*   **Log**: Immutable breach incident record created in the governance registry.

### 📢 Phase 3: Regulatory Notification (Hour 24-72)
*   **Cert-In**: Mandatory reporting of significant cybersecurity incidents.
*   **Customer Notification**: Trigger email/SMS alerts to affected policyholders if unmasked PII was breached.
*   **Remediation**: Re-encryption of vector stores and key rotation.

---

### 📞 Emergency Contact Registry
*   **Security Ops**: `soc@xinsur.ai`
*   **Compliance**: `compliance-audit@bank.internal`
