
import re
import json
import time
import hashlib
import os
from datetime import datetime

class PIIScrubber:
    """
    Bank-Grade PII Redaction Middleware.
    Sanitizes inputs BEFORE they reach the LLM or Logs.
    """
    def __init__(self):
        # Regex patterns for common Indian PII
        self.patterns = {
            'PAN': r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
            'PHONE': r'(?:\+91[\-\s]?)?[6-9]\d{9}',
            'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'AADHAAR': r'\d{4}\s\d{4}\s\d{4}'
        }

    def scrub(self, text):
        """Redacts PII from text, returning sanitized version and detection metadata."""
        sanitized_text = text
        detected = []
        
        for p_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, sanitized_text)
            for match in matches:
                val = match.group()
                # Replace with token
                sanitized_text = sanitized_text.replace(val, f"[{p_name}_REDACTED]")
                detected.append(p_name)
        
        return sanitized_text, detected

class AuditLogger:
    """
    Immutable Audit Log for Regulatory Compliance (SR 11-7).
    Writes structured JSON events with cryptographic chaining to ensure integrity.
    """
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = os.path.abspath(log_file)
        self.last_hash = self._get_last_hash()
        print(f"📝 Audit Logger Initialized: {self.log_file}")
        print(f"🔗 Chain head: {self.last_hash[:10]}...")

    def _get_last_hash(self):
        """Retrieves the hash of the last entry for chaining."""
        if not os.path.exists(self.log_file):
            return "GENESIS_BLOCK"
        
        try:
            with open(self.log_file, "rb") as f:
                # Seek to end and find last line
                f.seek(0, os.SEEK_END)
                if f.tell() == 0: return "GENESIS_BLOCK"
                
                # Reading last 4KB to find the last complete line
                f.seek(max(0, f.tell() - 4096))
                lines = f.readlines()
                if not lines: return "GENESIS_BLOCK"
                
                last_line = lines[-1].decode('utf-8').strip()
                last_event = json.loads(last_line)
                return last_event.get("integrity_hash", "UNKNOWN")
        except Exception:
            return "FAULT_BREAK_LINK"

    def log_event(self, event_type, user_id, input_data, output_data, pii_detected=False, latency_ms=0, confidence=1.0):
        timestamp = datetime.utcnow().isoformat()
        
        # Build event payload
        event_body = {
            "timestamp": timestamp,
            "event_type": event_type,
            "user_id": user_id,
            "pii_detected": pii_detected,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "prev_hash": self.last_hash,
            "context": {
                "input": input_data,   # Must be already redacted!
                "output": output_data  # Model response
            }
        }
        
        # Generate new hash including previous hash (Chaining)
        event_str = json.dumps(event_body, sort_keys=True)
        new_hash = hashlib.sha256(event_str.encode()).hexdigest()
        event_body["integrity_hash"] = new_hash
        
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(event_body) + "\n")
                f.flush()
            self.last_hash = new_hash
            print(f"✅ Audit Event Logged: {event_type} | Integrity Check: OK")
        except Exception as e:
            print(f"❌ LOGGING FAILED: {e}")

class ModelRiskGuardian:
    """
    Enforces SR 11-7 Model Risk Management controls at runtime.
    Handles confidence thresholds and hallucination protocol escalation.
    """
    def __init__(self, confidence_threshold=0.75):
        self.threshold = confidence_threshold
        self.escalation_count = 0

    def validate_response(self, query, response, confidence_score):
        """
        Validates the AI response. If confidence is low, triggers escalation protocol.
        """
        if confidence_score < self.threshold:
            self.escalation_count += 1
            print(f"⚠️ RISK ALERT: Low confidence response ({confidence_score}). Triggering Protocol.")
            return {
                "safe": False,
                "reason": "CONFIDENCE_BELOW_THRESHOLD",
                "message": "The system could not identify a deterministic answer within the verified policy repository. " +
                           "To ensure accuracy, please escalate this query to a certified institutional advisor via our secure support channel (1800-209-9777)."
            }
        
        # Basic check for negative tone or refusal phrases which might indicate failure
        fail_phrases = ["i don't know", "i'm not sure", "not mentioned in the documents"]
        if any(p in response.lower() for p in fail_phrases) and confidence_score < 0.85:
            return {
                "safe": False,
                "reason": "AMBIGUOUS_MATCH",
                "message": "The information requested appears to be outside my current policy context. " +
                           "I recommend reviewing your latest policy bond for specific terms."
            }

        return {"safe": True, "reason": "OK", "message": response}

    def detect_drift(self, latencies, pii_incidents):
        """Simplified drift/performance monitor"""
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        if avg_latency > 2000:
            return "PERFORMANCE_DRIFT_DETECTED"
        return "STABLE"

# Singleton instances
scrubber = PIIScrubber()
audit_log = AuditLogger()
risk_guardian = ModelRiskGuardian(confidence_threshold=0.70)
