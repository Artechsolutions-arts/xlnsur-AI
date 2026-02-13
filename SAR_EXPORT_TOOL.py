
import json
import os
import argparse
from datetime import datetime

class SARExporter:
    """
    Automated Subject Access Request (SAR) Export Tool.
    Compliant with GDPR Article 15 and India DPDP Section 11.
    Searches chained audit logs for user data and exports a portable JSON.
    """
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = os.path.abspath(log_file)

    def export_user_data(self, user_id, output_dir="sar_exports"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        export_file = os.path.join(output_dir, f"SAR_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        user_events = []
        
        if not os.path.exists(self.log_file):
            print("❌ No audit logs found.")
            return

        print(f"🔍 Searching logs for User: {user_id}...")
        
        with open(self.log_file, "r", encoding='utf-8') as f:
            for line in f:
                event = json.loads(line)
                if event.get("user_id") == user_id:
                    # Remove internal hashes for the user-facing report
                    clean_event = {
                        "timestamp": event.get("timestamp"),
                        "interaction": event.get("context"),
                        "pii_redacted": event.get("pii_detected"),
                        "confidence_score": event.get("confidence")
                    }
                    user_events.append(clean_event)

        if not user_events:
            print(f"⚠️ No data found for user ID: {user_id}")
            return

        with open(export_file, "w", encoding='utf-8') as out:
            json.dump({
                "report_metadata": {
                    "user_id": user_id,
                    "request_date": datetime.utcnow().isoformat(),
                    "total_records": len(user_events),
                    "compliance_standard": "GDPR/DPDP"
                },
                "audit_entries": user_events
            }, out, indent=4)
        
        print(f"✅ Export Complete: {export_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR Export Utility")
    parser.add_argument("--user", required=True, help="User ID to export data for")
    args = parser.parse_args()
    
    exporter = SARExporter()
    exporter.export_user_data(args.user)
