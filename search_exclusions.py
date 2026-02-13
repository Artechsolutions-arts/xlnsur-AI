
import pdfplumber
import re

def find_exclusions():
    with pdfplumber.open('ICICI_Insurance.pdf') as pdf:
        all_text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                all_text += content
    
    # Search for accidental death benefit section and its exclusions
    pattern = re.compile(r"Accidental Death Benefit.*?Exclusions", re.IGNORECASE | re.DOTALL)
    match = pattern.search(all_text)
    if match:
        start = match.start()
        print(all_text[start:start+3000])
    else:
        # Just find any exclusions section
        print("Standard Exclusions search:")
        ex_pattern = re.compile(r"Exclusions.*?\n", re.IGNORECASE)
        for m in ex_pattern.finditer(all_text):
            print(f"--- Found Exclusions at index {m.start()} ---")
            print(all_text[m.start():m.start()+1000])

if __name__ == "__main__":
    find_exclusions()
