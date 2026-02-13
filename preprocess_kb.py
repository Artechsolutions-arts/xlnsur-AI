"""
Pre-compute KB artifacts locally. Run this ONCE on your machine.
It generates chunks.json and faiss_index.bin that Railway just loads.
"""
import os
import json
import numpy as np
import pdfplumber
import faiss
import requests
import re
import sys

from dotenv import load_dotenv
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

def clean_pdf(text):
    """Clean PDF text by removing headers, footers, and formatting issues (Matches backend.py)"""
    text = re.sub(r'Page \d+\n', '', text)  # Remove page numbers
    text = re.sub(r'ICICI .*?\n', '', text) # Remove ICICI header/footer
    text = re.sub(r'IRDAI Regn.*?\n', '', text) # Remove regulator lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Fix hyphenated words
    text = re.sub(r'[\u2022•]', '-', text) # Replace bullets with dash
    text = re.sub(r'(?<![.!?])\n', ' ', text) # Join lines that don't end with punctuation
    text = re.sub(r'\n+', '\n', text) # Remove extra newlines
    text = re.sub(r' {2,}', ' ', text) # Remove extra spaces
    return text.strip()

def simple_chunk_text(text, chunk_size=500):
    """
    Smarter chunking that respects sentence boundaries. (Matches backend.py)
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
            
        sentence_len = len(sentence)
        
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def embed_with_jina(texts, api_key, batch_size=16):
    JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    all_vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch = texts[i:i+batch_size]
        inputs = [{"text": t[:8000]} for t in batch]
        payload = {
            "model": "jina-embeddings-v4",
            "task": "text-matching",
            "input": inputs
        }
        try:
            print(f"  Embedding batch {batch_num}/{total_batches}...")
            resp = requests.post(JINA_EMBED_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                vectors = [item["embedding"] for item in data["data"]]
                all_vectors.extend(vectors)
            else:
                print(f"  ERROR: Batch {batch_num} failed: {resp.status_code} - {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"  ERROR: {e}")
            return None

    return np.array(all_vectors, dtype="float32")

def main():
    print("=" * 60)
    print("xInsur AI — Local KB Pre-Processor")
    print("=" * 60)

    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set in .env")
        sys.exit(1)

    pdf_path = "ICICI_Insurance.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    # Step 1: Extract PDF
    print(f"\n[1/4] Extracting PDF ({pdf_path})...")
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
            if (i+1) % 20 == 0:
                print(f"  Processed {i+1}/{len(pdf.pages)} pages")
        print(f"  Done: {len(pdf.pages)} pages extracted")

    combined = "\n".join(texts)
    cleaned = clean_pdf(combined)
    print(f"  Text length: {len(cleaned)} chars")

    # Step 2: Chunk
    print("\n[2/4] Chunking text...")
    chunks = simple_chunk_text(cleaned, 500)
    print(f"  Generated {len(chunks)} chunks")

    # Step 3: Save chunks
    print("\n[3/4] Saving chunks.json...")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  Saved chunks.json ({os.path.getsize('chunks.json')} bytes)")

    # Step 4: Embed and save FAISS index
    print(f"\n[4/4] Building embeddings ({len(chunks)} chunks)...")
    embeds = embed_with_jina(chunks, JINA_API_KEY)

    if embeds is not None:
        # Normalize for cosine similarity
        faiss.normalize_L2(embeds)
        dim = embeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeds)
        faiss.write_index(index, "faiss_index.bin")
        print(f"  Saved faiss_index.bin ({os.path.getsize('faiss_index.bin')} bytes)")
        print(f"  Embedding dimensions: {dim}")
    else:
        print("  ERROR: Embedding failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUCCESS! Upload these files with your deployment:")
    print(f"  - chunks.json ({os.path.getsize('chunks.json') / 1024:.1f} KB)")
    print(f"  - faiss_index.bin ({os.path.getsize('faiss_index.bin') / 1024:.1f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
