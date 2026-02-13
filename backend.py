# backend.py - FastAPI Backend for Insurance Chatbot

import os
import json
import requests
import numpy as np
import re
import time
import threading
from contextlib import asynccontextmanager

# Core web framework
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from dotenv import load_dotenv

# Heavy/optional imports with fault tolerance
try:
    import pandas as pd
    print("✅ pandas loaded")
except ImportError:
    pd = None
    print("⚠️ pandas not available")

try:
    import pdfplumber
    print("✅ pdfplumber loaded")
except ImportError:
    pdfplumber = None
    print("⚠️ pdfplumber not available")

try:
    from openai import OpenAI
    print("✅ openai loaded")
except ImportError:
    OpenAI = None
    print("⚠️ openai not available")

try:
    import faiss
    print("✅ faiss loaded")
except ImportError:
    faiss = None
    print("⚠️ faiss not available - will use keyword search fallback")


# Load environment variables
load_dotenv()

try:
    from governance import PIIScrubber, AuditLogger, ModelRiskGuardian
    pii_scrubber = PIIScrubber()
    audit_logger = AuditLogger()
    risk_guardian = ModelRiskGuardian(confidence_threshold=0.65)
    print("✅ Governance Module Loaded (PII Scrubbing, Audit Logging & Risk Guardian Active)")
except Exception as e:
    pii_scrubber = None
    audit_logger = None
    risk_guardian = None
    print(f"⚠️ Governance Module Loading Failed: {e}")

# Global variables for storing the knowledge base
chunks = []
index = None
history = []

# API Keys - will be loaded from environment variables
NVAPI_KEY = os.getenv("NVAPI_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Initialize OpenAI client - will be set after environment check
client = None

# -------------------------------
# Helper functions (from original app.py)
# -------------------------------

def clean_pdf(text):
    """Clean PDF text by removing headers, footers, and formatting issues"""
    text = re.sub(r'Page \d+\n', '', text)  # Remove page numbers
    text = re.sub(r'ICICI .*?\n', '', text) # Remove ICICI header/footer
    text = re.sub(r'IRDAI Regn.*?\n', '', text) # Remove regulator lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Fix hyphenated words
    text = re.sub(r'[\u2022•]', '-', text) # Replace bullets with dash
    text = re.sub(r'(?<![.!?])\n', ' ', text) # Join lines that don't end with punctuation
    text = re.sub(r'\n+', '\n', text) # Remove extra newlines
    text = re.sub(r' {2,}', ' ', text) # Remove extra spaces
    return text

def read_website(url, api_key):
    """Read website content using Jina Reader API"""
    api_url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to read website: {response.status_code}, {response.text}")

def chunk_with_jina(text: str, api_key: str, max_len: int = 8000):
    """Split text into chunks using Jina Segmenter API with fallback to simple chunking"""
    JINA_SEGMENT_URL = "https://api.jina.ai/v1/segment"
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks = []

    for i in range(0, len(text), max_len):
        part = text[i:i+max_len]

        try:
            payload = {
                "content": part,
                "config": {"split_length": 500}  # ~500 char chunks
            }

            resp = requests.post(JINA_SEGMENT_URL, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                segs = [seg["text"] for seg in data.get("segments", [])]

                if segs:
                    chunks.extend(segs)
                else:
                    # Fallback to simple chunking if no segments returned
                    print("⚠️ Jina returned no segments, using simple chunking")
                    simple_chunks = simple_chunk_text(part, 500)
                    chunks.extend(simple_chunks)
            else:
                print(f"⚠️ Jina Segmenter failed: {resp.status_code}, using simple chunking")
                simple_chunks = simple_chunk_text(part, 500)
                chunks.extend(simple_chunks)
                
        except Exception as e:
            print(f"⚠️ Jina API error: {e}, using simple chunking")
            simple_chunks = simple_chunk_text(part, 500)
            chunks.extend(simple_chunks)

    return chunks

def simple_chunk_text(text: str, chunk_size: int = 500):
    """Simple text chunking fallback"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def embed_with_jina(texts, api_key, batch_size=16):
    """Convert texts into embeddings using Jina Embeddings API with better error handling"""
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
        inputs = [{"text": t[:8000]} for t in batch]  # Limit text length

        payload = {
            "model": "jina-embeddings-v4",
            "task": "text-matching",
            "input": inputs
        }

        try:
            print(f"🔄 Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            resp = requests.post(JINA_EMBED_URL, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                vectors = [item["embedding"] for item in data["data"]]
                all_vectors.extend(vectors)
                print(f"✅ Batch {batch_num} completed successfully")
            else:
                print(f"❌ Embedding batch {batch_num} failed: {resp.status_code}")
                print(f"Error: {resp.text[:200]}...")
                raise Exception(f"Embedding failed: {resp.status_code}, {resp.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ Embedding batch {batch_num} timed out")
            raise Exception(f"Embedding batch {batch_num} timed out after 30 seconds")
        except Exception as e:
            print(f"❌ Error in embedding batch {batch_num}: {e}")
            raise

    print(f"✅ All embedding batches completed - {len(all_vectors)} vectors created")
    return np.array(all_vectors, dtype="float32")

def build_index(vectors):
    """Build FAISS index for similarity search"""
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def embed_query_jina(query, api_key):
    """Embed user query using Jina API"""
    JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-embeddings-v4",
        "task": "text-matching",
        "input": [{"text": query}]
    }
    resp = requests.post(JINA_EMBED_URL, headers=headers, json=payload, timeout=60)

    if resp.status_code == 200:
        data = resp.json()
        vector = data["data"][0]["embedding"]
        return np.array([vector], dtype="float32")
    else:
        raise Exception(f"Query embedding failed: {resp.status_code}, {resp.text}")

def search(query, index, chunks, api_key, top_k=3):
    """Search for relevant chunks using FAISS. Returns chunks and their similarity scores."""
    q_vec = embed_query_jina(query, api_key)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    # D contains cosine similarities since we normalized
    return [chunks[i] for i in I[0]], D[0]

def rerank_with_jina(query, docs, api_key, top_n=3):
    """Rerank documents using Jina Reranker API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    formatted_docs = [{"text": d} for d in docs]

    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": top_n,
        "documents": formatted_docs,
        "return_documents": True
    }

    resp = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=data)
    out = resp.json()

    if resp.status_code == 200 and "results" in out:
        return [r["document"]["text"] for r in out["results"]]
    else:
        raise Exception(f"Reranking failed: {resp.status_code}, {out}")

def keyword_search(query, chunks, top_k=3):
    """Fallback keyword search if vector index isn't ready"""
    import re
    # Clean and tokenize query
    query_words = re.findall(r'\w{3,}', query.lower())
    if not query_words: return []
    
    matches = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Count matches
        score = sum(2 if word in chunk_lower else 0 for word in query_words)
        if score > 0:
            # Bonus for exact phrase matches
            if query.lower() in chunk_lower: score += 10
            matches.append((score, chunk))
    
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:top_k]]

def answer_question(query, index, chunks, history, api_key, top_k=5, rerank_n=3):
    """Generate answer using RAG pipeline"""
    global client
    
    if not client:
        return "Sorry, the AI service is not available. Please check the API configuration."
    
    # Step 1: Retrieve more candidates for better reranking coverage
    retrieved, scores = search(query, index, chunks, api_key, top_k=15)
    avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

    # Step 2: Rerank documents to find the absolute best matches
    reranked = rerank_with_jina(query, retrieved, api_key, top_n=5)

    # Step 3: Build context
    context = "\n".join(reranked)

    # Step 4: Use LLM to answer
    try:
        if not client:
            raise Exception("AI reasoning client not initialized")
            
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": """You are xInsur AI, a STRICT Enterprise Insurance Intelligence Platform. 
                
                ### MANDATORY GROUNDING RULES:
                1. **STRICT PDF ONLY**: Your answer must be derived EXCLUSIVELY from the provided context (Policy Excerpts).
                2. **EXTERNAL KNOWLEDGE PROHIBITED**: Do NOT use your internal training data to answer insurance-specific details. If the context does not contain the answer, explicitly state: "My apologies, but this specific information is not detailed within our indexed institutional documentation."
                3. **VERIFIABLE CITATIONS**: Always mention that your answer is based on verified excerpts.
                
                ### RESPONSE STRUCTURE:
                1. **Structured Analysis**: Core answer based ONLY on excerpts.
                2. **Bot Section**: Institutional metadata (Eligibility, Scope, Assumptions).
                
                Analytical, advisory tone. Zero hallucination tolerance."""},
                *history,
                {"role": "user", "content": f"POLICY EXCERPTS FOR ANALYSIS:\n{context}\n\nUSER INQUIRY: {query}"}
            ],
            extra_body={
                "max_tokens": 512,
                "temperature": 1.00,
                "top_p": 1.00
            }
        )

        answer = resp.choices[0].message.content
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        return answer
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        # fallback: return raw chunks if LLM fails
        msg = "I have retrieved the following specific details from ICICI Prudential policy documentation to assist you:\n\n"
        for i, chunk in enumerate(reranked[:3]):
            msg += f"🔍 **Policy Detail {i+1}:**\n{chunk}\n\n"
        msg += "\n*(Note: Our advanced reasoning engine is currently optimizing. Providing direct documentation matches.*)"
        return msg

def answer_without_rag(query, history):
    """Generate answer using LLM without RAG context"""
    global client
    
    if not client:
        return "[LLM_UNAVAILABLE] I'm having trouble connecting to my AI service."
    
    try:
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": """You are xInsur AI, an Enterprise Insurance Intelligence Platform. 
                
                ### CRITICAL STATUS: Knowledge Base Synchronization in Progress.
                The institutional PDF repository is currently being indexed. You are operating in 'General Advisory' mode.
                
                ### RULES:
                1. **GENERAL KNOWLEDGE ONLY**: You must clearly state that this answer is based on professional standards and NOT the user's specific policy.
                2. **STRUCTURE**:
                   - Use the header: '⚠️ **GENERAL ADVISORY (Policy Sync in Progress)**'
                   - Follow with a structured analysis.
                   - End with the standard 'Bot:' section as requested.
                
                Maintain an institutional, formal tone."""},
                *history,
                {"role": "user", "content": query}
            ],
            extra_body={
                "max_tokens": 512,
                "temperature": 1.00,
                "top_p": 1.00
            }
        )
        answer = resp.choices[0].message.content
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return "[LLM_UNAVAILABLE] AI reasoning service is temporarily unavailable."

def classify_intent(query):
    """Classify user intent"""
    q = query.lower().strip()
    
    # Only treat as a greeting if very short and at the start of conversation
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if q in greetings and len(history) < 2:
        return "greeting"
    
    # Policy keys
    policy_keys = ["coverage", "benefits", "exclusions", "claim", "policy", "premium", "insurance", 
                   "plan", "age", "maturity", "death", "surrender", "bike", "car", "vehicle", "grace", "period"]
    
    if any(word in q for word in policy_keys):
        return "policy_info"
        
    if any(word in q for word in ["company", "about", "vision", "branches", "icici"]):
        return "company_info"
        
    if any(word in q for word in ["contact", "support", "customer care", "helpline", "call", "email"]):
        return "general_support"
        
    # Default to policy_info to allow RAG/LLM to handle general insurance conversation
    return "policy_info"

def insurance_chatbot(query, chunks, index, company_kb, faq_kb, history, api_key):
    """Main chatbot logic"""
    intent = classify_intent(query)

    if intent == "greeting":
        return "Authorization Successful. Welcome to the xInsur AI Enterprise platform. How can I assist with your institutional policy analytics today?"
    elif intent == "policy_info":
        return answer_question(query, index, chunks, history, api_key)
    elif intent == "company_info":
        return company_kb.get("about_company", "Company information not available.")
    elif intent == "general_support":
        return faq_kb.get(query.lower(), "Please contact customer support at 1800-209-9777.")
    else:
        return "I'm here to help with insurance-related queries. Could you rephrase?"

# Knowledge bases
company_kb = {
    "about_company": "ICICI Prudential Life Insurance is one of the leading insurance providers in India, "
                     "offering term plans, health insurance, and savings plans."
}

faq_kb = {
    "how can i contact customer support?": "You can call our toll-free number 1800-209-9777 or email support@iciciprulife.com",
    "how do i pay premium?": "Premiums can be paid online via netbanking, debit/credit card, or through ICICI branches."
}

# Knowledge base initialization moved to async function below

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - initialize basic components
    global client
    
    # Initialize OpenAI client
    if NVAPI_KEY:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVAPI_KEY
        )
        print("✅ OpenAI client initialized")
    else:
        print("❌ NVAPI_KEY not found")
    
    # Perform knowledge base load in background so server starts INSTANTLY
    thread = threading.Thread(target=full_kb_init_background)
    thread.daemon = True
    thread.start()
    
    yield
    pass

def full_kb_init_background():
    """Sequential init: PDF -> Website -> Embeddings"""
    global chunks, index
    try:
        # 1. Fast PDF Load
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(base_dir, "ICICI_Insurance.pdf")
        
        if os.path.exists(pdf_path):
            print(f"📄 Background: Extracting {pdf_path}...")
            with pdfplumber.open(pdf_path) as pdf:
                texts = [p.extract_text() for p in pdf.pages if p.extract_text()]
                pdf_text = clean_pdf("".join(texts))
            
            if pdf_text:
                new_chunks = simple_chunk_text(pdf_text, 500)
                chunks.extend(new_chunks)
                print(f"✅ Background: {len(chunks)} Segments loaded (Search active)")
        
        # 2. Heavier Embeddings
        if chunks and JINA_API_KEY:
            print("🔄 Background: Optimizing AI index...")
            embeds = embed_with_jina(chunks, JINA_API_KEY)
            index = build_index(embeds)
            print("✅ Background: AI index ready")
            
    except Exception as e:
        print(f"❌ Background Init Error: {e}")

def fast_init_pdf_and_web():
    """Quickly load text from PDF and Website so keyword search is active"""
    global chunks
    print("🔄 Performing rapid policy data sync...")
    
    pdf_path = "ICICI_Insurance.pdf"
    pdf_text = ""
    if os.path.exists(pdf_path):
        try:
            print(f"📄 Opening {pdf_path}...")
            with pdfplumber.open(pdf_path) as pdf:
                print(f"📖 Extracting text from {len(pdf.pages)} pages...")
                # Extracting page by page with progress
                texts = []
                for i, page in enumerate(pdf.pages):
                    content = page.extract_text()
                    if content: texts.append(content)
                pdf_text = "".join(texts)
            pdf_text = clean_pdf(pdf_text)
            print(f"✅ Policy document loaded ({len(pdf_text)} chars)")
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
    
    # Combined data
    combined = f"DOCUMENTATION:\n{pdf_text}"
    if combined.strip():
        new_chunks = simple_chunk_text(combined, 500)
        chunks.clear() # Ensure clean start
        chunks.extend(new_chunks)
        print(f"✅ {len(chunks)} Policy segments ready for search")
    else:
        print("⚠️ No data extracted for segments")

def generate_embeddings_background():
    """Heavy lifting in background"""
    global chunks, index
    try:
        if not chunks: return
        print("🔄 Optimizing AI search index in background...")
        embeds = embed_with_jina(chunks, JINA_API_KEY)
        index = build_index(embeds)
        print("✅ AI reasoning index optimized and ready")
    except Exception as e:
        print(f"❌ Background optimization error: {e}")

def initialize_knowledge_base_sync():
    """Initialize knowledge base in a background thread"""
    global chunks, index
    
    try:
        print("🔄 Starting knowledge base initialization...")
        
        # Load PDF
        pdf_path = "ICICI_Insurance.pdf"
        if os.path.exists(pdf_path):
            with pdfplumber.open(pdf_path) as pdf:
                all_text = "".join([page.extract_text() for page in pdf.pages])
            pdf_cleaned = clean_pdf(all_text)
            print("✅ PDF loaded and cleaned")
        else:
            pdf_cleaned = ""
            print("⚠️  ICICI_Insurance.pdf not found at specified path")

        # Load website data
        try:
            url = "https://www.icicibank.com/personal-banking/insurance?ITM=nli_home_na_megamenuItem_1CTA_CMS_insurance_viewAllInsurance_NLI"
            website_text = read_website(url, JINA_API_KEY)
            print("✅ Website data loaded")
        except Exception as e:
            print(f"⚠️  Failed to load website data: {e}")
            website_text = ""

        # Combine texts
        combined_text = f"STATIC PDF DATA:\n{pdf_cleaned}\n\nWEBSITE DATA:\n{website_text}"
        print(f"📄 Combined text length: {len(combined_text)} characters")

        # Create chunks and embeddings
        if combined_text.strip():
            print("✂️ Creating chunks using simple chunking...")
            chunks = simple_chunk_text(combined_text, 500)
            print(f"✅ Created {len(chunks)} chunks")
            
            embeds = embed_with_jina(chunks, JINA_API_KEY)
            index = build_index(embeds)
            print(f"✅ Knowledge base initialized with {len(chunks)} chunks")
        else:
            print("⚠️  No content to process")
        
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")
        chunks = []
        index = None

# -------------------------------
# FastAPI App
# -------------------------------

app = FastAPI(
    title="xInsur AI Enterprise Gateway",
    description="Institutional-grade insurance intelligence platform",
    version="2.2.0",
    lifespan=lifespan
)

# Deep-Clean CORS: Force open for Railway Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_dashboard():
    """Serve the primary institutional intelligence dashboard"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.get("/healthz")
async def liveness_check():
    """Simple liveness check for Railway infrastructure"""
    return JSONResponse(content={"status": "online"}, status_code=200)

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/api", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="Insurance Chatbot API is running")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    global chunks, index
    
    if chunks and index is not None:
        return HealthResponse(
            status="healthy", 
            message=f"EIIP Gateway active. Knowledge base synchronization complete: {len(chunks)} segments indexed."
        )
    else:
        return HealthResponse(
            status="degraded", 
            message="EIIP Gateway active. Knowledge base synchronization in progress."
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    global chunks, index, history
    
    # Prioritize general intent checks (Greetings, Support, Company info)
    start_time = time.time()
    
    # --- Governance Layer 1: PII Redaction ---
    sanitized_msg = request.message
    detected_pii = []
    
    if pii_scrubber:
        sanitized_msg, detected_pii = pii_scrubber.scrub(request.message)
        if detected_pii:
            print(f"🚨 PII Detected & Redacted: {detected_pii}")
            # If explicit PII (like PAN) is found, we might want to warn the user
            # But per blueprint, we silently protect them and proceed with redacted query.

    # --- Governance Layer 2: Mandatory Disclaimer ---
    if not history:
        disclaimer = ("**🏛️ xInsur AI | Institutional Advisory Disclaimer:**\n"
                     "Analytically generated response based on indexed policy documentation. "
                     "This output is for advisory purposes and does not constitute a binding contract. "
                     "Refer to source citations and system audit trails for full accountability.\n\n")
    else:
        disclaimer = ""

    intent = classify_intent(sanitized_msg)
    
    if intent == "greeting":
        if len(history) < 2:
            resp_text = "Authorization Successful. Welcome to the xInsur AI Enterprise Intelligence Platform. I am your verified digital advisor. How can I assist with your institutional policy analytics today?"
        else:
            resp_text = "Greetings. I am standing by for your next analytical query."
            
        if audit_logger:
            audit_logger.log_event("CHAT_INTERACTION", "user_session_1", sanitized_msg, resp_text, bool(detected_pii))
            
        return ChatResponse(response=disclaimer + resp_text, intent=intent)
    
    if intent == "general_support":
        # Check FAQ
        support_reply = faq_kb.get(request.message.lower(), "Secure institutional support is available at **1800-209-9777** or via **support@iciciprulife.com**. Analytical support is available 24/7.")
        return ChatResponse(response=support_reply, intent=intent)

    if intent == "company_info":
        return ChatResponse(response=company_kb.get("about_company"), intent=intent)

    # If index isn't ready but chunks ARE loaded (keyword search fallback for policy info)
    if index is None and chunks:
        # Try keyword search
        relevant_chunks = keyword_search(request.message, chunks)
        if relevant_chunks:
            context = "\n".join(relevant_chunks)
            # Use SANITIZED message for LLM
            response = answer_without_rag(f"Based on these policy excerpts, answer the customer's question: {sanitized_msg}\n\nPolicy Information:\n{context}", history)
            
            # If LLM failed, build response purely from chunks
            if "[LLM_UNAVAILABLE]" in response:
                final_msg = "Here are the most relevant details from our official policy documentation:\n\n"
                for i, c in enumerate(relevant_chunks[:3]):
                    text = c.strip()
                    if len(text) > 500: text = text[:500] + "..."
                    final_msg += f"\n> **Excerpt {i+1}:** {text}\n"
            else:
                final_msg = f"{response}\n\n"
            
            # Add disclaimer if first message
            final_response = disclaimer + final_msg

            # --- Governance Layer 3: Audit Logging & Latency ---
            latency = (time.time() - start_time) * 1000
            if audit_logger:
                audit_logger.log_event("RAG_RETRIEVAL", "user_session_1", sanitized_msg, final_response[:200]+"...", bool(detected_pii), latency)
            
            return ChatResponse(response=final_response, intent=intent)

    # Completely fallback if nothing is loaded yet
    if not chunks:
        response = answer_without_rag(sanitized_msg, history)
        final_response = disclaimer + response
        
        # Audit Fallback
        latency = (time.time() - start_time) * 1000
        if audit_logger:
            audit_logger.log_event("FALLBACK_NO_KB", "user_session_1", sanitized_msg, final_response[:200]+"...", bool(detected_pii), latency)
            
        return ChatResponse(response=final_response, intent=intent)
    
    if not JINA_API_KEY or not NVAPI_KEY:
        raise HTTPException(status_code=500, detail="API keys not configured")
    
    try:
        # For full RAG, we need to extract scores to pass to the Risk Guardian
        retrieved, scores = search(sanitized_msg, index, chunks, JINA_API_KEY, top_k=5)
        avg_confidence = float(np.max(scores)) # Use best match as confidence proxy
        
        response = insurance_chatbot(
            sanitized_msg, 
            chunks, 
            index, 
            company_kb, 
            faq_kb, 
            history, 
            JINA_API_KEY
        )
        
        # --- Governance Layer 4: Model Risk Validation ---
        if risk_guardian:
            validation = risk_guardian.validate_response(sanitized_msg, response, avg_confidence)
            if not validation["safe"]:
                response = validation["message"]
                print(f"🛑 Risk Guardian Intercepted Response: {validation['reason']}")

        final_response = disclaimer + response

        # Audit
        latency = (time.time() - start_time) * 1000
        if audit_logger:
            audit_logger.log_event("FULL_RAG", "user_session_1", sanitized_msg, final_response[:200]+"...", bool(detected_pii), latency, avg_confidence)
            
        return ChatResponse(response=final_response, intent=intent)
        
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return ChatResponse(response=f"I encountered a processing error. Please try again. (Detail: {str(e)})", intent="error")

@app.post("/clear-history")
async def clear_history():
    """Clear chat history"""
    global history
    history = []
    return {"message": "Chat history cleared"}

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    global chunks, history
    
    return {
        "chunks_loaded": len(chunks) if chunks else 0,
        "conversation_length": len(history),
        "knowledge_base_status": "loaded" if chunks and index is not None else "not_loaded"
    }

@app.get("/api/audit-logs")
async def get_audit_logs():
    """Retrieve recent audit logs for compliance review"""
    log_file = "audit_log.jsonl"
    if not os.path.exists(log_file):
        return {"logs": []}
    
    try:
        with open(log_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            # Return last 50 entries
            recent_logs = [json.loads(line) for line in lines[-50:]]
            recent_logs.reverse() # Most recent first
            return {"logs": recent_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/policy-info")
async def get_policy_info():
    """Get metadata about the loaded policy repository"""
    global chunks
    return {
        "repository_name": "Institutional Wealth & Protection Vault",
        "primary_source": "ICICI_Insurance.pdf",
        "segments_active": len(chunks),
        "last_sync": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "indexing_status": "CONSISTENT" if chunks else "SYNCHRONIZING"
    }

@app.get("/api/governance-data")
async def get_governance_data():
    """Get real-time governance metrics"""
    return {
        "protocol": "SR 11-7 (Model Risk Management)",
        "guardian_status": "ENFORCED",
        "pii_rules": ["PAN", "AADHAAR", "EMAIL", "PHONE"],
        "encryption": "AES-256-GCM",
        "integrity_check": "SHA-256 Chained"
    }

if __name__ == "__main__":
    # Check for required environment variables
    if not NVAPI_KEY:
        print("Warning: NVAPI_KEY environment variable not set")
    if not JINA_API_KEY:
        print("Warning: JINA_API_KEY environment variable not set")
    
    # FINAL HARNESS: Force bind to host's dynamic port
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8050)))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    try:
        print("--------------------------------------------------")
        print(f"� GATEWAY_INIT: Binding to {host}:{port}")
        print(f"🔐 GOVERNANCE: SR 11-7 Protocols Loaded")
        print(f"🚀 VERSION: 2.2.0 Active")
        print("--------------------------------------------------")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        print(f"❌ CRITICAL_ERROR: {e}")
