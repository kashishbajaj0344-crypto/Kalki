#!/usr/bin/env python3
"""
Kalki Mini Jarvis - Tailored Version
Python 3.10+
Features:
- GUI drag-and-drop PDF ingestion
- PDF chunking and embeddings via OpenAI
- Chroma vector DB with auto-recreation if schema mismatches
- RAG query answering
- Playful chat
- Logging and query tracking
"""

import os
import sys
import time
import json
import logging
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from pathlib import Path
from typing import List, Dict, Optional

# --- External libraries ---
try:
    from openai import OpenAI
except:
    print("Please install openai: pip install openai")
    raise
try:
    import chromadb
except:
    print("Please install chromadb: pip install chromadb")
    raise
try:
    import pdfplumber
except:
    print("Please install pdfplumber: pip install pdfplumber")
    raise

# --- LangChain Community imports ---
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.chat_models import ChatOpenAI
except:
    print("Please install langchain-community: pip install -U langchain-community")
    raise

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path.home() / "Desktop" / "Kalki"
PDF_DIR = BASE_DIR / "pdfs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
RESOURCES_JSON = BASE_DIR / "kalki_resources.json"
QUERY_LOG = BASE_DIR / "query_cost.json"
LOG_FILE = BASE_DIR / "kalki_gui.log"

DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_CHAT_MODEL = "gpt-4o"
EMBED_CHUNK_WORDS = 100
EMBED_OVERLAP_WORDS = 20
TOP_K = 5
MAX_CONTEXT_CHARS = 30000
RETRY_ATTEMPTS = 2

for p in [BASE_DIR, PDF_DIR, VECTOR_DB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# --- OpenAI + Chroma setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable.")

client_openai = OpenAI()
try:
    client_chroma = chromadb.Client()
    collection = client_chroma.get_or_create_collection("kalki_collection")
except Exception as e:
    logging.warning(f"Chroma init failed, recreating DB: {e}")
    import shutil
    shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    client_chroma = chromadb.Client()
    collection = client_chroma.get_or_create_collection("kalki_collection")

if not QUERY_LOG.exists():
    with open(QUERY_LOG, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

# -------------------------
# Utilities
# -------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"load_json error {path}: {e}")
        return default

def save_json(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"save_json error {path}: {e}")

def log_query(entry: Dict):
    logs = load_json(QUERY_LOG, [])
    logs.append(entry)
    save_json(QUERY_LOG, logs)

# -------------------------
# PDF ingestion
# -------------------------
def extract_text_from_pdf(path: Path) -> str:
    parts: List[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
        return "\n".join(parts)
    except Exception as e:
        logging.error(f"extract_text_from_pdf failed {path}: {e}")
        return ""

def chunk_text(text: str, chunk_words: int = EMBED_CHUNK_WORDS, overlap: int = EMBED_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_words]))
        i += chunk_words - overlap
    return chunks

def ingest_pdf_file(pdf_path: Path, domain: str = "general", title: Optional[str] = None, author: Optional[str] = None):
    title = title or pdf_path.stem
    author = author or ""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.warning(f"No text extracted from {pdf_path}")
        return False
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks, start=1):
        try:
            emb_resp = client_openai.embeddings.create(model=DEFAULT_EMBED_MODEL, input=chunk)
            emb = emb_resp.data[0].embedding
            meta = {
                "title": title,
                "author": author,
                "source": str(pdf_path),
                "domain": domain,
                "chunk_id": f"{pdf_path.name}chunk{idx}",
                "ingested_at": now_ts()
            }
            collection.add(documents=[chunk], embeddings=[emb], metadatas=[meta], ids=[meta["chunk_id"]])
        except Exception as e:
            logging.error(f"Ingest failed {pdf_path} chunk {idx}: {e}")
            continue
    logging.info(f"Ingested {pdf_path} ({len(chunks)} chunks)")
    return True

# -------------------------
# RAG
# -------------------------
def retrieve_chunks(query: str, domain_priority: Optional[List[str]] = None, top_k: int = TOP_K):
    results = []
    domains = domain_priority or []
    if not domains:
        try:
            meta_info = collection.get(include=["metadatas"])
            metas = meta_info.get("metadatas", [])
            domains = list({m.get("domain") for m in (metas[0] if metas else []) if m.get("domain")})
        except:
            domains = ["general"]
    for domain in domains:
        try:
            res = collection.query(query_texts=[query], n_results=top_k, where={"domain": domain})
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            results.extend({"chunk": d, "metadata": m, "score": float(dist) if dist is not None else None} 
                           for d, m, dist in zip(docs, metas, dists))
        except Exception as e:
            logging.warning(f"retrieve_chunks domain {domain} failed: {e}")
            continue
    results = [r for r in results if r.get("chunk")]
    results.sort(key=lambda x: (x["score"] is None, x["score"]))
    return results

def assemble_context(chunks: List[Dict], max_chars: int = MAX_CONTEXT_CHARS):
    pieces, used_ids = [], []
    total = 0
    for c in chunks:
        meta = c.get("metadata", {})
        citation = " | ".join(filter(None, [meta.get("title"), meta.get("author"), meta.get("source")])) or meta.get("chunk_id","")
        piece = f"[{meta.get('domain','general')}] {citation}\n{c.get('chunk')}\n"
        if total + len(piece) > max_chars:
            break
        pieces.append(piece)
        used_ids.append(meta.get("chunk_id") or citation)
        total += len(piece)
    return ("\n---\n".join(pieces) if pieces else "No context retrieved."), used_ids

def call_chat_model(query: str, context: str, model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.5, max_tokens: int = 1000):
    last_err = None
    used_model = model
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            system = {"role":"system","content":"You are Kalki, a playful helpful assistant."}
            user = {"role":"user","content":f"Context:\n{context}\n\nQuery:\n{query}"}
            resp = client_openai.chat.completions.create(model=used_model, messages=[system,user], max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content if resp.choices else resp["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            logging.warning(f"chat model call failed attempt {attempt} model {used_model}: {e}")
            if used_model == DEFAULT_CHAT_MODEL:
                used_model = "gpt-4o"
            time.sleep(1 + attempt*2)
    logging.error(f"LLM failed: {last_err}")
    return f"Error calling model: {last_err}"

def ask_kalki(query: str, domain_priority: Optional[List[str]] = None, top_k: int = TOP_K):
    start = time.time()
    chunks = retrieve_chunks(query, domain_priority=domain_priority, top_k=top_k)
    context, used_ids = assemble_context(chunks)
    answer = call_chat_model(query, context)
    elapsed = time.time() - start
    log_entry = {"timestamp": now_ts(), "query": query, "chunks_used": used_ids, "answer": answer[:1000], "elapsed_s": elapsed}
    log_query(log_entry)
    return answer

# -------------------------
# GUI
# -------------------------
class KalkiGUI:
    def __init__(self, master):
        self.master = master
        master.title("Kalki — Jarvis")
        master.geometry("900x700")

        self.left = tk.Frame(master, width=300)
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.right = tk.Frame(master)
        self.right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        tk.Label(self.left, text="PDFs (drag files)").pack()
        self.pdf_listbox = tk.Listbox(self.left, width=40, height=25)
        self.pdf_listbox.pack(padx=5, pady=5)
        tk.Button(self.left, text="Add PDFs", command=self.add_pdfs_dialog).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(self.left, text="Ingest Selected", command=self.ingest_selected).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(self.left, text="Refresh List", command=self.refresh_pdf_list).pack(fill=tk.X, padx=5, pady=2)

        tk.Label(self.right, text="Kalki Chat").pack()
        self.chat_display = scrolledtext.ScrolledText(self.right, wrap=tk.WORD, height=25)
        self.chat_display.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.user_entry = tk.Entry(self.right)
        self.user_entry.pack(fill=tk.X, padx=5, pady=5)
        self.user_entry.bind("<Return>", self.on_send)
        tk.Button(self.right, text="Send", command=self.on_send).pack(padx=5, pady=2)

        self.refresh_pdf_list()
        self.append_chat("Kalki", "Hello — I'm Kalki. Ready to help. 😊")

    def append_chat(self, who: str, text: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{who} [{time.strftime('%H:%M:%S')}]: {text}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def refresh_pdf_list(self):
        self.pdf_listbox.delete(0, tk.END)
        resources = load_json(RESOURCES_JSON, [])
        for idx, r in enumerate(resources):
            name = r.get("title") or Path(r.get("local_path","")).name
            domain = r.get("domain","general")
            self.pdf_listbox.insert(tk.END, f"{idx+1}. {name} [{domain}]")

    def add_pdfs_dialog(self):
        files = filedialog.askopenfilenames(title="Select PDFs", filetypes=[("PDF Files","*.pdf")])
        resources = load_json(RESOURCES_JSON, [])
        for f in files:
            resources.append({"title": Path(f).stem, "author": "", "type":"pdf", "domain":"general", "local_path": str(f)})
        save_json(RESOURCES_JSON, resources)
        self.refresh_pdf_list()
        messagebox.showinfo("Kalki", "PDFs added. Run 'Ingest Selected' to embed.")

    def ingest_selected(self):
        sel = self.pdf_listbox.curselection()
        if not sel:
            messagebox.showinfo("Kalki", "Select PDFs first.")
            return
        resources = load_json(RESOURCES_JSON, [])
        for i in sel:
            entry = resources[i]
            local_path = entry.get("local_path")
            if local_path:
                p = Path(local_path)
                target_dir = PDF_DIR / entry.get("domain","general")
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / p.name
                if not target.exists():
                    import shutil
                    shutil.copy(p, target)
                ingest_pdf_file(target, domain=entry.get("domain","general"), title=entry.get("title"), author=entry.get("author"))
        messagebox.showinfo("Kalki", "Ingestion complete.")

    def on_send(self, event=None):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return
        self.append_chat("You", user_text)
        self.user_entry.delete(0, tk.END)
        threading.Thread(target=self._handle_user_query, args=(user_text,), daemon=True).start()

    def _handle_user_query(self, text: str):
        try:
            answer = ask_kalki(text)
            answer += "\n\n(P.S. I can make jokes too!)"
            self.append_chat("Kalki", answer)
        except Exception as e:
            logging.error(f"_handle_user_query error: {e}")
            self.append_chat("Kalki", f"Error: {e}")

# -------------------------
# Main
# -------------------------
def main():
    root = tk.Tk()
    app = KalkiGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

