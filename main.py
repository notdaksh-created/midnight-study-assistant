import os, io, re, json, uuid, sqlite3
from pathlib import Path
import fitz, docx, chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()
BASE = Path(__file__).parent
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

col = chromadb.PersistentClient(path=str(BASE / "chroma_db")).get_or_create_collection(
    "study", metadata={"hnsw:space": "cosine"})

def db():
    c = sqlite3.connect(BASE / "study.db"); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.executescript("""
    create table if not exists docs(id text primary key, name text, pages int);
    create table if not exists pages(doc_id text, page int, text text);
    create table if not exists messages(id integer primary key autoincrement,
        doc_id text, role text, content text, sources text);""")

app = FastAPI(title="Midnight")

# ---------- 1. extract + clean ----------
def extract(name, data):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        d = fitz.open(stream=data, filetype="pdf")
        raw = [(i + 1, p.get_text()) for i, p in enumerate(d)]
    elif ext == "docx":
        paras = [p.text for p in docx.Document(io.BytesIO(data)).paragraphs]
        raw = [(i // 40 + 1, "\n".join(paras[i:i + 40])) for i in range(0, len(paras), 40)]
    elif ext == "txt":
        t = data.decode("utf-8", "ignore")
        raw = [(i // 3000 + 1, t[i:i + 3000]) for i in range(0, len(t), 3000)]
    else:
        raise HTTPException(400, "Upload a PDF, DOCX or TXT file.")
    pages = [(n, re.sub(r"\s+", " ", t).strip()) for n, t in raw]
    return [(n, t) for n, t in pages if t]

# ---------- 2. chunk ----------
def chunk(text, size=900, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size]); i += size - overlap
    return out

# ---------- 3. embeddings ----------
def embed(texts, task):
    out = []
    for i in range(0, len(texts), 50):
        r = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50],
                                        config=types.EmbedContentConfig(task_type=task))
        out += [e.values for e in r.embeddings]
    return out

def ask(prompt, as_json=False):
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if as_json else None
    r = client.models.generate_content(model=CHAT_MODEL, contents=prompt, config=cfg)
    return json.loads(r.text) if as_json else r.text.strip()

# ---------- upload ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    pages = extract(file.filename, await file.read())
    if not pages:
        raise HTTPException(422, "No text found. Scanned PDFs need OCR (bonus feature).")
    doc_id = uuid.uuid4().hex[:12]
    ids, texts, metas = [], [], []
    for n, t in pages:
        for k, ch in enumerate(chunk(t)):
            ids.append(f"{doc_id}-{n}-{k}"); texts.append(ch); metas.append({"doc_id": doc_id, "page": n})
    col.add(ids=ids, embeddings=embed(texts, "RETRIEVAL_DOCUMENT"), documents=texts, metadatas=metas)
    with db() as c:
        c.execute("insert into docs values(?,?,?)", (doc_id, file.filename, len(pages)))
        c.executemany("insert into pages values(?,?,?)", [(doc_id, n, t) for n, t in pages])
    return {"id": doc_id, "name": file.filename, "pages": len(pages), "chunks": len(ids)}

@app.get("/api/docs")
def docs():
    with db() as c:
        return [dict(r) for r in c.execute("select * from docs order by rowid desc")]

# ---------- RAG chat ----------
class ChatReq(BaseModel):
    doc_id: str
    question: str

SYS = ("You are a study assistant. Answer ONLY using the CONTEXT, which comes from the student's own notes. "
       "Cite pages like (Page 14). If the context does not contain the answer, reply with exactly: NOT_FOUND. "
       "Explain clearly and simply.")

@app.post("/api/chat")
def chat(r: ChatReq):
    q = embed([r.question], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[q], n_results=5, where={"doc_id": r.doc_id})
    chunks, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(chunks, metas))
    with db() as c:
        hist = c.execute("select role, content from messages where doc_id=? order by id desc limit 6",
                         (r.doc_id,)).fetchall()[::-1]
    history = "\n".join(f"{h['role']}: {h['content']}" for h in hist)
    answer = ask(f"{SYS}\n\nCONTEXT:\n{context}\n\nCHAT SO FAR:\n{history}\n\nQUESTION: {r.question}")
    if "NOT_FOUND" in answer:
        answer, sources = "I couldn't find that in your document. Try rephrasing, or ask about something it covers.", []
    else:
        sources = [{"page": m["page"], "snippet": d[:160]} for d, m in zip(chunks, metas)]
        sources = list({s["page"]: s for s in sources}.values())
        sources.sort(key=lambda s: s["page"])
    with db() as c:
        c.execute("insert into messages(doc_id,role,content) values(?,?,?)", (r.doc_id, "user", r.question))
        c.execute("insert into messages(doc_id,role,content,sources) values(?,?,?,?)",
                  (r.doc_id, "assistant", answer, json.dumps(sources)))
    return {"content": answer, "sources": sources}

@app.get("/api/history/{doc_id}")
def history(doc_id: str):
    with db() as c:
        rows = c.execute("select role, content, sources from messages where doc_id=? order by id", (doc_id,))
        return [{"role": x["role"], "content": x["content"], "sources": json.loads(x["sources"] or "[]")} for x in rows]

# ---------- study tools ----------
class DocReq(BaseModel):
    doc_id: str

PROMPTS = {
    "summary": "Summarize this study material for a student: a short overview, then key points as bullets. Mention page numbers.",
    "flashcards": 'Make 10 flashcards. Return JSON: [{"q": str, "a": str, "page": int}]',
    "quiz": 'Make 8 multiple-choice questions. Return JSON: [{"question": str, "options": [4 strings], "answer_index": int, "explanation": str, "page": int}]',
    "topics": 'List the 8 most important topics. Return JSON: [{"topic": str, "why": str, "page": int}]',
}

@app.post("/api/tool/{name}")
def tool(name: str, r: DocReq):
    if name not in PROMPTS:
        raise HTTPException(404, "Unknown tool")
    with db() as c:
        rows = c.execute("select page, text from pages where doc_id=? order by page", (r.doc_id,)).fetchall()
    text = "\n".join(f"[Page {x['page']}] {x['text']}" for x in rows)[:40000]
    prompt = f"{PROMPTS[name]}\nUse only this material.\n\nSTUDY MATERIAL:\n{text}"
    return {"result": ask(prompt, as_json=name != "summary")}

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(BASE / "index.html")
