# Midnight – AI Study Assistant

Upload notes (PDF/DOCX/TXT), then chat with them using RAG, get summaries, flashcards, quizzes and key topics. Answers come only from your document and cite pages.

## RAG flow
Upload → extract text per page (PyMuPDF / python-docx) → clean → chunk (900 chars, 150 overlap, page kept in metadata) → Gemini embeddings → ChromaDB → question embedded → top 5 chunks by cosine similarity → Gemini answers from those chunks only → answer + page sources. If the answer isn't in the context, the model returns NOT_FOUND and the app says so.

## Setup
```
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # add your GEMINI_API_KEY
uvicorn main:app --reload
```
Open http://localhost:8000

## Environment variables
GEMINI_API_KEY, CHAT_MODEL, EMBED_MODEL (see .env.example)

## Stack
React + Tailwind (single page, served by FastAPI), FastAPI, ChromaDB, SQLite (history, page text), Gemini API.

## Screenshots
Add 3–5 here before submitting.

## Not done yet (bonus)
Auth, OCR, multi-doc chat, progress tracking, PostgreSQL.
