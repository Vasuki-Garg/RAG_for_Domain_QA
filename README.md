# PDF Chat (Domain-Specific QA) — Streamlit + LangChain

A Streamlit app that lets you upload PDFs, indexes them into an in-memory vector store, and answers questions using retrieval + an OpenAI chat model. Includes basic prompt-injection guardrails and shows source pages for transparency.

## Features
- Upload multiple PDFs
- Chunking + embeddings + similarity search
- Retrieval-grounded answers (3-sentence max)
- Source page references + “Retrieved chunks” transparency panel
- Basic prompt-injection filtering

---

## 1) Setup (Local)

### Prerequisites
- Python 3.9+ recommended
- An OpenAI API key

### Install
```bash
pip install -r requirements.txt
