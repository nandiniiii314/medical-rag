# 🏥 Hybrid Medical Research Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to upload medical or IEEE research papers in PDF format, ask page-specific or paper-specific questions, and seamlessly falls back to web search using Tinyfish when the uploaded document lacks the required information.

## Features

- Upload medical or IEEE research papers (PDF)
- Ask page-wise questions
  - Explain page 1
  - Summarize page 5
  - What is on page 3?
- Ask paper-specific questions
  - What disease is discussed?
  - What methodology is used?
  - What are the results?
- Intelligent RAG pipeline using ChromaDB
- Tinyfish web search fallback for broader medical questions
- Automatic page detection
- Page-aware responses with citations
- Streamlit-based interactive interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq LLM
- ChromaDB
- HuggingFace Embeddings
- Tinyfish Search API
- PyMuPDF

---

## Project Structure

```
medical-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── medical_hybrid_db/
└── assets/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/nandiniiii314/medical-rag.git
cd medical-rag
```

Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Groq API Key

Replace

```python
api_key="your_groq_api_key_here"
```

with your Groq API key.

---

### Tinyfish API Key

Replace

```python
headers = {
    "X-API-Key": "your_tinyfish_api_key_here"
}
```

with your Tinyfish API key.

---

## Run the Application

```bash
streamlit run app.py
```

---

## Example Questions

### Page-wise

- Explain page 1
- Summarize page 4
- What is on page 2?

### Paper Questions

- What disease is discussed?
- Explain the methodology.
- What are the key findings?
- Summarize the paper.

### General Medical Questions

- What is pneumonia?
- Recent treatment for Alzheimer's disease.
- Compare this paper with current medical research.

---

## How It Works

1. Upload a PDF research paper.
2. The PDF is divided into individual pages.
3. Each page is split into chunks.
4. Chunks are embedded using HuggingFace embeddings.
5. ChromaDB stores the embeddings.
6. LangChain Agent determines which tool to use.
7. Responses are generated using Groq LLM.
8. If information is unavailable in the PDF, Tinyfish performs a web search.

---

## Architecture

```
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Page Extraction
 │
 ▼
Chunking
 │
 ▼
HuggingFace Embeddings
 │
 ▼
ChromaDB Vector Store
 │
 ▼
LangChain Agent
 ├── explain_page()
 ├── search_pdf()
 └── web_search()
 │
 ▼
Groq LLM
 │
 ▼
Final Answer
```

---

## Dependencies

- Streamlit
- LangChain
- LangChain Community
- LangChain Groq
- LangChain HuggingFace
- ChromaDB
- Sentence Transformers
- PyMuPDF
- Requests

---

## Future Enhancements

- Multi-PDF support
- Chat history
- PDF annotations
- Citation generation
- Medical image analysis
- Voice-based querying
- Research paper comparison
- Persistent vector database

---

## Author

**Nnadini**

GitHub: https://github.com/nandiniiii314

---

## License

This project is intended for educational and research purposes.
