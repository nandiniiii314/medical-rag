
# 🏥 Hybrid Medical Research Assistant
# Use case:
# Upload a medical / IEEE research paper PDF and ask:
#
# 1) Page-wise questions
#    - "Explain page 1"
#    - "What is on page 2?"
#    - "Summarize page 3"
#
# 2) Paper-specific questions
#    - "What disease is discussed?"
#    - "What methodology is used?"
#    - "What are the results?"
#
# 3) External / broader medical questions
#    - "What is pneumonia?"
#    - "What are recent treatments for this disease?"
#    - "Compare this with current medical research"
#
# Agent behavior:
# - If user asks about a specific page -> use explain_page
# - If user asks about uploaded paper -> use search_pdf
# - If PDF does not contain enough context / user asks broader medical info -> use web_search (Tinyfish)

import re
from openai import api_key
import requests
import streamlit as st
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_pages(uploaded_file):
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page": i + 1,
            "text": text.strip()
        })
    return pages


def build_page_store(pages):
    page_store = {}
    for p in pages:
        page_store[p["page"]] = p["text"]
    return page_store


def split_chunks_from_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )

    all_chunks = []
    metadatas = []

    for page in pages:
        page_no = page["page"]
        page_text = page["text"]

        if not page_text.strip():
            continue

        chunks = splitter.split_text(page_text)
        for ch in chunks:
            all_chunks.append(ch)
            metadatas.append({"page": page_no})

    return all_chunks, metadatas


def store_chunks(chunks, metadatas):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="medical_hybrid_db",
    )
    return vectorstore


# ── AGENT BUILDER ─────────────────────────────────────────────────────────────

def build_agent(vectorstore, page_store):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        api_key="your_groq_api_key_here",   # put your Groq API key here
        model="openai/gpt-oss-120b",
        temperature=0
    )

    @tool
    def explain_page(query: str) -> str:
        """Explain a specific page from the uploaded medical research paper."""
        q = query.lower()
        page_no = None

        # Detect "page 3"
        match = re.search(r'page\s+(\d+)', q)
        if match:
            page_no = int(match.group(1))
        else:
            # Support "first page", "second page", etc.
            page_word_map = {
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
                "fifth": 5,
                "sixth": 6,
                "seventh": 7,
                "eighth": 8,
                "ninth": 9,
                "tenth": 10
            }
            for word, num in page_word_map.items():
                if f"{word} page" in q or f"page {word}" in q:
                    page_no = num
                    break

        if page_no is None:
            return "I could not detect the page number. Please ask like 'Explain page 2'."

        if page_no not in page_store:
            return f"Page {page_no} not found in the uploaded PDF."

        page_text = page_store[page_no]
        if not page_text.strip():
            return f"Page {page_no} has little or no extractable text."

        prompt = f"""
You are a medical research paper assistant.

Explain ONLY the following page from the uploaded medical / IEEE research paper.

Instructions:
1. Start with: "This page mainly discusses..."
2. Explain the content in simple language.
3. If this page contains abstract, introduction, methodology, results, discussion, conclusion, or references, mention that clearly.
4. If medical terms appear, explain them simply.
5. If there are tables, figures, or formulas, mention what they represent.
6. Do not invent details outside this page.

Page number: {page_no}

Page text:
{page_text}
"""
        response = llm.invoke(prompt)
        return response.content

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF for relevant medical paper information. Always try this first for paper-related questions."""
        docs = retriever.invoke(query)

        if not docs:
            return "NO_CONTEXT_FOUND"

        formatted = []
        for d in docs:
            page_no = d.metadata.get("page", "Unknown")
            formatted.append(f"[Page {page_no}]\n{d.page_content}")

        joined = "\n\n".join(formatted)

        # Weak context guard
        if len(joined.strip()) < 120:
            return "NO_CONTEXT_FOUND"

        return joined

    @tool
    def web_search(query: str) -> str:
        """Search the web using Tinyfish. Use this if its necessary."""
        url = "https://api.search.tinyfish.ai"
        headers = {"X-API-Key": "your_tinyfish_api_key_here"}   # put your Tinyfish API key here
        params = {"query": query}
        
        response = requests.get(url, headers=headers, params=params)
        results = response.json()
        
        output = ""
        for r in results.get("results", []):
            output += (
                f"Title: {r.get('title', '')}\n"
                f"Summary: {r.get('snippet', '')}\n"
                f"URL: {r.get('url', '')}\n\n"
            )
        
        return output if output else "No results found."

    tools = [explain_page, search_pdf, web_search]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a hybrid medical research assistant.

You have access to three tools:

1) explain_page(query)
   - Use this when the user asks about a specific page of the uploaded paper.
   - Examples:
     "Explain page 1"
     "What is on page 2?"
     "Summarize page 3"

2) search_pdf(query)
   - Use this for questions about the uploaded paper.
   - Examples:
     "What disease is discussed?"
     "What methodology is used?"
     "What are the results?"
     "Summarize the paper"

3) web_search(query)
   - Use this if search_pdf returns NO_CONTEXT_FOUND
   - Also use this when the user asks broader medical context outside the uploaded paper.

Rules:
- If the user asks about a specific page, ALWAYS use explain_page first.
- Otherwise, for uploaded paper questions, use search_pdf first.
- If search_pdf returns NO_CONTEXT_FOUND, then use web_search.
- If the user asks something clearly outside the uploaded paper, web_search can be used.
- Answer clearly and in simple language.
- Mention page numbers when answering from the PDF.
- If the answer includes web-based context, make that clear.
- Do not fabricate information.
- The current year is 2026.
"""
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ── STREAMLIT UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hybrid Medical Research Assistant", layout="wide")
st.title("🏥 Hybrid Medical Research Assistant")
st.write(
    "Upload a medical / IEEE research paper PDF and ask page-wise questions, "
    "paper-specific questions, or broader medical questions with Tinyfish web fallback."
)

if "agent" not in st.session_state:
    st.session_state["agent"] = None

if "page_store" not in st.session_state:
    st.session_state["page_store"] = None

if "pages_count" not in st.session_state:
    st.session_state["pages_count"] = 0

if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0

upload_file = st.file_uploader("Upload a PDF file", type="pdf")

if upload_file:
    with st.spinner("Indexing your medical research paper..."):
        pages = extract_pages(upload_file)
        page_store = build_page_store(pages)
        chunks, metadatas = split_chunks_from_pages(pages)
        vectorstore = store_chunks(chunks, metadatas)
        agent_executor = build_agent(vectorstore, page_store)

        st.session_state["agent"] = agent_executor
        st.session_state["page_store"] = page_store
        st.session_state["pages_count"] = len(pages)
        st.session_state["chunk_count"] = len(chunks)

    st.success(
        f"Ready! Indexed {len(pages)} pages and {len(chunks)} chunks from the uploaded paper."
    )

    with st.expander("PDF details"):
        st.write(f"Pages extracted: {len(pages)}")
        st.write(f"Chunks created: {len(chunks)}")

question = st.text_input(
    "Ask a question about the uploaded paper or a related medical topic",
    placeholder=(
        "Examples: Explain page 1 / What disease is discussed? / "
        "What are the results? / What is pneumonia?"
    )
)

if question:
    if st.session_state["agent"] is None:
        st.warning("Please upload a PDF file first.")
    else:
        with st.spinner("Analyzing..."):
            result = st.session_state["agent"].invoke({"input": question})

        st.subheader("Answer")
        st.write(result["output"])

