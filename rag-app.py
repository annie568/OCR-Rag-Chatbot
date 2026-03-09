# streamlit_app.py
# ----------------
"""
RAG Chatbot using LangChain + Gemini 2.0 Flash (Simple Version)

Installation:
pip install langchain-core langchain-google-genai langchain-huggingface langchain-chroma langchain-text-splitters pypdf streamlit chromadb sentence-transformers
"""

import os
from pathlib import Path
import shutil
import streamlit as st  
from pypdf import PdfReader  

# LangChain - Core imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser 

# Google Generative AI (for LLM only)
from langchain_google_genai import ChatGoogleGenerativeAI 

# HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings 

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter  

# Vector store
from langchain_chroma import Chroma  



# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RAG (Gemini 2.0 Flash)", page_icon="✨")
st.header("✨ RAG Chatbot — LangChain + Gemini 2.0 Flash")
st.write("Upload PDFs → Build index → Ask questions. Simple RAG implementation.")


# =========================
# KEYS
# =========================
GOOGLE_API_KEY = "AAIzaSyA7gW8o_BDfq82kSBSANLuTU-fvABSvj_k"
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.text_input(
        "Google API Key (from Google AI Studio)",
        type="password",
        help="Or set the environment variable GOOGLE_API_KEY before running.",
    )
if not GOOGLE_API_KEY:
    st.warning("Please provide GOOGLE_API_KEY to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================
# PATHS
# =========================
UPLOAD_DIR = Path("./uploads")
CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# =========================
# HELPERS
# =========================
def save_uploaded_pdfs(files):
    """Save uploaded PDF files to disk"""
    saved = []
    for uf in files or []:
        if uf.name.lower().endswith(".pdf"):
            dest = UPLOAD_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.read())
            saved.append(dest)
    return saved


def extract_text_from_pdfs(pdf_paths):
    """Extract text from PDF files"""
    texts = []
    for path in pdf_paths:
        try:
            reader = PdfReader(str(path))
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip():
                    pages.append(t)
            doc_text = "\n".join(pages).strip()
            if doc_text:
                texts.append((path.name, doc_text))
        except Exception as e:
            st.error(f"Failed to read {path.name}: {e}")
    return texts


def clear_all():
    """Clear all uploads and index"""
    for p in UPLOAD_DIR.glob("**/*"):
        if p.is_file():
            p.unlink()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)
    
    # Clear session state
    for key in ["retriever", "vectorstore", "llm", "messages"]:
        st.session_state.pop(key, None)
    
    st.success("Cleared uploads and index.")


def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_chat_history_text(messages):
    """Convert chat history to text format"""
    history_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"
    return history_text


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("Controls")
    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        build_btn = st.button("Build / Update Index", use_container_width=True)
    with c2:
        clear_btn = st.button("Clear All", use_container_width=True)

    if clear_btn:
        clear_all()

    if files:
        saved = save_uploaded_pdfs(files)
        if saved:
            st.success(f"Saved {len(saved)} file(s): " + ", ".join(p.name for p in saved))
        else:
            st.warning("No PDFs saved. Please upload valid .pdf files.")


# =========================
# BUILD / LOAD INDEX
# =========================
if build_btn:
    pdf_paths = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_paths:
        st.warning("No PDFs in 'uploads/'. Upload first.")
    else:
        with st.spinner("Building index..."):
            extracted = extract_text_from_pdfs(pdf_paths)
            total_chars = sum(len(text) for _, text in extracted)
            st.info(f"Loaded {len(pdf_paths)} file(s). Extracted ~{total_chars:,} characters.")
            
            if total_chars == 0:
                st.warning("No extractable text found. If PDFs are scans, consider OCR.")
            else:
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, 
                    chunk_overlap=100
                )
                docs = []
                for filename, text in extracted:
                    chunks = splitter.create_documents([text])
                    for chunk in chunks:
                        chunk.metadata = {"source": filename}
                        docs.append(chunk)

                # Create HuggingFace embeddings (runs locally)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Create Chroma vector store
                vectordb = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    persist_directory=str(CHROMA_DIR),
                    collection_name="pdf_collection"
                )
                
                # Create retriever and store both
                st.session_state["vectorstore"] = vectordb
                st.session_state["retriever"] = vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6},
                )
                
                st.success(f"✅ Index built with {len(docs)} chunks! Scroll down to chat.")


# Auto-load existing Chroma index if present
if "retriever" not in st.session_state:
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        try:
            embeddings = GoogleGenerativeAIEmbeddings( # type: ignore
                model="models/embedding-001",
            )
            vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name="pdf_collection"
            )
            st.session_state["vectorstore"] = vectordb
            st.session_state["retriever"] = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6},
            )
        except Exception as e:
            st.sidebar.info("No existing index found. Upload PDFs and build index.")


# =========================
# INITIALIZE LLM
# =========================
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    )


# =========================
# RAG FUNCTION (Simple approach)
# =========================
def answer_question(question, chat_history):
    """Answer question using RAG"""
    
    # Get retriever and LLM
    retriever = st.session_state.get("retriever")
    llm = st.session_state.get("llm")
    
    if not retriever or not llm:
        return "Please build the index first.", []
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    if not docs:
        return "I couldn't find relevant information in the uploaded documents.", []
    
    # Format context
    context = format_docs(docs)
    
    # Format chat history
    history_text = get_chat_history_text(chat_history)
    
    # Create prompt
    system_prompt = """You are a helpful assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise and relevant.

Context:
{context}

Chat History:
{history}"""
    
    messages = [
        SystemMessage(content=system_prompt.format(context=context, history=history_text)),
        HumanMessage(content=question)
    ]
    
    # Get response
    response = llm.invoke(messages)
    answer = response.content
    
    return answer, docs


# =========================
# CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(
            content="👋 Hello! Upload PDFs on the left, build the index, then ask your questions below."
        )
    ]

# Render existing messages
for m in st.session_state["messages"]:
    with st.chat_message("assistant" if isinstance(m, AIMessage) else "user"):
        st.write(m.content)

# Input
user_q = st.chat_input("Ask a question about the uploaded PDFs…")
if user_q:
    st.session_state["messages"].append(HumanMessage(content=user_q))
    with st.chat_message("user"):
        st.write(user_q)

    with st.chat_message("assistant"):
        if "retriever" not in st.session_state:
            response = "⚠️ Please upload PDFs and click **Build / Update Index** first."
            st.warning(response)
            st.session_state["messages"].append(AIMessage(content=response))
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    # Get chat history (exclude current message)
                    history = st.session_state["messages"][:-1]
                    
                    # Get answer
                    answer, docs = answer_question(user_q, history)
                    
                    # Display answer
                    st.write(answer)
                    st.session_state["messages"].append(AIMessage(content=answer))

                    # Show sources
                    if docs:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, d in enumerate(docs, start=1):
                                preview = (d.page_content or "").strip().replace("\n", " ")[:400]
                                src = d.metadata.get("source", "unknown")
                                st.markdown(f"**{i}. {src}**")
                                st.caption(f"{preview}...")
                                if i < len(docs):
                                    st.divider()
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state["messages"].append(AIMessage(content=error_msg))
