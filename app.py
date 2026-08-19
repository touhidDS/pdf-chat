import time

import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, NotFound

st.set_page_config(page_title="PDF Chat (RAG)", page_icon="📄")
st.title("Chat with your PDFs")
st.caption("Upload one or more PDFs, then ask questions about their content.")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Models are tried in this order. The lite model is faster/cheaper and
# handles most questions fine; we fall back to the full model if needed.
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

RETRIEVAL_K = 6  # number of chunks pulled from the vector store per question


def extract_text_from_pdfs(files):
    """Pull raw text out of every page of every uploaded PDF."""
    text = ""
    for f in files:
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


def build_vectorstore(text):
    """Chunk the document text and embed it into a FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)


def build_prompt(context_chunks, question):
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
You may reason logically using the information present (for example, if a document lists only a Bachelor's degree and nothing else, you can correctly conclude no Master's degree is mentioned).
Do not use any outside knowledge or make assumptions beyond what the context supports.
If there is truly nothing relevant in the context to answer the question, say "I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""


def ask_gemini(context_chunks, question):
    genai.configure(api_key=GEMINI_API_KEY)
    prompt = build_prompt(context_chunks, question)

    for model_name in GEMINI_MODELS:
        model = genai.GenerativeModel(model_name)
        try:
            return model.generate_content(prompt).text

        except NotFound:
            # Model not available on this account/region, try the next one
            continue

        except InvalidArgument:
            # Usually means the prompt was too long - retry with a single chunk
            short_prompt = build_prompt(context_chunks[:1], question)
            try:
                return model.generate_content(short_prompt).text
            except Exception:
                continue

        except ResourceExhausted:
            # Free tier rate limit - wait a bit and try once more
            time.sleep(10)
            try:
                return model.generate_content(prompt).text
            except ResourceExhausted:
                return "You've hit the Gemini free tier rate limit. Please wait a minute and try again."

        except Exception as e:
            return f"Unexpected error: {e}"

    return "Could not get a response from Gemini right now. Please try again later."


# ---------------------------------------------------------------------------
# Sidebar: upload + process PDFs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    process_clicked = st.button("Process PDFs", use_container_width=True)
    st.divider()
    debug_mode = st.checkbox("Show retrieved chunks (debug)", value=False)

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting text and building the index..."):
            raw_text = extract_text_from_pdfs(uploaded_files)
            if not raw_text.strip():
                st.error("Couldn't extract any text - make sure the PDFs aren't scanned images.")
            else:
                st.session_state.vectorstore = build_vectorstore(raw_text)
                st.sidebar.success(f"Ready! Processed {len(uploaded_files)} file(s).")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your PDFs...")

if question:
    if "vectorstore" not in st.session_state:
        st.warning("Please upload and process your PDFs first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(question, k=RETRIEVAL_K)
                context_chunks = [d.page_content for d in docs]

                if debug_mode:
                    with st.expander("Retrieved chunks"):
                        for i, chunk in enumerate(context_chunks, start=1):
                            st.text(f"Chunk {i}: {chunk[:300]}")

                answer = ask_gemini(context_chunks, question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
