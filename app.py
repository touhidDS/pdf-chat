import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat (RAG)", page_icon="📄")
st.title("📄 Chat with your PDFs")
st.caption("Upload one or more PDFs, then ask anything about them.")

# ── API Key from Streamlit Secrets ────────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    process_btn = st.button("🚀 Process PDFs", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(files):
    combined = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined += text + "\n"
    return combined


def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def ask_gemini(api_key, context_chunks, question):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text

# ── Process PDFs ──────────────────────────────────────────────────────────────

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting text and building vector index…"):
            raw_text = extract_text(uploaded_files)
            if not raw_text.strip():
                st.error("Could not extract any text. Make sure the PDFs are not scanned images.")
            else:
                st.session_state["vectorstore"] = build_vectorstore(raw_text)
                st.session_state["api_key"] = api_key
                st.sidebar.success(f"✅ Ready! Processed {len(uploaded_files)} file(s).")

# ── Chat Interface ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your PDFs…"):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload and process your PDFs first (use the sidebar).")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                vs = st.session_state["vectorstore"]
                docs = vs.similarity_search(question, k=4)
                context_chunks = [d.page_content for d in docs]
                answer = ask_gemini(st.session_state["api_key"], context_chunks, question)
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
