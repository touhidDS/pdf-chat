# PDF Chat (RAG)

A Streamlit app that lets you upload one or more PDFs and ask questions about their content in natural language. Answers are generated using Retrieval-Augmented Generation (RAG) so the model responds only from what's actually in your documents, not from general knowledge.

**Live demo:** https://pdf-chat-zuebbtrz7yqkbepfrgffrb.streamlit.app/

## How it works

1. **Upload** — PDFs are uploaded through the sidebar.
2. **Extract** — Text is pulled from every page using `pdfplumber`.
3. **Chunk & embed** — The text is split into overlapping chunks (`RecursiveCharacterTextSplitter`) and embedded with the `all-MiniLM-L6-v2` sentence-transformer model.
4. **Index** — Chunk embeddings are stored in a local FAISS vector index.
5. **Retrieve** — When you ask a question, the top-k most similar chunks are pulled from the index.
6. **Generate** — Those chunks are passed as context to Gemini (`gemini-2.5-flash-lite`, with `gemini-2.5-flash` as a fallback), which answers strictly from the retrieved context.

This keeps answers grounded in the uploaded documents and avoids the model hallucinating information that isn't there.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Text chunking | LangChain (`langchain-text-splitters`) |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS |
| LLM | Google Gemini API |

## Project structure

```
.
├── app.py              # Streamlit app
├── requirements.txt    # Python dependencies
└── README.md
```

## Running locally

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your Gemini API key to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes / limitations

- Scanned (image-only) PDFs won't work since text extraction relies on a text layer — OCR isn't implemented yet.
- Runs on the Gemini free tier, so heavy usage can hit rate limits (the app retries once and then shows a friendly message rather than crashing).
- The vector index is rebuilt per session and isn't persisted across app restarts.

## Possible improvements

- OCR fallback for scanned documents
- Persisting the FAISS index so re-uploading isn't needed every session
- Source/page citations alongside answers
- Swapping in a larger embedding model for better retrieval on long, technical PDFs

## Author

Touhid Mahmud — [GitHub](https://github.com/touhidDS)
