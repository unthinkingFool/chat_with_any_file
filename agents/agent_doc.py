# agent to read and generate answer for pdf,docx,txt,ppt
#using docling for handling multi-type document

import os
import logging
import tempfile
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_docling.loader import DoclingLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()




FILE_PATH = "https://arxiv.org/pdf/2408.09869"

loader = DoclingLoader(file_path=FILE_PATH)


# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("doc_chat")
logger.setLevel(logging.DEBUG)


# ── LLM initialisation ───────────────────────────────────────────────────────
def init_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Initialise and return the Groq‑hosted LLM."""
    groq_api_key = os.getenv("GROQ_API_KEY_FOR_PDF")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    logger.info("Initialising LLM: %s", model_name)
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)


# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""\
You are a helpful assistant that answers questions based on the following context.
Please provide the most accurate answers based on the information given.
If the answer is not found in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:

At first give a short answer based on the document of these types (pdf,docx,txt,ppt), then give a detailed and most accurate answer.
""")


# ── document processing ────────────────────────────────────────────────────────────
def load_and_process_docs(
    doc_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Load one or more documents (PDF, DOCX, PPTX, HTML, TXT, URLs, ...),
    split into chunks, embed, and store in ChromaDB.
 
    DoclingLoader handles format detection automatically — no per-format
    branching needed.  Just pass any supported path or URL.
 
    Returns the Chroma vectorstore.
    """
    all_docs = []
    for path in doc_paths:
        logger.info("Loading document : %s", path)

        loader = DoclingLoader(file_path=path)
        all_docs.extend(loader.load())

    logger.info("Total documents loaded: %d", len(all_docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info("Total chunks created: %d", len(chunks))

    logger.info("Creating embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Storing chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    logger.info("Vectorstore ready with %d documents.", len(chunks))
    return vectorstore


# ── RAG chain ─────────────────────────────────────────────────────────────────
def _format_docs(docs):
    """Join document page_content with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm: ChatGroq, vectorstore: Chroma, k: int = 5):
    """Build and return the RAG chain + retriever."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    logger.info("Retriever ready (k=%d).", k)

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


# ── Query helpers ─────────────────────────────────────────────────────────────
def query_rag(rag_chain, question: str) -> str:
    """Run the RAG chain and return the answer string."""
    logger.info("Querying RAG chain: %s", question)
    response = rag_chain.invoke(question)
    logger.info("Response received (%d chars).", len(response))
    return response


def get_sources(retriever, question: str) -> List[dict]:
    """Return source metadata for the retrieved documents."""
    logger.info("Retrieving sources for: %s", question)
    docs = retriever.invoke(question)
    sources = []
    for i, doc in enumerate(docs):
        sources.append(
            {
                "index": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:300],
            }
        )
    return sources



#-----------demo-------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
 
    # Works with any mix of file types or URLs
    DOC_PATHS = [
        "https://arxiv.org/pdf/2408.09869",   # PDF via URL
        # "report.docx",
        # "slides.pptx",
        # "page.html",
    ]
 
    llm        = init_llm()
    vs         = load_and_process_docs(DOC_PATHS)
    chain, ret = build_rag_chain(llm, vs)
 
    answer  = query_rag(chain, "What is this document about?")
    sources = get_sources(ret,  "What is this document about?")
 
    print("\n=== Answer ===\n", answer)
    print("\n=== Sources ===")
    for s in sources:
        print(f"[{s['index']}] {s['source']} (page {s['page']})\n    {s['snippet']}\n")