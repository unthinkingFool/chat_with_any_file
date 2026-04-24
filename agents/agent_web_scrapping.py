import requests
from bs4 import BeautifulSoup

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
from langchain_core.documents import Document

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("doc_chat")
logger.setLevel(logging.DEBUG)



# ── LLM initialisation ───────────────────────────────────────────────────────
def init_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Initialise and return the Groq‑hosted LLM."""
    groq_api_key = os.getenv("GROQ_API_KEY_FOR_WEB")
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

At first give a short answer based on the web page, then give a detailed and most accurate answer.
""")


def scrape_web_page(url: str) -> str:
    """
    Fetches a web page and extracts its main text content using BeautifulSoup.
    Prioritizes semantic content containers and strips boilerplate.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove noise elements
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside", "form", "iframe", "svg"]):
            tag.decompose()

        # Try semantic/common main content containers in priority order
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="main-content") or
            soup.find(id="content") or
            soup.find(class_="post-content") or
            soup.find(class_="article-body") or
            soup.find(class_="entry-content") or
            soup.find("div", role="main") or
            soup.find("body")  # last resort
        )

        if main_content is None:
            return ""

        text = main_content.get_text(separator=' ', strip=True)

        # Collapse excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def load_and_process_docs(
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    web_url: str = None,
) -> Chroma:
    all_docs = []

    if web_url:
        logger.info("Scraping URL: %s", web_url)
        text = scrape_web_page(web_url)
        if text:
            all_docs.append(Document(page_content=text, metadata={"source": web_url}))

    
    
    

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

    WEB_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
 
    llm        = init_llm()
    vs         = load_and_process_docs(web_url=WEB_URL)
    chain, ret = build_rag_chain(llm, vs)
 
    answer  = query_rag(chain, "What is Artificial Intelligence about?")
    sources = get_sources(ret,  "What is Artificial Intelligence about?")
 
    print("\n=== Answer ===\n", answer)
    print("\n=== Sources ===")
    for s in sources:
        print(f"[{s['index']}] {s['source']} (page {s['page']})\n    {s['snippet']}\n")