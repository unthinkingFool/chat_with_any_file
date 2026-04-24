import os
import re
import logging
from typing import List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("code_chat")
logger.setLevel(logging.DEBUG)


# ── LLM initialisation ───────────────────────────────────────────────────────
def init_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Initialise and return the Groq‑hosted LLM."""
    groq_api_key = os.getenv("GROQ_API_KEY_FOR_CODE")
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

Answer: give a concise answer based on the context.Think carefully about the code's functionality and accuracy before answering.
""")




SUPPORTED_EXTENSIONS = [".py", ".js", ".java", ".c", ".cpp"]


# ── Step 1 : Read the code file ───────────────────────────────────────────────
def read_code_file(file_path: str) -> str:
    """
    Reads a code file and returns its content as a plain string.
    Supports .py, .js, .java, .c, .cpp files.
    """

    # extracting file extension
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    logger.info("File read successfully: %s (%d characters)", file_path, len(content))
    return content



# ── Step 2 : Load code into vectorstore ───────────────────────────────────────
def load_and_process_code(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Reads the code file, splits it into chunks, embeds them,
    and stores in ChromaDB. Returns the vectorstore.
    """
    # read raw code string
    code_text = read_code_file(file_path)

    # wrap in a Document so LangChain can process it
    doc = Document(
        page_content=code_text,
        metadata={"source": file_path}
    )

    logger.info("Splitting code into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents([doc])
    logger.info("Total chunks created: %d", len(chunks))

    logger.info("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Storing chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    logger.info("Vectorstore ready with %d chunks.", len(chunks))

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
    
    # ── Config ────────────────────────────────────────────────────────────────────
    CODE_FILE_PATH = "../code_exp/test.cpp"   
 
    llm        = init_llm()
    vs         = load_and_process_code(CODE_FILE_PATH)
    chain, ret = build_rag_chain(llm, vs)
 
    answer  = query_rag(chain, "what is the context of this code, the use case, time and space complexity?")
    sources = get_sources(ret,  "what is the context of this code, the use case, time and space complexity?")
 
    print("\n=== Answer ===\n", answer)
    print("\n=== Sources ===")
    for s in sources:
        print(f"[{s['index']}] {s['source']} (page {s['page']})\n    {s['snippet']}\n")