import os
import logging
import textwrap
import tempfile

# ── env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()   # reads GROQ_API_KEY from a .env file in the same folder

# ── third-party ───────────────────────────────────────────────────────────────
import whisper                                          # local transcription (free)

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("video_rag")

# =============================================================================
# CONFIGURATION  ←  Only change things here!
# =============================================================================
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "audio", "audio.mp3")
GROQ_MODEL  = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large
                               # 'base' is fast & good enough for most videos
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Transcribe audio with Whisper (runs locally, 100% free)
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str, model_size: str = WHISPER_MODEL_SIZE) -> str:
    """
    Uses OpenAI Whisper (runs on your machine, no API key needed) to transcribe
    the audio file into plain text.

    Parameters
    ----------
    audio_path  : path to the .mp3 file
    model_size  : whisper model to use ('base' is a good default)

    Returns
    -------
    Full transcript as a single string
    """
    logger.info("  Loading Whisper model '%s' …", model_size)
    model = whisper.load_model(model_size)

    logger.info(" Transcribing audio (this may take a minute) …")
    result = model.transcribe(audio_path, fp16=False)

    transcript = result["text"].strip()
    logger.info(" Transcription done — %d characters", len(transcript))
    return transcript


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build a ChromaDB vector store from the transcript
# ─────────────────────────────────────────────────────────────────────────────
def build_vectorstore(transcript: str, video_title: str) -> Chroma:
    """
    Splits the transcript into overlapping chunks, embeds them with a free
    HuggingFace model, and stores everything in ChromaDB (in-memory).

    Parameters
    ----------
    transcript  : the full transcript string
    video_title : used as metadata so you know which video each chunk came from

    Returns
    -------
    A Chroma vectorstore ready for similarity search
    """
    logger.info("✂  Splitting transcript into chunks …")

    # Wrap the transcript in a LangChain Document object
    doc = Document(
        page_content=transcript,
        metadata={"source": video_title}
    )

    # Split into overlapping chunks so context is preserved at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # ~1000 characters per chunk
        chunk_overlap=100,     # 100 char overlap between consecutive chunks
    )
    chunks = splitter.split_documents([doc])
    logger.info("   Created %d chunks", len(chunks))

    # Free sentence-transformer model for embeddings — no API key needed
    logger.info(" Computing embeddings (sentence-transformers/all-MiniLM-L6-v2) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info(" Storing in ChromaDB …")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    logger.info(" Vectorstore ready (%d chunks indexed)", len(chunks))
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build the LangChain RAG chain
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_chain(vectorstore: Chroma):
    """
    Wires together:
      retriever  →  fetches the most relevant transcript chunks
      prompt     →  tells the LLM to answer ONLY from the video content
      LLM        →  Groq's free llama-3.3-70b-versatile

    Parameters
    ----------
    vectorstore : the Chroma store built in Step 3

    Returns
    -------
    (rag_chain, retriever) tuple
    """
    groq_api_key = os.getenv("GROQ_API_KEY_FOR_AUDIO")
    if not groq_api_key:
        raise ValueError(
            "   GROQ_API_KEY not found!\n"
            "   Create a free key at https://console.groq.com and add it to .env"
        )

    # Free Groq-hosted LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=GROQ_MODEL)
    logger.info(" LLM ready: %s", GROQ_MODEL)

    # Retriever — fetches top-5 most relevant chunks per question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt — strictly grounds answers in the transcript
    prompt = ChatPromptTemplate.from_template("""\
You are a helpful assistant. Answer the user's question ONLY based on the
video transcript provided below. If the answer is not in the transcript,
say "I couldn't find that in the video."

--- VIDEO TRANSCRIPT CONTEXT ---
{context}
---------------------------------

Question: {question}

Answer (be clear and concise): At first give a short answer based on the document of these types (pdf,docx,txt,ppt), then give a detailed and most accurate answer.
""")

    def format_docs(docs):
        """Joins retrieved chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # The RAG chain: retrieve → format → prompt → LLM → parse
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(" RAG chain built and ready")
    return rag_chain, retriever




# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Ties all steps together
# ─────────────────────────────────────────────────────────────────────────────



def main():
    print("\n🎬  Video RAG — Ask a Fixed Question")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Step 2 — Transcribe
        transcript = transcribe_audio(AUDIO_PATH)

        # Step 3 — Vectorstore
        vectorstore = build_vectorstore(transcript, "local_audio")

        # Step 4 — RAG chain
        rag_chain, retriever = build_rag_chain(vectorstore)

        # ✅ Step 5 — Hardcoded question
        question = "What is this audio about?"

        answer = rag_chain.invoke(question)

        print(" Answer:\n")
        print(answer)


        # ✅ (Optional) Show sources like your doc system
        docs = retriever.invoke(question)

        print("\n Sources:\n")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.metadata.get('source', 'video')}")
            print(doc.page_content[:200])
            print("-" * 40)

if __name__ == "__main__":
    main()