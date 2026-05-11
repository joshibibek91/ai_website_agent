import os
import uuid
import requests
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

app = FastAPI(title="Website Intelligence Agent")

# In-memory session store: session_id -> session data
sessions: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────

class LoadRequest(BaseModel):
    api_key: str
    url: str

class LoadResponse(BaseModel):
    session_id: str
    chunks_indexed: int
    message: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str


# ── Helpers ───────────────────────────────────────────────────

def scrape_text(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def build_index(chunks: list[Document], embeddings_model) -> faiss.IndexFlatL2:
    texts = [c.page_content for c in chunks]
    vectors = np.array(embeddings_model.embed_documents(texts), dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/load", response_model=LoadResponse)
def load_website(body: LoadRequest):
    os.environ["GOOGLE_API_KEY"] = body.api_key

    # Validate key
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
        llm.invoke([{"role": "user", "content": "hi"}])
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {e}")

    # Scrape
    try:
        page_text = scrape_text(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not page_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this URL.")

    # Chunk and index
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents([Document(page_content=page_text, metadata={"source": body.url})])
    index = build_index(chunks, embeddings_model)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "llm": llm,
        "embeddings_model": embeddings_model,
        "chunks": chunks,
        "index": index,
        "url": body.url,
    }

    return LoadResponse(
        session_id=session_id,
        chunks_indexed=len(chunks),
        message=f"Loaded {len(chunks)} chunks from {body.url}",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /load first.")

    embeddings_model = session["embeddings_model"]
    index: faiss.IndexFlatL2 = session["index"]
    chunks: list[Document] = session["chunks"]
    llm = session["llm"]

    # Retrieve top-4 relevant chunks
    query_vec = np.array([embeddings_model.embed_query(body.question)], dtype=np.float32)
    _, indices = index.search(query_vec, 4)
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(doc.page_content for doc in retrieved)

    prompt = (
        f"Answer the question based only on the following context:\n\n"
        f"{context}\n\n"
        f"Question: {body.question}\n\n"
        f"If the answer is not in the context, say: "
        f"'This information is not available in the loaded website.'"
    )

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return ChatResponse(answer=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"message": "Session cleared."}


@app.get("/")
def root():
    return {"message": "Website Intelligence Agent API — visit /docs for the interactive UI."}
