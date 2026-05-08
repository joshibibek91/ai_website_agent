import os
import requests
import numpy as np
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Website Intelligence Agent", page_icon="🌐")
st.title("🌐 Website Intelligence Agent")
st.caption("Enter a website URL and ask questions about its content.")

# ── Session state — ALL variables initialized at the very top ─
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter your Google API Key", type="password")
    st.caption("Get your free key at aistudio.google.com")
    st.divider()
    url_input = st.text_input("Website URL", placeholder="https://example.com/article")
    load_button = st.button("🔍 Load Website")
    st.divider()
    if st.button("🗑️ Clear everything"):
        for key in ["chunks", "faiss_index", "messages", "loaded_url",
                    "api_key_valid", "llm", "embeddings_model"]:
            st.session_state[key] = None if key not in ["chunks", "messages"] else []
        st.session_state.api_key_valid = False
        st.rerun()
    if st.session_state.loaded_url:
        st.success(f"✅ Active: {st.session_state.loaded_url}")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")

# ── Stop if no API key ────────────────────────────────────────
if not api_key:
    st.info("👈 Please enter your Google API key in the sidebar to begin.")
    st.stop()

# ── Set API key in environment ────────────────────────────────
os.environ["GOOGLE_API_KEY"] = api_key

# ── Validate API key ONCE ─────────────────────────────────────
if not st.session_state.api_key_valid:
    try:
        with st.spinner("Validating API key..."):
            test_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
            test_llm.invoke([{"role": "user", "content": "hi"}])
            st.session_state.api_key_valid = True
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", temperature=0.1
            )
            st.session_state.embeddings_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2"
            )
    except Exception as e:
        st.error("❌ Invalid API key or connection failed.")
        st.caption(f"Error detail: {e}")
        st.stop()

# ── Load website when button clicked ─────────────────────────
if load_button:
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        with st.spinner("Fetching and indexing website..."):
            try:
                resp = requests.get(
                    url_input,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10
                )
                soup = BeautifulSoup(resp.content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                page_text = soup.get_text(separator="\n", strip=True)

                if not page_text.strip():
                    st.error("❌ Could not extract text from this URL.")
                    st.stop()

                data = [Document(page_content=page_text, metadata={"source": url_input})]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )
                chunks = text_splitter.split_documents(data)

                texts = [c.page_content for c in chunks]
                chunk_embeddings = st.session_state.embeddings_model.embed_documents(texts)
                vectors = np.array(chunk_embeddings, dtype=np.float32)
                index = faiss.IndexFlatL2(vectors.shape[1])
                index.add(vectors)

                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.loaded_url = url_input
                st.session_state.messages = []

                st.success(f"✅ Loaded {len(chunks)} chunks from {url_input}")

            except Exception as e:
                st.error(f"❌ Failed to load website: {e}")

# ── Block chat if no website loaded ──────────────────────────
if st.session_state.faiss_index is None:
    st.warning("👈 No website loaded yet. Enter a URL and click **Load Website** first.")
    st.stop()

# ── Display chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about the loaded website...")

if user_input:

    # ── Safety check — belt and suspenders ───────────────────
    if st.session_state.faiss_index is None or st.session_state.llm is None:
        st.error("⚠️ Please load a website first before asking questions.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ── Retrieve relevant chunks ──────────────────
                query_vec = np.array(
                    [st.session_state.embeddings_model.embed_query(user_input)],
                    dtype=np.float32
                )
                _, indices = st.session_state.faiss_index.search(query_vec, 4)
                retrieved = [
                    st.session_state.chunks[i]
                    for i in indices[0]
                    if i < len(st.session_state.chunks)
                ]
                context = "\n\n".join(doc.page_content for doc in retrieved)

                # ── Build prompt and call LLM directly ───────
                prompt_text = (
                    f"Answer the question based only on the following context:\n\n"
                    f"{context}\n\n"
                    f"Question: {user_input}\n\n"
                    f"If the answer is not in the context, say: "
                    f"'This information is not available in the loaded website.'"
                )

                response = st.session_state.llm.invoke(
                    [{"role": "user", "content": prompt_text}]
                )
                answer = response.content

                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"❌ Error getting answer: {e}")