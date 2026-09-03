import json
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
CHUNKS_PATH = "vector_store/chunks.json"

EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
LLM_MODEL = "sshleifer/tiny-gpt2"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

EXAMPLES = [
    "What is Docker?",
    "What is an EC2 instance?",
    "How do I connect to a Linux server?",
    "What is DNS?",
    "What is HTTPS?",
    "What is a virtual machine?",
    "How does cloud computing work?",
    "What is an API?",
]


# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }

    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700 !important;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 2rem;
    }

    /* Example buttons */
    .stButton > button {
        border-radius: 20px;
        border: 1.5px solid #d0d5ff;
        background: white;
        padding: 0.4rem 1.2rem;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: #f0f3ff;
        border-color: #667eea;
        color: #667eea;
    }

    /* Ask button */
    .ask-btn > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
    }

    .ask-btn > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Answer box */
    .answer-container {
        background: #f8f9ff;
        border: 2px solid #e8ecff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Sources panel */
    .sources-container {
        background: #fafbff;
        border: 1px solid #e8ecff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Stats bar */
    .stats-bar {
        background: linear-gradient(135deg, #f5f7ff 0%, #f0f3ff 100%);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-size: 0.9em;
        color: #555;
        border: 1px solid #e8ecff;
        margin-bottom: 1rem;
    }

    /* Source cards */
    .source-card {
        background: white;
        border: 1px solid #e8ecff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        transition: border-color 0.2s;
    }

    .source-card:hover {
        border-color: #667eea;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e8ecff;
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85em;
        margin-top: 3rem;
        padding: 1rem;
    }

    /* Input focus */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load models (cached)
# ============================================================

@st.cache_resource
def load_models():

    print("Loading vector store...", flush=True)
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Chunks: {len(chunks)}", flush=True)

    print("Loading embedding model...", flush=True)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading LLM (GPT-2)...", flush=True)
    llm = pipeline(
        "text-generation",
        model=LLM_MODEL,
        truncation=True
    )

    print("All models loaded!\n", flush=True)

    return embeddings, chunks, embedding_model, llm


# ============================================================
# RAG Function
# ============================================================

def ask_question(query, embeddings, chunks, embedding_model, llm):

    if not query or not query.strip():

        return None, None, "Please enter a question."

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # Normalize stored embeddings
    embeddings_norm = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    # Calculate similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings_norm
    )[0]

    # Sort by score
    top_indices = similarities.argsort()[::-1]

    # Collect results
    results = []

    for index in top_indices:

        score = similarities[index]

        if score < SIMILARITY_THRESHOLD:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "score": float(score),
                "chunk": chunks[index]
            }
        )

        if len(results) >= TOP_K:
            break

    if not results:

        return None, None, "No relevant documents found. Try rephrasing your question."

    # Build context
    context_parts = []

    for result in results:

        chunk = result["chunk"]

        context_parts.append(chunk.get("text", ""))

    context = "\n".join(context_parts)

    # Generate answer
    prompt = (
        f"Based on the following information, "
        f"answer the question.\n\n"
        f"Information:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    output = llm(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        truncation=True
    )

    generated_text = output[0]["generated_text"]

    answer_start = generated_text.find("Answer:") + len("Answer:")

    answer = generated_text[answer_start:].strip().split("\n")[0]

    # Build sources
    sources = []

    for result in results:

        chunk = result["chunk"]

        sources.append({
            "rank": result["rank"],
            "score": result["score"],
            "filename": chunk.get("filename", "Unknown"),
            "text": chunk.get("text", ""),
        })

    # Stats
    stats = {
        "matches": len(results),
        "top_score": results[0]["score"],
        "total_chunks": len(chunks),
    }

    return answer, sources, stats


# ============================================================
# UI
# ============================================================

# Load models
embeddings, chunks, embedding_model, llm = load_models()

# Header
st.markdown(
    '<h1>📚 Personal Document RAG Assistant</h1>'
    '<p class="subtitle">Ask questions about your documents — powered by AI search & generation</p>',
    unsafe_allow_html=True,
)

# Stats bar
st.markdown(
    f'<div class="stats-bar">'
    f"📊 <strong>Vector Store:</strong> {len(chunks)} chunks indexed &nbsp; | &nbsp; "
    f"🤖 <strong>LLM:</strong> tiny-gpt2 &nbsp; | &nbsp; "
    f"🔗 <strong>Embeddings:</strong> paraphrase-MiniLM-L3-v2"
    f"</div>",
    unsafe_allow_html=True,
)

# Example questions
st.markdown("**💡 Try these questions:**")

cols = st.columns(4)

for i, example in enumerate(EXAMPLES):

    col = cols[i % 4]

    with col:

        if st.button(example, key=f"ex_{i}", use_container_width=True):

            st.session_state["query"] = example

            st.rerun()

# Divider
st.markdown("---")

# Query input
query = st.text_area(
    "",
    value=st.session_state.get("query", ""),
    placeholder="Ask anything about your documents...",
    height=68,
    key="query_input",
    label_visibility="collapsed",
)

# Action buttons
col1, col2, col3 = st.columns([4, 1, 1])

with col1:

    ask_clicked = st.button(
        "🔍  Ask Question",
        type="primary",
        use_container_width=True,
        key="ask_btn",
    )

with col2:

    if st.button("✕  Clear", use_container_width=True):

        st.session_state["query"] = ""

        st.rerun()

# Process query
if ask_clicked and query:

    with st.spinner("🔍 Searching documents and generating answer..."):

        answer, sources, stats = ask_question(
            query, embeddings, chunks, embedding_model, llm
        )

    if answer is None:

        st.error(stats)

    else:

        # Answer
        st.markdown("---")

        st.markdown("### 💬 Answer")

        st.markdown(
            f'<div class="answer-container">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Stats
        if isinstance(stats, dict):

            st.markdown(
                f'<div class="stats-bar">'
                f"✅ <strong>{stats['matches']}</strong> matches found &nbsp; | &nbsp; "
                f"🎯 Top score: <strong>{stats['top_score']:.4f}</strong> &nbsp; | &nbsp; "
                f"📦 Searched <strong>{stats['total_chunks']}</strong> chunks"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Sources
        st.markdown("### 📑 Sources & References")

        for src in sources:

            score_pct = src["score"] * 100

            st.markdown(
                f'<div class="source-card">'
                f"<strong>📄 Source {src['rank']}</strong> — {src['filename']}<br>"
                f'<span style="color:#667eea;">Relevance: {score_pct:.1f}%</span>'
                f"<br><br>"
                f"<em>{src['text']}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    '<div class="footer">'
    "Built with Streamlit &nbsp;•&nbsp; Embedding: paraphrase-MiniLM-L3-v2 &nbsp;•&nbsp; LLM: tiny-gpt2"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    print("Starting Streamlit app...", flush=True)
    print("Run: streamlit run app.py", flush=True)
