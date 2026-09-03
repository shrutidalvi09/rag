import json
import numpy as np
import gradio as gr

from huggingface_hub import InferenceClient


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
CHUNKS_PATH = "vector_store/chunks.json"

LLM_MODEL = "sshleifer/tiny-gpt2"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

EXAMPLES = [
    ["What is Docker?"],
    ["What is an EC2 instance?"],
    ["How do I connect to a Linux server?"],
    ["What is DNS?"],
    ["What is HTTPS?"],
    ["What is a virtual machine?"],
    ["How does cloud computing work?"],
    ["What is an API?"],
]


# ============================================================
# Load data
# ============================================================

print("Loading vector store...", flush=True)
embeddings = np.load(EMBEDDINGS_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks: {len(chunks)}", flush=True)

print("Initializing HuggingFace API...", flush=True)
hf_client = InferenceClient()

print("Ready!\n", flush=True)


# ============================================================
# Keyword-based search (no API needed)
# ============================================================

def keyword_search(query, chunks):

    query_words = set(query.lower().split())

    scores = []

    for chunk in chunks:

        text = chunk.get("text", "").lower()

        chunk_words = set(text.split())

        overlap = len(query_words.intersection(chunk_words))

        scores.append(overlap)

    scores = np.array(scores, dtype=float)

    if scores.max() == 0:

        return []

    scores = scores / scores.max()

    top_indices = scores.argsort()[::-1][:TOP_K]

    results = []

    rank = 1

    for idx in top_indices:

        if scores[idx] < 0.1:

            continue

        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "chunk": chunks[idx]
        })

        rank += 1

    return results


# ============================================================
# RAG Function
# ============================================================

def ask_question(query):

    if not query or not query.strip():

        return "Please enter a question.", ""

    # Search using keyword matching (fast, no API)
    results = keyword_search(query, chunks)

    if not results:

        return "No relevant documents found. Try rephrasing your question.", ""

    # Build context
    context_parts = []

    for result in results:

        chunk = result["chunk"]

        context_parts.append(chunk.get("text", ""))

    context = "\n".join(context_parts)

    # Generate answer using HuggingFace API
    prompt = (
        f"Based on the following information, "
        f"answer the question.\n\n"
        f"Information:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    try:

        generated_text = hf_client.text_generation(
            prompt,
            model=LLM_MODEL,
            max_new_tokens=100,
            temperature=0.7,
        )

        answer_start = generated_text.find("Answer:") + len("Answer:")

        answer = generated_text[answer_start:].strip().split("\n")[0]

        if not answer:

            answer = generated_text.strip()

    except Exception as e:

        # Fallback: extract relevant sentence from context
        answer = (
            f"Based on the documents: {context_parts[0][:300]}..."
        )

    # Build sources
    sources = ""

    for result in results:

        chunk = result["chunk"]

        score_pct = result["score"] * 100

        sources += (
            f"### Source {result['rank']} — {chunk.get('filename', 'Unknown')}\n"
            f"**Relevance:** {score_pct:.1f}%\n\n"
            f"> {chunk.get('text', '')[:500]}\n\n"
            f"---\n\n"
        )

    stats = (
        f"**Results:** {len(results)} matches | "
        f"**Chunks:** {len(chunks)}"
    )

    return answer, sources + "\n\n" + stats


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
.main-title {
    text-align: center;
    font-size: 2em !important;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    text-align: center;
    color: #666;
    font-size: 1.05em;
}

footer { display: none !important; }
"""


# ============================================================
# Build UI
# ============================================================

demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="e.g. What is Docker?",
        lines=1
    ),
    outputs=[
        gr.Textbox(label="Answer", lines=3),
        gr.Markdown(label="Sources")
    ],
    title="📚 Personal Document RAG Assistant",
    description=(
        "Ask questions about your documents. "
        "Powered by AI search & generation."
    ),
    examples=EXAMPLES,
    css=CUSTOM_CSS,
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    print("Starting Gradio app...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
