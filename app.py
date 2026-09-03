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
# Load data
# ============================================================

print("Loading vector store...", flush=True)
embeddings = np.load(EMBEDDINGS_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks: {len(chunks)} loaded", flush=True)

hf_client = InferenceClient()

print("Ready!\n", flush=True)


# ============================================================
# Search
# ============================================================

def search(query):

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
# Chat function
# ============================================================

def respond(message, history):

    if not message or not message.strip():

        return ""

    results = search(message)

    if not results:

        return "No relevant documents found. Try rephrasing your question."

    context_parts = []

    for result in results:

        context_parts.append(result["chunk"].get("text", ""))

    context = "\n".join(context_parts)

    prompt = (
        f"Based on the following information, answer the question.\n\n"
        f"Information:\n{context}\n\n"
        f"Question: {message}\n"
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

    except Exception:

        answer = context_parts[0][:300] + "..."

    sources = "\n\n---\n**Sources:**\n"

    for result in results:

        chunk = result["chunk"]

        score_pct = result["score"] * 100

        sources += (
            f"- **{chunk.get('filename', 'Unknown')}** "
            f"(Relevance: {score_pct:.0f}%)\n"
        )

    return answer + sources


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """

footer { display: none !important; }

.app-title {
    text-align: center;
    font-size: 2.4em !important;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

.app-subtitle {
    text-align: center;
    color: #888;
    font-size: 1.05em;
    margin-top: -8px !important;
    margin-bottom: 1rem !important;
}

.stats-card {
    background: white;
    border-radius: 14px;
    padding: 10px 20px;
    font-size: 0.85em;
    color: #666;
    border: 1px solid #e0e0e0;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stats-card strong { color: #667eea; }

.example-chip {
    border-radius: 20px !important;
    border: 1.5px solid #d0d5ff !important;
    background: white !important;
    font-size: 0.82em !important;
    padding: 6px 14px !important;
    transition: all 0.25s ease !important;
}

.example-chip:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-color: transparent !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35) !important;
}

.app-footer {
    text-align: center;
    color: #bbb;
    font-size: 0.75em;
    margin-top: 1rem;
    padding: 0.5rem;
}

.app-footer span { color: #667eea; font-weight: 600; }
"""


# ============================================================
# Build UI
# ============================================================

with gr.Blocks(title="RAG Chatbot") as demo:

    gr.HTML(
        '<div class="app-title">📚 RAG Assistant</div>'
        '<p class="app-subtitle">Ask anything about your documents</p>'
    )

    gr.HTML(
        f'<div class="stats-card">'
        f"📊 <strong>{len(chunks)}</strong> chunks indexed &nbsp;&bull;&nbsp; "
        f"🤖 <strong>tiny-gpt2</strong> &nbsp;&bull;&nbsp; "
        f"🔍 <strong>{TOP_K}</strong> results"
        f"</div>"
    )

    with gr.Row():

        for ex in EXAMPLES[:4]:

            btn = gr.Button(ex, elem_classes=["example-chip"], scale=1, min_width=0)

    with gr.Row():

        for ex in EXAMPLES[4:]:

            btn = gr.Button(ex, elem_classes=["example-chip"], scale=1, min_width=0)

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=480),
        textbox=gr.Textbox(
            placeholder="Type your question here...",
            container=False,
            scale=7,
        ),
        submit_btn="Send ➤",
        stop_btn="⏹ Stop",
        clear_btn="🗑 Clear",
    )

    gr.HTML(
        '<div class="app-footer">'
        "Built with <span>Gradio</span> &bull; "
        "Embeddings: <span>all-MiniLM-L6-v2</span> &bull; "
        "LLM: <span>tiny-gpt2</span>"
        "</div>"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    print("Starting Gradio chatbot...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
