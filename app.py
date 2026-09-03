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

    # Add sources
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

/* Hide footer */
footer { display: none !important; }

/* Title */
.gradio-container .main-title {
    text-align: center;
    font-size: 2em !important;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

.gradio-container .subtitle {
    text-align: center;
    color: #888;
    font-size: 1em;
    margin-top: -10px !important;
    margin-bottom: 0.5rem !important;
}

/* Chat container */
.chatbot {
    border-radius: 16px !important;
    border: 1px solid #e0e0e0 !important;
    min-height: 450px !important;
}

/* Input box */
.chat-input textarea {
    border-radius: 16px !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 1em !important;
    padding: 12px 16px !important;
}

.chat-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12) !important;
}

/* Send button */
.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    min-width: 100px !important;
    font-weight: 600 !important;
}

.send-btn:hover {
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transform: translateY(-1px);
}

/* Clear button */
.clear-btn {
    border-radius: 16px !important;
    border: 2px solid #e0e0e0 !important;
    background: white !important;
    min-width: 50px !important;
}

.clear-btn:hover {
    border-color: #ff4b4b !important;
    color: #ff4b4b !important;
}

/* Stats */
.stats-bar {
    background: linear-gradient(135deg, #f5f7ff 0%, #f0f3ff 100%);
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 0.82em;
    color: #777;
    border: 1px solid #e8ecff;
    text-align: center;
    margin-bottom: 0.5rem;
}
"""


# ============================================================
# Build UI
# ============================================================

with gr.Blocks(css=CUSTOM_CSS, title="RAG Chatbot", theme=gr.themes.Soft()) as demo:

    # Header
    gr.HTML(
        '<div style="text-align:center; margin-bottom:0.5rem;">'
        '<div class="main-title">📚 RAG Assistant</div>'
        '<p class="subtitle">Ask anything about your documents</p>'
        "</div>"
    )

    # Stats
    gr.HTML(
        f'<div class="stats-bar">'
        f"{len(chunks)} chunks indexed &nbsp;|&nbsp; "
        f"LLM: tiny-gpt2 &nbsp;|&nbsp; "
        f"Results: {TOP_K} per query"
        f"</div>"
    )

    # Chat interface
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            height=450,
            show_copy_button=True,
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/408/books_1f4da.png",
            ),
        ),
        textbox=gr.Textbox(
            placeholder="Ask anything about your documents...",
            container=True,
            scale=7,
            elem_classes=["chat-input"],
        ),
        submit_btn="Send ➤",
        clear_btn="Clear",
        examples=[
            "What is Docker?",
            "What is an EC2 instance?",
            "How do I connect to a Linux server?",
            "What is DNS?",
            "What is HTTPS?",
            "What is a virtual machine?",
            "How does cloud computing work?",
            "What is an API?",
        ],
        cache_examples=False,
    )

    # Footer
    gr.HTML(
        '<p style="text-align:center; color:#aaa; font-size:0.75em; margin-top:0.5rem;">'
        "Embedding: all-MiniLM-L6-v2 | LLM: tiny-gpt2 | Built with Gradio"
        "</p>"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    print("Starting Gradio chatbot...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
