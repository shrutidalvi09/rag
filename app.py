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

print(f"Chunks: {len(chunks)}", flush=True)

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
# Chat
# ============================================================

def chat(query, history):

    if not query or not query.strip():

        return history, ""

    results = search(query)

    if not results:

        history.append((query, "No relevant documents found. Try rephrasing."))

        return history, ""

    context_parts = []

    for result in results:

        context_parts.append(result["chunk"].get("text", ""))

    context = "\n".join(context_parts)

    prompt = (
        f"Based on the following information, answer the question.\n\n"
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

    except Exception:

        answer = context_parts[0][:300] + "..."

    # Format answer with sources
    source_text = ""

    for result in results:

        chunk = result["chunk"]

        score_pct = result["score"] * 100

        source_text += (
            f"\n\n---\n"
            f"**Source {result['rank']}** ({chunk.get('filename', 'Unknown')}) "
            f"— Relevance: {score_pct:.0f}%\n"
            f"> {chunk.get('text', '')[:200]}..."
        )

    full_answer = answer + source_text

    history.append((query, full_answer))

    return history, ""


def clear_chat():

    return [], ""


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """

/* Title */
.main-title {
    text-align: center;
    font-size: 2.2em !important;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1em;
    margin-top: -10px !important;
    margin-bottom: 1rem !important;
}

/* Chatbot container */
.chatbot {
    border-radius: 16px !important;
    border: 1px solid #e0e0e0 !important;
    min-height: 400px !important;
}

/* Input */
.query-input textarea {
    border-radius: 16px !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 1.05em !important;
    padding: 14px !important;
}

.query-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
}

/* Send button */
.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
    min-width: 120px !important;
}

.send-btn:hover {
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

/* Clear button */
.clear-btn {
    border-radius: 16px !important;
    border: 2px solid #e0e0e0 !important;
    background: white !important;
}

.clear-btn:hover {
    border-color: #ff4b4b !important;
    color: #ff4b4b !important;
}

/* Examples */
.example-btn {
    border-radius: 20px !important;
    border: 1.5px solid #d0d5ff !important;
    background: white !important;
    font-size: 0.85em !important;
    padding: 8px 16px !important;
}

.example-btn:hover {
    background: #f0f3ff !important;
    border-color: #667eea !important;
}

/* Stats bar */
.stats-bar {
    background: linear-gradient(135deg, #f5f7ff 0%, #f0f3ff 100%);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.85em;
    color: #666;
    border: 1px solid #e8ecff;
    text-align: center;
    margin-top: 0.5rem;
}

footer { display: none !important; }
"""


# ============================================================
# Build UI
# ============================================================

with gr.Blocks(css=CUSTOM_CSS, title="RAG Chatbot") as demo:

    # Header
    gr.Markdown(
        '<div class="main-title">📚 Personal RAG Assistant</div>'
        '<p class="subtitle">Ask questions about your documents</p>'
    )

    # Stats
    gr.Markdown(
        f'<div class="stats-bar">'
        f"📊 {len(chunks)} chunks indexed &nbsp;|&nbsp; "
        f"🤖 tiny-gpt2 &nbsp;|&nbsp; "
        f"🔍 {TOP_K} results per query"
        f"</div>"
    )

    # Examples
    with gr.Row():

        for ex in EXAMPLES[:4]:

            btn = gr.Button(ex, elem_classes=["example-btn"], scale=1)

            btn.click(
                fn=lambda text=ex: text,
                outputs=None,
                js=f"() => {{ document.querySelector('.query-input textarea').value = '{ex}'; return []; }}",
            )

    with gr.Row():

        for ex in EXAMPLES[4:]:

            btn = gr.Button(ex, elem_classes=["example-btn"], scale=1)

            btn.click(
                fn=lambda text=ex: text,
                outputs=None,
                js=f"() => {{ document.querySelector('.query-input textarea').value = '{ex}'; return []; }}",
            )

    gr.Markdown("---")

    # Chatbot
    chatbot = gr.Chatbot(
        label="",
        height=420,
        bubble_full_width=False,
        show_copy_button=True,
        avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/books_1f4da.png"),
    )

    # Input row
    with gr.Row():

        query_input = gr.Textbox(
            label="",
            placeholder="Ask anything about your documents...",
            lines=1,
            max_lines=3,
            elem_classes=["query-input"],
            show_label=False,
            scale=5,
        )

        send_btn = gr.Button(
            "Send ➤",
            elem_classes=["send-btn"],
            scale=1,
        )

        clear_btn = gr.Button(
            "✕",
            elem_classes=["clear-btn"],
            scale=0,
        )

    # Events
    send_btn.click(
        fn=chat,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )

    query_input.submit(
        fn=chat,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, query_input],
    )

    # Footer
    gr.Markdown(
        '<p style="text-align:center; color:#aaa; font-size:0.8em; margin-top:1rem;">'
        "Embedding: all-MiniLM-L6-v2 &nbsp;|&nbsp; LLM: tiny-gpt2 &nbsp;|&nbsp; Built with Gradio"
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
