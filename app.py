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
# CSS — Professional Design System
# ============================================================

CUSTOM_CSS = """

/* ============================================
   RESET & BASE
   ============================================ */

footer { display: none !important; }

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ============================================
   TYPOGRAPHY
   ============================================ */

.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}

.app-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: #1e293b;
    border-radius: 12px;
    margin-bottom: 0.75rem;
}

.app-logo svg {
    width: 24px;
    height: 24px;
    fill: white;
}

.app-title {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.025em;
    margin: 0 !important;
    line-height: 1.2;
    -webkit-text-fill-color: #0f172a !important;
    background: none !important;
}

.app-subtitle {
    color: #64748b !important;
    font-size: 0.95rem !important;
    margin: 0.25rem 0 0 !important;
    font-weight: 400;
}

/* ============================================
   STATS BAR
   ============================================ */

.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 0.75rem 1.5rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin: 1.25rem auto;
    max-width: 520px;
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 500;
}

.stat-icon {
    width: 16px;
    height: 16px;
    color: #3b82f6;
    flex-shrink: 0;
}

.stat-value {
    color: #0f172a;
    font-weight: 600;
}

/* ============================================
   EXAMPLE CHIPS
   ============================================ */

.examples-section {
    margin: 1.5rem auto;
    max-width: 700px;
}

.examples-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    text-align: center;
}

.example-chip {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    background: white !important;
    color: #475569 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
    text-align: center !important;
}

.example-chip:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
    background: #f0f7ff !important;
    box-shadow: 0 1px 3px rgba(59, 130, 246, 0.1) !important;
}

/* ============================================
   CHATBOT
   ============================================ */

.chatbot {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    min-height: 480px !important;
    background: #ffffff !important;
}

/* User message bubbles */
.chatbot .message.user {
    background: #3b82f6 !important;
    color: white !important;
    border-radius: 12px 12px 4px 12px !important;
    padding: 10px 14px !important;
    max-width: 80% !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

/* Bot message bubbles */
.chatbot .message.bot {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border-radius: 12px 12px 12px 4px !important;
    padding: 10px 14px !important;
    max-width: 85% !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

/* ============================================
   INPUT AREA
   ============================================ */

.chat-input textarea {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    background: #ffffff !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.chat-input textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

/* Send button */
.send-btn {
    background: #1e293b !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
}

.send-btn:hover {
    background: #334155 !important;
    box-shadow: 0 2px 8px rgba(30, 41, 59, 0.25) !important;
}

/* Clear button */
.clear-btn {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: white !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}

.clear-btn:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    background: #fef2f2 !important;
}

/* Stop button */
.stop-btn {
    border-radius: 10px !important;
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
}

.stop-btn:hover {
    background: #dc2626 !important;
}

/* ============================================
   FOOTER
   ============================================ */

.app-footer {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 0.5rem;
}

.footer-text {
    font-size: 0.75rem;
    color: #94a3b8;
    letter-spacing: 0.01em;
}

.footer-text strong {
    color: #64748b;
    font-weight: 600;
}

.footer-divider {
    display: inline-block;
    width: 3px;
    height: 3px;
    background: #cbd5e1;
    border-radius: 50%;
    vertical-align: middle;
    margin: 0 0.5rem;
}

/* ============================================
   RESPONSIVE
   ============================================ */

@media (max-width: 640px) {

    .app-title {
        font-size: 1.4rem !important;
    }

    .stats-bar {
        flex-direction: column;
        gap: 0.5rem;
        align-items: center;
    }

    .example-chip {
        font-size: 0.78rem !important;
        padding: 6px 10px !important;
    }
}
"""


# ============================================================
# Build UI
# ============================================================

with gr.Blocks(title="RAG Assistant") as demo:

    # ---- Header ----
    gr.HTML(
        '<div class="app-header">'
        '<div class="app-logo">'
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>'
        "</svg>"
        "</div>"
        '<h1 class="app-title">RAG Assistant</h1>'
        '<p class="app-subtitle">Ask questions about your documents</p>'
        "</div>"
    )

    # ---- Stats ----
    gr.HTML(
        '<div class="stats-bar">'
        '<div class="stat-item">'
        '<svg class="stat-icon" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>'
        "</svg>"
        f'<span><span class="stat-value">{len(chunks)}</span> chunks indexed</span>'
        "</div>"
        '<div class="stat-item">'
        '<svg class="stat-icon" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>'
        "</svg>"
        f'<span><span class="stat-value">tiny-gpt2</span> model</span>'
        "</div>"
        '<div class="stat-item">'
        '<svg class="stat-icon" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0016 9.5 6.5 6.5 0 109.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>'
        "</svg>"
        f'<span><span class="stat-value">{TOP_K}</span> results per query</span>'
        "</div>"
        "</div>"
    )

    # ---- Example Questions ----
    gr.HTML(
        '<div class="examples-section">'
        '<div class="examples-label">Suggested Questions</div>'
        "</div>"
    )

    with gr.Row():

        for ex in EXAMPLES[:4]:

            btn = gr.Button(ex, elem_classes=["example-chip"], scale=1, min_width=0)

    with gr.Row():

        for ex in EXAMPLES[4:]:

            btn = gr.Button(ex, elem_classes=["example-chip"], scale=1, min_width=0)

    # ---- Chat Interface ----
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=480),
        textbox=gr.Textbox(
            placeholder="Ask anything about your documents...",
            container=False,
            scale=7,
        ),
        submit_btn="Send",
        stop_btn="Stop",
        clear_btn="Clear",
    )

    # ---- Footer ----
    gr.HTML(
        '<div class="app-footer">'
        '<div class="footer-text">'
        "Built with <strong>Gradio</strong>"
        '<span class="footer-divider"></span>'
        "Embeddings: <strong>all-MiniLM-L6-v2</strong>"
        '<span class="footer-divider"></span>'
        "LLM: <strong>tiny-gpt2</strong>"
        "</div>"
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
