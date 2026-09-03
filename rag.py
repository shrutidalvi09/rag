import json
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
CHUNKS_PATH = "vector_store/chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt2"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30


# ============================================================
# 1. Load Vector Store
# ============================================================

def load_vector_store():

    try:

        embeddings = np.load(EMBEDDINGS_PATH)

        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        print("Vector store loaded successfully.")
        print(f"Chunks: {len(chunks)}")
        print(f"Embedding shape: {embeddings.shape}")

        return embeddings, chunks

    except FileNotFoundError as error:

        print("ERROR: Vector store file not found.")
        print(error)

        exit(1)

    except Exception as error:

        print("ERROR while loading vector store.")
        print(error)

        exit(1)


# ============================================================
# 2. Load Embedding Model
# ============================================================

def load_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded.")

    return model


# ============================================================
# 3. Load LLM
# ============================================================

def load_llm():

    print("Loading LLM (GPT-2)...")

    llm = pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        truncation=True
    )

    print("LLM loaded.")

    return llm


# ============================================================
# 4. Create Query Embedding
# ============================================================

def create_query_embedding(model, query):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return query_embedding


# ============================================================
# 5. Retrieve Relevant Chunks
# ============================================================

def retrieve_chunks(
    query_embedding,
    embeddings,
    chunks,
    top_k=TOP_K,
    threshold=SIMILARITY_THRESHOLD
):

    embeddings_norm = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings_norm
    )[0]

    top_indices = similarities.argsort()[::-1]

    results = []

    for index in top_indices:

        score = similarities[index]

        if score < threshold:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "score": float(score),
                "chunk": chunks[index]
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 6. Build Context from Retrieved Chunks
# ============================================================

def build_context(results):

    context_parts = []

    for result in results:

        chunk = result["chunk"]

        context_parts.append(
            chunk.get("text", "")
        )

    context = "\n".join(context_parts)

    return context


# ============================================================
# 7. Generate Answer with LLM
# ============================================================

def generate_answer(llm, query, context):

    prompt = (
        f"Based on the following information, "
        f"answer the question.\n\n"
        f"Information:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    output = llm(
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        truncation=True
    )

    generated_text = output[0]["generated_text"]

    answer_start = generated_text.find("Answer:") + len("Answer:")

    answer = generated_text[answer_start:].strip()

    lines = answer.split("\n")

    clean_answer = lines[0].strip()

    return clean_answer


# ============================================================
# 8. Display Results
# ============================================================

def display_results(query, results, context, answer, search_time):

    print("\n" + "=" * 70)

    print("QUESTION")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)

    print("RETRIEVED CHUNKS")
    print("=" * 70)

    print(f"Search Time: {search_time:.4f} seconds")

    if not results:

        print("\nNo relevant documents found.")

        print("Try asking a different question.")

        return

    for result in results:

        chunk = result["chunk"]

        print("\n" + "-" * 70)

        print(f"Rank: {result['rank']}")

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source File: "
            f"{chunk.get('filename', 'Unknown')}"
        )

        print("\nChunk:")

        print(chunk.get("text", ""))

    print("\n" + "=" * 70)

    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("=" * 70)


# ============================================================
# 9. Main RAG Pipeline
# ============================================================

def main():

    print("=" * 70)
    print("PERSONAL DOCUMENT RAG ASSISTANT")
    print("=" * 70)

    # Load vector database
    embeddings, chunks = load_vector_store()

    # Load embedding model
    embedding_model = load_embedding_model()

    # Load LLM
    llm = load_llm()

    # Ask user for query
    query = input(
        "\nAsk your question: "
    ).strip()

    # Validate query
    if not query:

        print("\nERROR: Question cannot be empty.")

        return

    # Start timer
    start_time = time.time()

    # Convert query into vector
    query_embedding = create_query_embedding(
        embedding_model,
        query
    )

    # Retrieve relevant chunks
    results = retrieve_chunks(
        query_embedding,
        embeddings,
        chunks,
        top_k=TOP_K,
        threshold=SIMILARITY_THRESHOLD
    )

    # Build context from retrieved chunks
    context = build_context(results)

    # Generate answer with LLM
    answer = ""

    if context:

        answer = generate_answer(
            llm,
            query,
            context
        )

    # Calculate search time
    search_time = time.time() - start_time

    # Display results
    display_results(
        query,
        results,
        context,
        answer,
        search_time
    )


# ============================================================
# 10. Run Program
# ============================================================

if __name__ == "__main__":
    main()
