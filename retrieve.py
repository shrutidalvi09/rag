import json
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = "vector_store/embeddings.npy"
CHUNKS_PATH = "vector_store/chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"

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

def load_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ============================================================
# 3. Create Query Embedding
# ============================================================

def create_query_embedding(
    model,
    query
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return query_embedding


# ============================================================
# 4. Retrieve Relevant Chunks
# ============================================================

def retrieve_chunks(
    query_embedding,
    embeddings,
    chunks,
    top_k=TOP_K,
    threshold=SIMILARITY_THRESHOLD
):

    # Normalize stored embeddings
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Sort by highest similarity
    top_indices = similarities.argsort()[::-1]

    results = []

    for index in top_indices:

        score = similarities[index]

        # Ignore weak matches
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
# 5. Display Results
# ============================================================

def display_results(
    query,
    results,
    search_time
):

    print("\n" + "=" * 70)

    print("QUESTION")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)

    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Search Time: {search_time:.4f} seconds"
    )

    if not results:

        print(
            "\nNo relevant documents found."
        )

        print(
            "Try asking a different question."
        )

        return

    for result in results:

        chunk = result["chunk"]

        print("\n" + "-" * 70)

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source File: "
            f"{chunk.get('filename', 'Unknown')}"
        )

        print("\nChunk:")

        print(
            chunk.get("text", "")
        )

    print("\n" + "=" * 70)


# ============================================================
# 6. Main Retrieval Pipeline
# ============================================================

def main():

    print("=" * 70)
    print("PERSONAL DOCUMENT SEMANTIC SEARCH")
    print("=" * 70)

    # Load vector database
    embeddings, chunks = load_vector_store()

    # Load embedding model
    model = load_model()

    # Ask user for query
    query = input(
        "\nAsk your question: "
    ).strip()

    # Validate query
    if not query:

        print(
            "\nERROR: Question cannot be empty."
        )

        return

    # Start timer
    start_time = time.time()

    # Convert query into vector
    query_embedding = create_query_embedding(
        model,
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

    # Calculate search time
    search_time = time.time() - start_time

    # Display results
    display_results(
        query,
        results,
        search_time
    )


# ============================================================
# 7. Run Program
# ============================================================

if __name__ == "__main__":
    main()