import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load documents
# -----------------------------

def load_documents():

    documents = []

    data_folder = "data"

    for filename in os.listdir(data_folder):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                data_folder,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

                documents.append({
                    "filename": filename,
                    "text": text
                })

    return documents


# -----------------------------
# 2. Create chunks
# -----------------------------

def create_chunks(text, chunk_size=50, overlap=10):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# 3. Load documents
# -----------------------------

documents = load_documents()


# -----------------------------
# 4. Create chunks
# -----------------------------

all_chunks = []

for document in documents:

    chunks = create_chunks(
        document["text"]
    )

    for chunk in chunks:

        all_chunks.append({
            "filename": document["filename"],
            "text": chunk
        })


print("Number of chunks:", len(all_chunks))


# -----------------------------
# 5. Load embedding model
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# 6. Create embeddings
# -----------------------------

chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings = model.encode(
    chunk_texts
)


# -----------------------------
# 7. Create vector_store folder
# -----------------------------

os.makedirs(
    "vector_store",
    exist_ok=True
)


# -----------------------------
# 8. Save embeddings
# -----------------------------

np.save(
    "vector_store/embeddings.npy",
    embeddings
)


# -----------------------------
# 9. Save chunk information
# -----------------------------

with open(
    "vector_store/chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=4
    )


print("\nIndex created successfully!")

print("Saved:")
print("vector_store/embeddings.npy")
print("vector_store/chunks.json")