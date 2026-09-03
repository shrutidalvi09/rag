# Personal RAG Assistant — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [How RAG Works](#3-how-rag-works)
4. [File Structure](#4-file-structure)
5. [Data Pipeline](#5-data-pipeline)
6. [Search Algorithm](#6-search-algorithm)
7. [LLM Integration](#7-llm-integration)
8. [User Interface](#8-user-interface)
9. [Deployment](#9-deployment)
10. [API Flow](#10-api-flow)
11. [Configuration](#11-configuration)
12. [Limitations](#12-limitations)

---

## 1. Project Overview

### What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:

- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved information

Instead of relying solely on the LLM's training data, RAG fetches relevant documents first, then uses them as context for the LLM to generate accurate, grounded answers.

### What This Project Does

This project is a **Personal Document RAG Assistant** that:

1. Reads documents from text files
2. Splits them into chunks
3. Creates embeddings (vector representations)
4. Stores them in a vector database
5. When a user asks a question:
   - Searches for relevant chunks
   - Sends chunks + question to an LLM
   - Returns the generated answer with sources

### Knowledge Base

The system is trained on 4 documents covering:

| Document | Topics |
|----------|--------|
| `aws.txt` | AWS, EC2, S3, Lambda |
| `docker.txt` | Docker, containers, images, Dockerfiles |
| `linux.txt` | Linux, SSH, ls, cd, pwd commands |
| `networking.txt` | DNS, firewall, TCP, UDP, IP, HTTP/HTTPS |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                    (Gradio Chatbot UI)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     QUERY PROCESSING                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  User Input  │───▶│  Keyword    │───▶│  Search     │     │
│  │  "What is    │    │  Matching   │    │  Results    │     │
│  │   Docker?"  │    │  Algorithm  │    │  (Top 3)    │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                                │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT BUILDING                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Retrieved Chunks:                                   │   │
│  │  1. "Docker is a containerization platform..."       │   │
│  │  2. "Docker allows applications to run..."          │   │
│  │  3. "Containers package applications..."            │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prompt:                                             │   │
│  │  "Based on the following information, answer..."     │   │
│  │  "Information: [chunks]"                             │   │
│  │  "Question: What is Docker?"                         │   │
│  │  "Answer:"                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  HuggingFace Inference API                           │   │
│  │  Model: sshleifer/tiny-gpt2                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSE                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Answer: "Docker is a containerization platform..."  │   │
│  │                                                      │   │
│  │  Sources:                                            │   │
│  │  - docker.txt (Relevance: 100%)                      │   │
│  │  - docker.txt (Relevance: 85%)                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How RAG Works

### Step-by-Step Process

#### Phase 1: Indexing (Offline — `build_index.py`)

```
Documents ──▶ Chunking ──▶ Embedding ──▶ Vector Store
```

1. **Load Documents**: Read `.txt` files from `data/` folder
2. **Chunking**: Split each document into overlapping chunks (50 words, 10 word overlap)
3. **Embedding**: Convert each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`
4. **Storage**: Save embeddings (`.npy`) and chunk metadata (`.json`)

#### Phase 2: Retrieval (Online — `app.py`)

```
Query ──▶ Search ──▶ Top-K Results ──▶ Context ──▶ LLM ──▶ Answer
```

1. **User Query**: "What is Docker?"
2. **Keyword Search**: Find chunks with overlapping words
3. **Ranking**: Sort by relevance score
4. **Context Building**: Combine top-3 chunks
5. **LLM Generation**: Send context + question to tiny-gpt2
6. **Response**: Return answer with source citations

---

## 4. File Structure

```
personal-rag/
├── app.py                  # Main Gradio web application
├── build_index.py          # Script to create vector store
├── retrieve.py             # CLI semantic search tool
├── rag.py                  # CLI RAG assistant (local LLM)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── data/                   # Source documents
│   ├── aws.txt             # AWS cloud services
│   ├── docker.txt          # Docker containerization
│   ├── linux.txt           # Linux commands
│   └── networking.txt      # Network protocols
│
├── vector_store/           # Pre-computed index
│   ├── embeddings.npy      # 8 vectors (384 dimensions each)
│   └── chunks.json         # 8 chunks with metadata
│
├── .streamlit/             # Streamlit config (deprecated)
│   └── config.toml
│
└── .gradio/                # Gradio cache
    └── certificate.pem
```

---

## 5. Data Pipeline

### 5.1 Document Loading (`build_index.py:12-40`)

```python
def load_documents():
    documents = []
    data_folder = "data"
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append({
                    "filename": filename,
                    "text": text
                })
    return documents
```

**Input**: 4 text files
**Output**: List of `{"filename": "aws.txt", "text": "AWS stands for..."}`

### 5.2 Chunking (`build_index.py:47-67`)

```python
def create_chunks(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks
```

**Parameters**:
- `chunk_size = 50` words per chunk
- `overlap = 10` words overlap between chunks

**Example**:
```
Original: "Docker is a containerization platform. Docker allows..."
Chunk 1:  "Docker is a containerization platform. Docker allows..." (50 words)
Chunk 2:  "...platform. Docker allows applications to run..." (starts 40 words in)
```

### 5.3 Embedding (`build_index.py:104-120`)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(chunk_texts)
```

**Model**: `all-MiniLM-L6-v2`
- Output: 384-dimensional vectors
- Normalizes embeddings to unit vectors
- Optimized for semantic similarity

### 5.4 Storage

**Embeddings** (`vector_store/embeddings.npy`):
```python
# Shape: (8, 384)
# 8 chunks, 384 dimensions each
np.save("vector_store/embeddings.npy", embeddings)
```

**Chunks** (`vector_store/chunks.json`):
```json
[
    {
        "filename": "aws.txt",
        "text": "AWS stands for Amazon Web Services..."
    },
    {
        "filename": "docker.txt",
        "text": "Docker is a containerization platform..."
    }
    // ... 8 chunks total
]
```

---

## 6. Search Algorithm

### Current Implementation: Keyword Search (`app.py:52-96`)

The app uses **keyword overlap search** (not semantic search) to fit in Render's free tier:

```python
def search(query):
    # 1. Tokenize query
    query_words = set(query.lower().split())
    
    # 2. Score each chunk
    scores = []
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        chunk_words = set(text.split())
        overlap = len(query_words.intersection(chunk_words))
        scores.append(overlap)
    
    # 3. Normalize scores
    scores = np.array(scores, dtype=float)
    if scores.max() == 0:
        return []
    scores = scores / scores.max()
    
    # 4. Get top-K results
    top_indices = scores.argsort()[::-1][:TOP_K]
    
    # 5. Filter by threshold
    results = []
    for idx in top_indices:
        if scores[idx] < 0.1:
            continue
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "chunk": chunks[idx]
        })
    
    return results
```

### Search Example

**Query**: "What is Docker?"

| Chunk | Text | Overlap Words | Score |
|-------|------|---------------|-------|
| 0 | "AWS stands for..." | {} | 0.0 |
| 1 | "service. AWS Lambda..." | {} | 0.0 |
| 2 | "Docker is a containerization platform..." | {docker, is, a, containerization, platform} | 1.0 |
| 3 | "Dockerfiles are used to create..." | {dockerfiles, are, used, to, create, docker, images} | 0.85 |
| 4 | "Linux is an open-source..." | {} | 0.0 |
| 5 | "command displays..." | {} | 0.0 |
| 6 | "DNS translates..." | {} | 0.0 |
| 7 | "subnet divides..." | {} | 0.0 |

**Results**: Chunks 2, 3 returned (score > 0.1 threshold)

### Alternative: Semantic Search (`retrieve.py`)

The `retrieve.py` file uses **cosine similarity** with the embedding model:

```python
def retrieve_chunks(query_embedding, embeddings, chunks, top_k=3, threshold=0.30):
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings_norm)[0]
    
    # Sort and filter
    top_indices = similarities.argsort()[::-1]
    
    results = []
    for index in top_indices:
        score = similarities[index]
        if score < threshold:
            continue
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "chunk": chunks[index]
        })
        if len(results) >= top_k:
            break
    
    return results
```

**Note**: `retrieve.py` and `rag.py` are CLI tools, not used by the web app. The web app (`app.py`) uses keyword search to avoid loading the sentence-transformers model (saves memory).

---

## 7. LLM Integration

### HuggingFace Inference API (`app.py:130-149`)

```python
from huggingface_hub import InferenceClient

hf_client = InferenceClient()

# Generate answer
generated_text = hf_client.text_generation(
    prompt,
    model="sshleifer/tiny-gpt2",
    max_new_tokens=100,
    temperature=0.7,
)
```

### Prompt Template

```
Based on the following information, answer the question.

Information:
[Docker is a containerization platform. Docker allows applications...]
[Docker allows applications to run inside containers.]
[Containers package applications together with their dependencies.]

Question: What is Docker?
Answer:
```

### Response Processing

```python
# Find answer after "Answer:" marker
answer_start = generated_text.find("Answer:") + len("Answer:")
answer = generated_text[answer_start:].strip().split("\n")[0]
```

### Fallback

If HuggingFace API fails, returns first chunk excerpt:
```python
except Exception:
    answer = context_parts[0][:300] + "..."
```

---

## 8. User Interface

### Gradio Chatbot (`app.py:461-543`)

**Components**:
1. **Header**: Logo + title + subtitle
2. **Stats Bar**: Chunks indexed, model name, results count
3. **Example Chips**: 8 clickable question buttons
4. **Chat Interface**: Message bubbles + input box
5. **Footer**: Technology credits

### UI Flow

```
┌─────────────────────────────────────────────────────┐
│  🌐 RAG Assistant                                   │
│  Ask questions about your documents                 │
├─────────────────────────────────────────────────────┤
│  📄 8 chunks  │  🤖 tiny-gpt2  │  🔍 3 results    │
├─────────────────────────────────────────────────────┤
│  SUGGESTED QUESTIONS                                │
│  [What is Docker?] [What is an EC2 instance?] ...  │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐   │
│  │ User: What is Docker?                        │   │
│  ├─────────────────────────────────────────────┤   │
│  │ Bot: Docker is a containerization platform...│   │
│  │                                              │   │
│  │ Sources:                                     │   │
│  │ - docker.txt (Relevance: 100%)              │   │
│  └─────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  [Type your question here...]           [Send]     │
└─────────────────────────────────────────────────────┘
```

### Example Questions

| Question | Expected Topic |
|----------|----------------|
| "What is Docker?" | Docker, containers |
| "What is an EC2 instance?" | AWS, virtual servers |
| "How do I connect to a Linux server?" | SSH, Linux |
| "What is DNS?" | DNS, domain names |
| "What is HTTPS?" | HTTPS, encryption |
| "What is a virtual machine?" | VMs, virtualization |
| "How does cloud computing work?" | Cloud computing |
| "What is an API?" | APIs |

---

## 9. Deployment

### Render Free Tier

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3.14 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app.py` |
| **Instance Type** | Free (512MB RAM) |

### Memory Optimization

The app uses **keyword search** instead of **semantic search** to fit in 512MB:

| Component | Semantic Search | Keyword Search |
|-----------|----------------|----------------|
| sentence-transformers | ~90MB | Not loaded |
| torch | ~400MB | Not loaded |
| HuggingFace API | ~10MB | ~10MB |
| **Total** | ~500MB+ | ~50MB |

### Environment Variables

None required — all configuration is hardcoded for simplicity.

---

## 10. API Flow

### Request/Response Cycle

```
1. User sends: "What is Docker?"
   
2. App processes:
   ├── search("What is Docker?")
   │   ├── Tokenize: {"what", "is", "docker"}
   │   ├── Score chunks: [0, 0, 1.0, 0.85, 0, 0, 0, 0]
   │   └── Return top-3: [chunk_2, chunk_3]
   │
   ├── Build context:
   │   "Docker is a containerization platform..."
   │   "Docker allows applications to run inside..."
   │
   ├── Generate prompt:
   │   "Based on the following information..."
   │   "Question: What is Docker?"
   │   "Answer:"
   │
   └── Call HuggingFace API:
       model: sshleifer/tiny-gpt2
       max_new_tokens: 100
       temperature: 0.7
   
3. App returns:
   "Docker is a containerization platform that allows 
    applications to run inside containers.
    
    Sources:
    - docker.txt (Relevance: 100%)
    - docker.txt (Relevance: 85%)"
```

### Error Handling

```python
try:
    # Try HuggingFace API
    generated_text = hf_client.text_generation(...)
except Exception:
    # Fallback: return first chunk excerpt
    answer = context_parts[0][:300] + "..."
```

---

## 11. Configuration

### Hardcoded Settings

| Setting | Value | Location |
|---------|-------|----------|
| Embeddings Path | `vector_store/embeddings.npy` | `app.py:12` |
| Chunks Path | `vector_store/chunks.json` | `app.py:13` |
| LLM Model | `sshleifer/tiny-gpt2` | `app.py:15` |
| Top K Results | `3` | `app.py:17` |
| Server Host | `0.0.0.0` | `app.py:556` |
| Server Port | `7860` | `app.py:557` |

### Chunking Settings (build_index.py)

| Setting | Value |
|---------|-------|
| Chunk Size | 50 words |
| Overlap | 10 words |
| Embedding Model | `all-MiniLM-L6-v2` |

---

## 12. Limitations

### Current Limitations

| Issue | Description |
|-------|-------------|
| **Keyword Search** | Not semantic — misses synonyms and context |
| **Tiny LLM** | `tiny-gpt2` produces low-quality answers |
| **No Streaming** | Response appears all at once |
| **No Auth** | Anyone can use the app |
| **No Upload** | Can't add new documents via UI |
| **No Logging** | No request/error tracking |
| **Rate Limits** | HuggingFace API has usage limits |
| **Static Data** | Must rebuild index to add documents |

### Quality Issues

1. **Search**: "How do I connect to a Linux server?" may not match "SSH is commonly used to remotely connect" because keywords don't overlap
2. **Generation**: `tiny-gpt2` often produces incomplete or incorrect answers
3. **Chunking**: Fixed 50-word chunks may split sentences awkwardly

### Potential Improvements

| Area | Improvement |
|------|-------------|
| Search | Use TF-IDF or BM25 for better retrieval |
| LLM | Use Mistral-7B or GPT-3.5-turbo via API |
| Chunking | Use semantic chunking or sentence-based splitting |
| UI | Add streaming responses, document upload |
| Backend | Add caching, rate limiting, error tracking |

---

## Appendix: Code Snippets

### Loading Vector Store

```python
embeddings = np.load("vector_store/embeddings.npy")  # Shape: (8, 384)

with open("vector_store/chunks.json", "r") as file:
    chunks = json.load(file)  # List of 8 dicts
```

### Keyword Search

```python
query_words = set("what is docker".lower().split())
# {'what', 'is', 'docker'}

chunk_words = set("docker is a containerization platform".lower().split())
# {'docker', 'is', 'a', 'containerization', 'platform'}

overlap = len(query_words.intersection(chunk_words))
# 3 (what, is, docker)
```

### HuggingFace API Call

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

response = client.text_generation(
    "Based on the following information, answer the question.\n\n"
    "Information:\nDocker is a containerization platform.\n\n"
    "Question: What is Docker?\n"
    "Answer:",
    model="sshleifer/tiny-gpt2",
    max_new_tokens=100,
    temperature=0.7,
)
```

---

*Document generated for Personal RAG Assistant project.*
*Last updated: September 3, 2026*
