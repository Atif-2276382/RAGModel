# RAGModel

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that combines document retrieval with local LLM inference to generate grounded, contextual answers.

## 🎯 Overview

This project implements a complete RAG system that:
1. **Indexes** documents (PDFs) by extracting text, chunking, and creating embeddings
2. **Retrieves** relevant context using semantic similarity search
3. **Generates** accurate answers using Llama 3.2 running locally via Ollama

**Use Case**: Question answering over large documents (demonstrated with Amazon 2023 Sustainability Report)

---

## 🔄 Pipeline Architecture

### Phase 1: Indexing / Ingestion
```
PDF → Extract Text → Chunking → Embeddings → Storage
```

| Stage | Tool | Details |
|-------|------|---------|
| **Extract Text** | pypdf | Parse PDF documents into raw text |
| **Chunking** | Custom | Split text into 1,000-char chunks with 200-char overlap |
| **Embeddings** | all-MiniLM-L6-v2 | Generate 384-dimensional dense vectors |
| **Storage** | JSON | Store chunks + vectors for fast retrieval |

### Phase 2: Query / Retrieval / Generation
```
Question → Query Embedding → Similarity Search → Top-K Context → LLM → Answer
```

| Stage | Tool | Details |
|-------|------|---------|
| **Query Embedding** | all-MiniLM-L6-v2 | Embed user question in same 384-D space |
| **Similarity Search** | NumPy (cosine) | Rank all chunks by relevance |
| **Context Selection** | Top-5 chunks | Combine most relevant evidence |
| **Generation** | Llama 3.2 (Ollama) | Generate grounded response |

---

## 📂 Project Structure

```
RAGModel/
├── src/                          # Core pipeline modules
│   ├── extract_text.py          # PDF → text extraction
│   ├── chunk_text.py            # Text → semantic chunks
│   ├── embed_chunks.py          # Chunks → embeddings
│   ├── search.py                # Similarity search
│   ├── build_context.py         # Build retrieval context
│   ├── generate_answer.py       # OpenAI-based generation
│   ├── generate_answer_ollama.py # Ollama-based generation
│   ├── load_metadata.py         # Metadata utilities
│   ├── test_embedding.py        # Embedding verification
│   ├── test_similarity.py       # Search verification
│   ├── test_openai.py           # OpenAI integration test
│   └── test_ollama.py           # Ollama integration test
├── data/                         # Document storage
├── rag_pipeline_interactive.html # Visual pipeline documentation
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Ollama** (for local LLM inference): [Download](https://ollama.ai)
- **Models**:
  - `ollama pull llama2` or `ollama pull llama3.2`
  - Embeddings model: `all-MiniLM-L6-v2` (auto-downloaded)

### Installation

```bash
git clone https://github.com/Atif-2276382/RAGModel.git
cd RAGModel

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
ollama serve  # In a separate terminal
```

### Quick Start

#### 1. Index a Document

```python
from src.extract_text import extract_text_from_pdf
from src.chunk_text import chunk_text
from src.embed_chunks import embed_and_store_chunks

# Extract text from PDF
text = extract_text_from_pdf("path/to/document.pdf")

# Chunk the text
chunks = chunk_text(text, chunk_size=1000, overlap=200)

# Create embeddings and store
embed_and_store_chunks(chunks, output_file="embeddings.json")
```

#### 2. Query the Index

```python
from src.search import search_similar_chunks
from src.build_context import build_context
from src.generate_answer_ollama import generate_answer_ollama

# Search for relevant chunks
query = "What are the sustainability goals?"
results = search_similar_chunks(query, embeddings_file="embeddings.json", top_k=5)

# Build context from results
context = build_context(results)

# Generate answer with Llama
answer = generate_answer_ollama(query, context, model="llama2")
print(f"Q: {query}\nA: {answer}")
```

---

## 🛠️ Configuration

### Embedding Model

Default: `all-MiniLM-L6-v2` (384 dimensions)

To change, modify in `embed_chunks.py`:
```python
model = SentenceTransformer("your-model-name")
```

### Chunking Parameters

Adjust in `chunk_text.py`:
- `chunk_size`: Characters per chunk (default: 1000)
- `overlap`: Character overlap between chunks (default: 200)

### LLM Model

Switch models in `generate_answer_ollama.py`:
```python
generate_answer_ollama(query, context, model="llama3.2")
```

Supported: Any model available via Ollama (`llama2`, `llama3.2`, `mistral`, etc.)

---

## 📊 Example Usage

See `rag_pipeline_interactive.html` for an interactive visualization of the pipeline.

**Example Query**:
```
Q: What are Amazon's goals related to carbon-free energy?

Result:
- 563 total chunks indexed
- Top 5 relevant chunks retrieved via semantic search
- Context passed to Llama 3.2
- Grounded answer generated from retrieved evidence
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# Test embeddings
python src/test_embedding.py

# Test similarity search
python src/test_similarity.py

# Test Ollama integration
python src/test_ollama.py

# Test OpenAI integration (requires API key)
python src/test_openai.py
```

---

## 🔐 API Keys (Optional)

For OpenAI-based generation, set your API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

Then use `src/generate_answer.py` instead of Ollama.

---

## 📈 Performance Considerations

| Aspect | Details |
|--------|---------|
| **Embedding Time** | ~1-5 min for 50k+ chunks (one-time) |
| **Search Time** | <100ms for 1000+ chunks |
| **Generation Time** | 2-10 sec per query (Ollama, CPU) |
| **Storage** | ~500MB per 100k chunks + metadata |

### Optimization Tips

- Use GPU acceleration with Ollama for faster inference
- Increase `top_k` for better context but slower retrieval
- Experiment with chunk size/overlap for your domain
- Pre-compute embeddings for all documents

---

## 📝 Features

✅ **Document Ingestion**: PDF text extraction  
✅ **Semantic Search**: Dense vector similarity  
✅ **Local LLM**: Run Ollama models without API costs  
✅ **Context Building**: Automatic evidence compilation  
✅ **Flexible Backends**: OpenAI or Ollama  
✅ **Interactive Visualization**: HTML pipeline diagram  
✅ **Modular Design**: Easy to extend and customize  

---

## 🤝 Contributing

Contributions welcome! Please feel free to:
- Submit issues for bugs or feature requests
- Fork and create pull requests
- Improve documentation or tests

---

## 📄 License

This project is open source. Check the repository for license details.

---

## 📧 Support

For questions or issues, please open a GitHub issue in this repository.

---

## 🙏 Acknowledgments

- Built with [Sentence Transformers](https://www.sbert.net/)
- LLM inference via [Ollama](https://ollama.ai)
- PDF processing with [pypdf](https://github.com/py-pdf/pypdf)
- Similarity search with [NumPy](https://numpy.org/)

---

**Happy RAG-ing! 🚀**
