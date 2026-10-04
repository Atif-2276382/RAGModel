from sentence_transformers import SentenceTransformer
from ollama import chat
import json
import numpy as np


# --------------------------------------------------
# 1. Load stored chunks and embeddings
# --------------------------------------------------

input_path = "data/documents/amazon2023_embeddings.json"

with open(input_path, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

print("Loaded chunks:", len(chunk_data))


# --------------------------------------------------
# 2. Load the embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# 3. Get user's question
# --------------------------------------------------

question = input("\nEnter your question: ")

print("\nQuestion:", question)


# --------------------------------------------------
# 4. Convert question into an embedding
# --------------------------------------------------

question_embedding = model.encode(question)

print(
    "Question embedding dimensions:",
    len(question_embedding)
)


# --------------------------------------------------
# 5. Calculate similarity with every chunk
# --------------------------------------------------

results = []

for record in chunk_data:

    chunk_embedding = np.array(record["embedding"])

    similarity = np.dot(
        question_embedding,
        chunk_embedding
    ) / (
        np.linalg.norm(question_embedding)
        * np.linalg.norm(chunk_embedding)
    )

    results.append({
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "similarity": float(similarity)
    })


# --------------------------------------------------
# 6. Sort chunks by similarity
# --------------------------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# --------------------------------------------------
# 7. Select Top 5 chunks
# --------------------------------------------------

top_results = results[:5]


# --------------------------------------------------
# 8. Build context for the LLM
# --------------------------------------------------

context = ""

for result in top_results:

    context += (
        f"\n[Chunk {result['chunk_id']}]\n"
        f"{result['text']}\n"
    )


# --------------------------------------------------
# 9. Display retrieved chunks
# --------------------------------------------------

print("\n========================================")
print("RETRIEVED CONTEXT")
print("========================================")

for result in top_results:

    print(
        f"\nChunk {result['chunk_id']} "
        f"→ similarity = {result['similarity']:.4f}"
    )


# --------------------------------------------------
# 10. Create the RAG prompt
# --------------------------------------------------

prompt = f"""
You are answering questions about Amazon's 2023 Sustainability Report.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""


# --------------------------------------------------
# 11. Send context + question to Ollama
# --------------------------------------------------

print("\n========================================")
print("GENERATING ANSWER WITH LLAMA 3.2")
print("========================================")

response = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 12. Display final answer
# --------------------------------------------------

answer = response["message"]["content"]

print("\n========================================")
print("FINAL ANSWER")
print("========================================")

print(answer)