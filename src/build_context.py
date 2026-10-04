from sentence_transformers import SentenceTransformer
import json
import numpy as np


# --------------------------------------------------
# 1. Load chunks and embeddings
# --------------------------------------------------

input_path = "data/documents/amazon2023_embeddings.json"

with open(input_path, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

print("Loaded chunks:", len(chunk_data))


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Get user question
# --------------------------------------------------

question = input("\nEnter your question: ")

print("\nQuestion:", question)


# --------------------------------------------------
# 4. Create embedding for the question
# --------------------------------------------------

question_embedding = model.encode(question)


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
# 6. Sort by similarity
# --------------------------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# --------------------------------------------------
# 7. Select top 5 chunks
# --------------------------------------------------

top_results = results[:5]


# --------------------------------------------------
# 8. Build context
# --------------------------------------------------

context = ""

for result in top_results:

    context += (
        f"\n[Chunk {result['chunk_id']}]\n"
        f"{result['text']}\n"
    )


# --------------------------------------------------
# 9. Display retrieved context
# --------------------------------------------------

print("\n========================================")
print("RETRIEVED CONTEXT")
print("========================================")

print(context)


# --------------------------------------------------
# 10. Display similarity scores
# --------------------------------------------------

print("\n========================================")
print("RETRIEVAL SCORES")
print("========================================")

for result in top_results:

    print(
        f"Chunk {result['chunk_id']} "
        f"→ similarity = {result['similarity']:.4f}"
    )