from sentence_transformers import SentenceTransformer
import json
import numpy as np


# --------------------------------------------------
# 1. Load saved chunks and embeddings
# --------------------------------------------------

input_path = "data/documents/amazon2023_embeddings.json"

with open(input_path, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

print("Number of chunks loaded:", len(chunk_data))


# --------------------------------------------------
# 2. Load the embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Ask the user for a question
# --------------------------------------------------

question = input("\nEnter your question: ")

print("\nQuestion:", question)


# --------------------------------------------------
# 4. Convert the question into an embedding
# --------------------------------------------------

question_embedding = model.encode(question)

print("Question embedding dimensions:", len(question_embedding))


# --------------------------------------------------
# 5. Compare question with every chunk
# --------------------------------------------------

results = []

for record in chunk_data:

    chunk_embedding = np.array(record["embedding"])

    similarity = np.dot(question_embedding, chunk_embedding) / (
        np.linalg.norm(question_embedding)
        * np.linalg.norm(chunk_embedding)
    )

    results.append({
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "similarity": float(similarity)
    })


# --------------------------------------------------
# 6. Sort results by similarity
# --------------------------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# --------------------------------------------------
# 7. Display top 5 results
# --------------------------------------------------

print("\n==============================")
print("TOP 5 RESULTS")
print("==============================")

for result in results[:5]:

    print("\n--------------------------------")
    print("Chunk ID:", result["chunk_id"])
    print("Similarity:", round(result["similarity"], 4))
    print("--------------------------------")

    print(result["text"][:1000])