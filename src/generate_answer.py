from sentence_transformers import SentenceTransformer
import json
import numpy as np
from openai import OpenAI


# ---------------------------------------
# 1. Load stored document chunks
# ---------------------------------------

input_path = "data/documents/amazon2023_embeddings.json"

with open(input_path, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

print("Loaded chunks:", len(chunk_data))


# ---------------------------------------
# 2. Load embedding model
# ---------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------
# 3. Get user's question
# ---------------------------------------

question = input("\nEnter your question: ")

print("\nQuestion:", question)


# ---------------------------------------
# 4. Convert question into embedding
# ---------------------------------------

question_embedding = model.encode(question)


# ---------------------------------------
# 5. Search for similar chunks
# ---------------------------------------

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


# ---------------------------------------
# 6. Sort by similarity
# ---------------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# ---------------------------------------
# 7. Select top 5 chunks
# ---------------------------------------

top_results = results[:5]


# ---------------------------------------
# 8. Build context
# ---------------------------------------

context = ""

for result in top_results:

    context += (
        f"\n[Chunk {result['chunk_id']}]\n"
        f"{result['text']}\n"
    )


# ---------------------------------------
# 9. Create RAG prompt
# ---------------------------------------

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


# ---------------------------------------
# 10. Create OpenAI client
# ---------------------------------------

client = OpenAI()


# ---------------------------------------
# 11. Send question + context to LLM
# ---------------------------------------

response = client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)


# ---------------------------------------
# 12. Print generated answer
# ---------------------------------------

print("\n========================================")
print("GENERATED ANSWER")
print("========================================")

print(response.output_text)