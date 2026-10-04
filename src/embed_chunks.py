from sentence_transformers import SentenceTransformer
import json


# --------------------------------------------------
# 1. Read the extracted text
# --------------------------------------------------

input_path = "data/documents/amazon2023.txt"

with open(input_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))


# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 3. Load the embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embeddings generated.")


# --------------------------------------------------
# 5. Create records containing chunk + embedding
# --------------------------------------------------

chunk_data = []

for i, chunk in enumerate(chunks):

    record = {
        "chunk_id": i,
        "text": chunk,
        "embedding": embeddings[i].tolist()
    }

    chunk_data.append(record)


# --------------------------------------------------
# 6. Save chunks and embeddings
# --------------------------------------------------

output_path = "data/documents/amazon2023_embeddings.json"

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        chunk_data,
        file
    )


print("\nEmbedding process completed.")
print("Saved to:", output_path)
print("Total records:", len(chunk_data))
print("Embedding dimensions:", len(chunk_data[0]["embedding"]))