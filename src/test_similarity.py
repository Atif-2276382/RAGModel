from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Three sentences
sentence1 = "Amazon is investing in renewable energy."
sentence2 = "Amazon is increasing its use of clean energy."
sentence3 = "The library has many books."

# Convert sentences into embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

# Calculate cosine similarity
similarity_12 = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)

similarity_13 = np.dot(embedding1, embedding3) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding3)
)

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Sentence 3:", sentence3)

print("\nSimilarity between Sentence 1 and Sentence 2:")
print(similarity_12)

print("\nSimilarity between Sentence 1 and Sentence 3:")
print(similarity_13)