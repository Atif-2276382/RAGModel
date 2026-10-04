from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
text = "Amazon is investing in renewable energy."

# Convert text into an embedding
embedding = model.encode(text)

print("Embedding type:", type(embedding))
print("Number of dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])