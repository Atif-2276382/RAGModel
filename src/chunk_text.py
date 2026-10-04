# Read the extracted text
input_path = "data/documents/amazon2023.txt"

with open(input_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))


# Chunk configuration
chunk_size = 1000
overlap = 200

# Store the chunks
chunks = []

# Starting position
start = 0

# Create chunks
while start < len(text):

    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("Number of chunks:", len(chunks))


# Display the first three chunks
for i, chunk in enumerate(chunks[:3], start=1):

    print("\n==============================")
    print("Chunk", i)
    print("==============================")

    print(chunk)