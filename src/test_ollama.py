from ollama import chat

print("Calling Ollama...")

response = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Explain carbon-free energy in one sentence."
        }
    ]
)

print("\nAnswer:")
print(response["message"]["content"])