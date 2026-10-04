from openai import OpenAI

print("Starting test...")

client = OpenAI()

print("Client created.")
print("Calling OpenAI...")

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Say hello in one sentence."
)

print("Response received.")
print(response.output_text)