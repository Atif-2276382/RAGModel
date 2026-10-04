from pypdf import PdfReader

# Path of the PDF file
pdf_path = "data/documents/amazon2023.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

# Store all extracted text
all_text = ""

# Process every page
for page_number, page in enumerate(reader.pages, start=1):

    print(f"Extracting page {page_number}...")

    text = page.extract_text()

    if text:
        all_text += text + "\n"

# Save the extracted text
output_path = "data/documents/amazon2023.txt"

with open(output_path, "w", encoding="utf-8") as file:
    file.write(all_text)

print("\nExtraction completed.")
print("Text saved to:", output_path)
print("Total characters:", len(all_text))