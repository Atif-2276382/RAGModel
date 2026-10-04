from pypdf import PdfReader

# Path of the PDF file
pdf_path = "data/documents/amazon2023.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

# Extract text from the first page
page = reader.pages[0]
text = page.extract_text()

print("\n--- First Page Text ---\n")
print(text)