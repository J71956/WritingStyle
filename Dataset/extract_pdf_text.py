import os
from PyPDF2 import PdfReader

# Directory containing the PDF files
pdf_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(pdf_dir, "output.txt")

all_text = []

for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_text.append(f"--- {filename} ---\n{text}\n")
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print(f"Extracted text from all PDFs into {output_file}")
