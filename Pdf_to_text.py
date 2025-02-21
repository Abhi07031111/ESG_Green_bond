import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Specify the path to the local PDF file
pdf_file_path = "ubs_esg_report.pdf"

# Extract text from the local PDF
pdf_text = extract_text_from_pdf(pdf_file_path)

# Save the extracted text to a .txt file
if pdf_text:
    with open("ubs_esg_report.txt", "w", encoding="utf-8") as text_file:
        text_file.write(pdf_text)
    print("✅ PDF text saved successfully to 'ubs_esg_report.txt'.")
else:
    print("⚠️ No text found in the PDF to save.")
