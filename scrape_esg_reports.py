import requests
import pdfplumber

UBS_ESG_PDF_URL = "https://www.ubs.com/content/dam/assets/cc/investor-relations/sustainability-report/2023/sustainability-report-2023.pdf"

def download_pdf(url):
    """Downloads the PDF file from the given URL and returns the file content."""
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("ubs_esg_report.pdf", "wb") as file:
            file.write(response.content)
        print("✅ PDF downloaded successfully.")
    else:
        print(f"⚠️ Failed to download PDF. Status code: {response.status_code}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Download the PDF from UBS ESG report URL
download_pdf(UBS_ESG_PDF_URL)

# Extract text from the downloaded PDF
pdf_text = extract_text_from_pdf("ubs_esg_report.pdf")

# Save the extracted text to a .txt file
if pdf_text:
    with open("ubs_esg_report.txt", "w", encoding="utf-8") as text_file:
        text_file.write(pdf_text)
    print("✅ PDF text saved successfully to 'ubs_esg_report.txt'.")
else:
    print("⚠️ No text found in the PDF to save.")
