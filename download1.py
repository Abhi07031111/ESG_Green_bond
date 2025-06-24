import requests
from bs4 import BeautifulSoup
import pdfkit

def scrape_and_save_pdf(url, output_filename):
    try:
        # Fetch the web page
        response = requests.get(url)
        response.raise_for_status()

        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Optional: Clean up script and style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Convert soup back to HTML string
        html_content = str(soup)

        # Convert HTML to PDF
        pdfkit.from_string(html_content, output_filename)
        print(f"PDF saved successfully as {output_filename}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
url = "https://example.com"
output_pdf = "output.pdf"
scrape_and_save_pdf(url, output_pdf)
