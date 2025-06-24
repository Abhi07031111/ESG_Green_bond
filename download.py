import requests
from bs4 import BeautifulSoup
from weasyprint import HTML

def scrape_and_save_pdf(url, output_filename):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()

        html_content = str(soup)
        HTML(string=html_content, base_url=url).write_pdf(output_filename)
        print(f"PDF saved successfully as {output_filename}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
scrape_and_save_pdf("https://www.ecfr.gov/current/title-17/chapter-II/part-200", "output.pdf")
