import requests
import pandas as pd
from bs4 import BeautifulSoup

# Define the company name and the target URL
company_name = "UBS"
url = "https://www.knowesg.com/company-esg-ratings?company-esg-ratings%5Bquery%5D=ubs"

# Headers to mimic a real browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com",
    "Connection": "keep-alive"
}

# Make the request
response = requests.get(url, headers=headers)

# Check the status code
if response.status_code == 200:
    print("✅ Successfully accessed the page")
    soup = BeautifulSoup(response.text, "html.parser")  # Parse HTML

    # Check the HTML structure
    print(soup.prettify()[:2000])  # Print first 2000 characters of HTML

    # Modify this selector to extract ESG Score correctly
    esg_element = soup.find("div", class_="esg-score")  # Update class if necessary
    
    if esg_element:
        esg_score = esg_element.text.strip()
    else:
        esg_score = "Not Found"

    # Save data in a CSV file
    df = pd.DataFrame([[company_name, esg_score]], columns=["Company", "ESG Score"])
    df.to_csv("ubs_esg_ratings.csv", index=False)

    print("✅ ESG Data saved as ubs_esg_ratings.csv")

else:
    print(f"❌ Failed to access the page: {response.status_code}")
    print("Check if the website blocks bots or requires login.")
