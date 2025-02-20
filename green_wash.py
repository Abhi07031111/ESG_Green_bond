import pandas as pd
import numpy as np

def load_esg_data():
    df = pd.read_csv('Esg_data.csv')
    df.columns = df.columns.str.strip()
    df["Company Name"] = "Company Name"  # Hardcoding the company name
    df["ESG Rating"] = pd.to_numeric(df["ESG Rating"], errors='coerce')
    return df

def load_sentiment():
    # Using the uploaded average_sentiment.txt file
    sentiment_scores = {}
    with open('average_sentiment.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Average Sentiment Score" in line:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    score = float(parts[1].strip())
                    sentiment_scores["Company Name"] = score  # Hardcoding the company name
    return sentiment_scores

def detect_greenwashing(esg_data, sentiment_scores, threshold=20):
    anomalies = []
    companies = esg_data["Company Name"].unique()
    
    for company in companies:
        company_data = esg_data[esg_data["Company Name"] == company]
        
        sustainalytics = company_data.loc[company_data["Rating Agency"] == "Sustainalytics", "ESG Rating"].values
        lseg = company_data.loc[company_data["Rating Agency"] == "LSEG", "ESG Rating"].values
        msci = company_data.loc[company_data["Rating Agency"] == "MSCI", "ESG Rating"].values
        
        # Calculate average ESG score if all ratings are available
        if len(sustainalytics) > 0 and len(lseg) > 0 and len(msci) > 0:
            avg_esg = np.mean([sustainalytics[0], lseg[0], msci[0]])
            
            # Get sentiment score for the company
            sentiment_score = sentiment_scores.get(company, 0)  # Default to 0 if not found
            
            # Calculate discrepancy
            discrepancy = avg_esg - (sentiment_score * 10)  # Scale sentiment score
            
            # Check if discrepancy exceeds the threshold
            if discrepancy > threshold:
                anomalies.append({
                    "Company Name": company,
                    "Average ESG Score": round(avg_esg, 2),
                    "Sentiment Score": round(sentiment_score * 10, 2),
                    "Discrepancy": round(discrepancy, 2)
                })
    
    return anomalies

def save_anomalies(anomalies, output_file):
    if anomalies:
        df = pd.DataFrame(anomalies)
        df.to_csv(output_file, index=False)
        print(f"Anomalies saved to {output_file}")
    else:
        print("No greenwashing anomalies detected.")

# Load data
esg_data = load_esg_data()
sentiment_scores = load_sentiment()

# Detect greenwashing anomalies
anomalies = detect_greenwashing(esg_data, sentiment_scores)

# Save anomalies to CSV
save_anomalies(anomalies, "greenwashing_anomalies.csv")
