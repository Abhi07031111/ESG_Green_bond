import pandas as pd

def load_esg_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # Strip extra spaces from column names
    df["ESG Rating"] = pd.to_numeric(df["ESG Rating"], errors='coerce')  # Convert ESG ratings to numeric  # Strip extra spaces from column names
    print(df.columns)  # Debugging: Print column names to check
    return {
        "Company Name": "UBS",  # Hardcoded company name
        "Sustainalytics": df.loc[df["Rating Agency"] == "Sustainalytics", "ESG Rating"].values[0],
        "LSEG": df.loc[df["Rating Agency"] == "LSEG", "ESG Rating"].values[0],
        "MSCI": df.loc[df["Rating Agency"] == "MSCI", "ESG Rating"].values[0]
    }

def load_sentiment(txt_file):
    with open(txt_file, "r") as file:
        sentiment_score = float(file.readline().strip().split(":")[1].strip())
    return sentiment_score

def recommend_green_bonds(esg_data, sentiment_score, bond_data, output_file):
    # Green Bond Recommendation Criteria
    bond_recommendations = []
    for _, row in bond_data.iterrows():
        if row["Bond type"].lower() == "green":
            score = (esg_data["LSEG"] * 0.5 + esg_data["Sustainalytics"] * 0.3 +
                     (100 if esg_data["MSCI"] in ["AA", "AAA"] else 50))
            score += sentiment_score * 10
            
            if score > 75:
                recommendation = "Strong Buy"
            elif score > 50:
                recommendation = "Buy"
            else:
                recommendation = "Hold"
            
            bond_recommendations.append({
                "Company Name": esg_data["Company Name"],
                "Issuer Name": row["Issuer Name"],
                "Recommendation": recommendation,
                "Score": round(score, 2),
                "Sustainalytics": esg_data["Sustainalytics"],
                "LSEG": esg_data["LSEG"],
                "MSCI": esg_data["MSCI"]
            })  # Appending bond recommendation
    
    # Creating a CSV output with ESG scores followed by recommendations
    with open(output_file, "w") as f:
        f.write("Company Name,Sustainalytics,LSEG,MSCI\n")
        f.write(f"{esg_data['Company Name']},{esg_data['Sustainalytics']},{esg_data['LSEG']},{esg_data['MSCI']}\n")
        f.write("\n")
        
        if bond_recommendations:
            # Limit recommendations to top 5 by score
            bond_recommendations = sorted(bond_recommendations, key=lambda x: x["Score"], reverse=True)[:5]
            recommendations_df = pd.DataFrame(bond_recommendations)
            recommendations_df.to_csv(f, index=False, mode='a')
        else:
            f.write("No recommendations available.\n")
    
    print(f"Recommendations saved to {output_file}")

# Load data from files
esg_data = load_esg_data("Esg_data.csv")
sentiment_score = load_sentiment("average_sentiment.txt")
bond_data = pd.read_csv("bond_data.csv")

# Get Recommendations and save to CSV
recommend_green_bonds(esg_data, sentiment_score, bond_data, "recommendations.csv")
