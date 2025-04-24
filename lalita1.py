import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl import load_workbook

# Helper functions to normalize ESG scores from different agencies
def normalize_sustainalytics(score):
    """Normalize Sustainalytics score to a 1-10 scale."""
    if pd.isna(score):
        return None
    return round(10 - (max(0, min(100, score)) / 100) * 10, 2)

def normalize_lseg(score):
    """Normalize LSEG score to a 1-10 scale."""
    if pd.isna(score):
        return None
    return round((max(0, min(100, score)) / 100) * 10, 2)

def normalize_msci(rating):
    """Normalize MSCI rating to a 1-10 scale based on predefined categories."""
    if not isinstance(rating, str):
        return None
    rating_map = {
        'CCC': 2.0, 'B': 3.5, 'BB': 5.0, 'BBB': 6.0, 'A': 7.0, 'AA': 8.5, 'AAA': 10.0
    }
    return rating_map.get(rating.strip().upper(), None)

# Load and clean ESG ratings from CSV file
def load_esg_data(csv_file, company_name):
    """Load and normalize ESG ratings for a company."""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()
    df['RATING_AGENCY'] = df['RATING_AGENCY'].str.upper()
    df['ESG_RATING'] = df['ESG_RATING'].astype(str).str.strip()

    # Extract raw scores for each agency
    sustainalytics_raw = df[df['RATING_AGENCY'] == 'SUSTAINALYTICS']['ESG_RATING']
    lseg_raw = df[df['RATING_AGENCY'] == 'LSEG']['ESG_RATING']
    msci_raw = df[df['RATING_AGENCY'] == 'MSCI']['ESG_RATING']

    # Normalize raw scores
    esg_data = {
        'Company Name': company_name,
        'Sustainalytics_RAW': sustainalytics_raw.iloc[0] if not sustainalytics_raw.empty else None,
        'LSEG_RAW': lseg_raw.iloc[0] if not lseg_raw.empty else None,
        'MSCI_RAW': msci_raw.iloc[0] if not msci_raw.empty else None,
        'Sustainalytics': normalize_sustainalytics(pd.to_numeric(sustainalytics_raw.iloc[0], errors='coerce')) if not sustainalytics_raw.empty else None,
        'LSEG': normalize_lseg(pd.to_numeric(lseg_raw.iloc[0], errors='coerce')) if not lseg_raw.empty else None,
        'MSCI': normalize_msci(msci_raw.iloc[0]) if not msci_raw.empty else None
    }

    # Calculate average ESG score
    ratings = [score for score in [esg_data['Sustainalytics'], esg_data['LSEG'], esg_data['MSCI']] if score is not None]
    esg_data['Average ESG (1-10)'] = round(np.mean(ratings), 2) if ratings else np.nan

    return esg_data

# Load sentiment score from a text file
def load_sentiment(txt_file):
    """Load sentiment score from a text file."""
    try:
        with open(txt_file, 'r') as file:
            sentiment_score = float(file.readline().split(':')[1].strip())
    except (IndexError, ValueError):
        sentiment_score = 0.0
    return sentiment_score

# Main Recommendation Engine
def recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, output_file,
                              lseg_weight=0.4, sustainalytics_weight=0.3, msci_weight=0.3, industry_benchmark=75):
    """Generate green bond recommendations using machine learning."""
    
    # Calculate overall ESG score with sentiment impact
    msci_boost = 10 if esg_data['MSCI'] and esg_data['MSCI'] >= 8.5 else 5
    overall_score = (
        (esg_data['LSEG'] or 0) * lseg_weight +
        (esg_data['Sustainalytics'] or 0) * sustainalytics_weight +
        msci_boost * msci_weight
    )
    sentiment_impact = sentiment_score * 15
    overall_score += sentiment_impact

    # Add new features to bond data
    bond_data['Overall Score'] = overall_score
    bond_data['Sentiment Impact'] = sentiment_impact
    bond_data['Quality Score'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 0.5
    bond_data['SECTOR_SCORE'] = pd.factorize(bond_data['ISSUER_SECTOR'])[0]
    bond_data['REVIEWER_SCORE'] = pd.factorize(bond_data['EXTERNAL_REVIEWER'])[0]

    # Train a machine learning model
    features = ['Overall Score', 'Sentiment Impact', 'Quality Score', 'SECTOR_SCORE', 'REVIEWER_SCORE']
    X = bond_data[features]
    y = bond_data['Recommended Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'Model Performance: MSE = {mean_squared_error(y_test, y_pred):.2f}, R2 = {r2_score(y_test, y_pred):.2f}')

    # Generate bond recommendations
    bond_recommendations = []
    for _, row in bond_data.iterrows():
        recommendation = 'Hold'
        if row['BOND_TYPE'].lower() == 'green':
            if overall_score < 50:
                recommendation = 'Strong Buy'
            elif overall_score < 70:
                recommendation = 'Buy'
            elif overall_score < industry_benchmark:
                recommendation = 'Strategic Buy'

        potential_esg_score = min(esg_data['Average ESG (1-10)'] + 0.1, 10)

        bond_recommendations.append({
            'Company Name': esg_data['Company Name'],
            'Issuer Name': row['ISSUER_NAME'].title(),
            'Recommendation': recommendation if recommendation != 'Hold' else 'Strategic Buy',
            'Potential ESG Score (1-10)': round(potential_esg_score, 2),
            'Comments': 'High Priority Purchase' if recommendation in ['Strong Buy', 'Buy'] else 'Strategic Investment',
            'Score': round(overall_score, 2),
            'Sentiment Impact': round(sentiment_impact, 2)
        })

    # Save top recommendations to an Excel file
    top_recommendations = sorted(bond_recommendations, key=lambda x: x['Score'], reverse=True)[:5]
    recommendations_df = pd.DataFrame(top_recommendations)

    # Check if the output file exists, otherwise create it
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
            existing_recommendations_df = pd.read_excel(output_file, sheet_name='Top Recommendations')
            updated_recommendations_df = pd.concat([existing_recommendations_df, recommendations_df], ignore_index=True)
            updated_recommendations_df.to_excel(writer, sheet_name='Top Recommendations', index=False, header=False)

            if 'Company ESG Ratings' not in writer.book.sheetnames:
                esg_df = pd.DataFrame([{
                    'Company Name': esg_data['Company Name'],
                    'Sustainalytics ESG Score': esg_data.get('Sustainalytics_RAW'),
                    'LSEG ESG Score': esg_data.get('LSEG_RAW'),
                    'MSCI ESG Rating': esg_data.get('MSCI_RAW'),
                    'Average ESG (1-10)': esg_data.get('Average ESG (1-10)'),
                    'Comment': 'Excellent' if esg_data.get('Average ESG (1-10)') >= 8 else 'Good' if esg_data.get('Average ESG (1-10)') >= 6 else 'Average' if esg_data.get('Average ESG (1-10)') >= 4 else 'Poor',
                    'Recommendation': 'Maintain or Increase Green Investments' if esg_data.get('Average ESG (1-10)') >= 8 else 'Consider Strategic Investments' if esg_data.get('Average ESG (1-10)') >= 6 else 'Focus on ESG Improvement'
                }])
                esg_df.to_excel(writer, sheet_name='Company ESG Ratings', index=False, header=False)

    else:
        recommendations_df.to_excel(output_file, sheet_name='Top Recommendations', index=False, header=True)

    print(f'Recommendations saved to {output_file}')


# === RUN SECTION ===
company_name = 'Allianz SE'
esg_data = load_esg_data('Esg_data.csv', company_name)
sentiment_score = load_sentiment('average_sentiment.txt')
bond_data = pd.read_csv('bond_data.csv')

# Clean bond data columns
bond_data.columns = bond_data.columns.str.strip().str.replace(' ', '_').str.upper().str.replace('__', '_')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = pd.to_numeric(
    bond_data['AMOUNT_ISSUED_(USD_BN.)'].astype(str).str.replace(',', '').str.replace('$', ''),
    errors='coerce'
)
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'].fillna(0)
bond_data['Recommended Quantity'] = np.round(bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 10)
bond_data['Recommended Quantity'] = bond_data['Recommended Quantity'].fillna(0)

# Execute recommendation engine
recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, 'ml_based_recommendations13.xlsx')
