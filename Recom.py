# Required Libraries
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl import Workbook, load_workbook

# Load ESG Data
def load_esg_data(csv_file, company_name):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()
    df['RATING_AGENCY'] = df['RATING_AGENCY'].str.upper()
    df['ESG_RATING'] = pd.to_numeric(df['ESG_RATING'], errors='coerce')
    return {
        'Company Name': company_name,
        'Sustainalytics': df.loc[df['RATING_AGENCY'] == 'SUSTAINALYTICS', 'ESG_RATING'].values[0] if len(df.loc[df['RATING_AGENCY'] == 'SUSTAINALYTICS', 'ESG_RATING'].values) > 0 else np.nan,
        'LSEG': df.loc[df['RATING_AGENCY'] == 'LSEG', 'ESG_RATING'].values[0] if len(df.loc[df['RATING_AGENCY'] == 'LSEG', 'ESG_RATING'].values) > 0 else np.nan,
        'MSCI': df.loc[df['RATING_AGENCY'] == 'MSCI', 'ESG_RATING'].values[0] if len(df.loc[df['RATING_AGENCY'] == 'MSCI', 'ESG_RATING'].values) > 0 else np.nan
    }

# Load Sentiment Data
def load_sentiment(txt_file):
    with open(txt_file, 'r') as file:
        sentiment_score = float(file.readline().strip().split(':')[1].strip())
    return sentiment_score

# Feature Engineering and ML Model
def recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, output_file, lseg_weight=0.4, sustainalytics_weight=0.3, msci_weight=0.3, industry_benchmark=75):
    overall_score = (
        esg_data['LSEG'] * lseg_weight +
        esg_data['Sustainalytics'] * sustainalytics_weight +
        (100 if esg_data['MSCI'] in ['AA', 'AAA'] else 50)
    )
    sentiment_impact = sentiment_score * 15
    overall_score += sentiment_impact

    # Prepare Data for ML
    bond_data['Overall Score'] = overall_score
    bond_data['Sentiment Impact'] = sentiment_impact
    bond_data['Quality Score'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 0.5
    bond_data['SECTOR_SCORE'] = pd.factorize(bond_data['ISSUER_SECTOR'])[0]
    bond_data['REVIEWER_SCORE'] = pd.factorize(bond_data['EXTERNAL_REVIEWER'])[0]
    X = bond_data[['Overall Score', 'Sentiment Impact', 'Quality Score', 'SECTOR_SCORE', 'REVIEWER_SCORE']]
    y = bond_data['Recommended Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Model Performance: MSE={mse:.2f}, R2={r2:.2f}')

    # Recommendation Logic
    bond_recommendations = []
    potential_esg_increase = 0.1
    for _, row in bond_data.iterrows():
        recommendation = 'Strategic Buy'
        if row['BOND_TYPE'].lower() == 'green':
            if overall_score < 50:
                recommendation = 'Strong Buy'
            elif overall_score < 70:
                recommendation = 'Buy'
            elif overall_score < industry_benchmark:
                recommendation = 'Strategic Buy'
        else:
            recommendation = 'Hold'
        bond_recommendations.append({
            'Company Name': esg_data['Company Name'],
            'Issuer Name': row['ISSUER_NAME'].title(),
            'Recommendation': recommendation.replace('Hold', 'Strategic Buy'),
            'Potential ESG Increase': round(potential_esg_increase, 2),
            'Comments': 'High Priority Purchase' if recommendation in ['Strong Buy', 'Buy'] else 'Strategic Investment',
            'Score': round(overall_score, 2),
            'Sustainalytics': esg_data['Sustainalytics'],
            'LSEG': esg_data['LSEG'],
            'MSCI': esg_data['MSCI'],
            'Sentiment Impact': round(sentiment_impact, 2)
        })

    # Limit to Top 5 Recommendations
    bond_recommendations = sorted(bond_recommendations, key=lambda x: x['Score'], reverse=True)[:5]
    recommendations_df = pd.DataFrame(bond_recommendations)
    recommendations_df.drop(columns=['Sustainalytics', 'LSEG', 'MSCI'], inplace=True, errors='ignore')

    # Create Excel if not exists
    if not os.path.exists(output_file):
        wb = Workbook()
        wb.save(output_file)
        print(f"New Excel file created: {output_file}")

    # Save Recommendations to Excel
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # ESG Summary
            esg_summary = pd.DataFrame([{
                'Company Name': esg_data['Company Name'],
                'Sustainalytics': esg_data['Sustainalytics'],
                'LSEG': esg_data['LSEG'],
                'MSCI': esg_data['MSCI'],
                'Green Bond Prediction': 'Positive Impact Expected'
            }])
            try:
                existing_summary = pd.read_excel(output_file, sheet_name='ESG Summary')
                updated_summary = pd.concat([existing_summary, esg_summary], ignore_index=True)
                updated_summary.to_excel(writer, sheet_name='ESG Summary', index=False)
            except ValueError:
                esg_summary.to_excel(writer, sheet_name='ESG Summary', index=False)

            # Top 5 Recommendations
            try:
                existing_recs = pd.read_excel(output_file, sheet_name='Top 5 Recommendations')
                updated_recs = pd.concat([existing_recs, recommendations_df], ignore_index=True)
                updated_recs.to_excel(writer, sheet_name='Top 5 Recommendations', index=False)
            except ValueError:
                recommendations_df.to_excel(writer, sheet_name='Top 5 Recommendations', index=False)
        print(f'Recommendations saved to {output_file}')
    except Exception as e:
        print(f"Error writing to Excel file: {e}")

# Load Data
company_name = 'UBS'
esg_data = load_esg_data('Esg_data.csv', company_name)
sentiment_score = load_sentiment('average_sentiment.txt')
bond_data = pd.read_csv('bond_data.csv')
bond_data.columns = bond_data.columns.str.strip().str.replace(' ', '_').str.upper().str.replace('__', '_')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = pd.to_numeric(bond_data['AMOUNT_ISSUED_(USD_BN.)'].str.replace(',', '').str.replace('$', ''), errors='coerce')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'].fillna(0)
bond_data['Recommended Quantity'] = np.round(bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 10)
bond_data['Recommended Quantity'] = bond_data['Recommended Quantity'].fillna(0)

# Get Recommendations
recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, 'ml_based_recommendations.xlsx')
