import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl import Workbook

def normalize_sustainalytics(score):
    if pd.isna(score):
        return None
    score = max(0, min(100, score))
    return round(10 - (score / 100) * 10, 2)

def normalize_lseg(score):
    if pd.isna(score):
        return None
    score = max(0, min(100, score))
    return round((score / 100) * 10, 2)

def normalize_msci(rating):
    if not isinstance(rating, str):
        return None
    rating = rating.strip().upper()
    rating_map = {
        'CCC': 2.0,
        'B': 3.5,
        'BB': 5.0,
        'BBB': 6.0,
        'A': 7.0,
        'AA': 8.5,
        'AAA': 10.0
    }
    return rating_map.get(rating, None)

def load_esg_data(csv_file, company_name):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()
    df['RATING_AGENCY'] = df['RATING_AGENCY'].str.upper()
    df['ESG_RATING'] = df['ESG_RATING'].astype(str).str.strip()

    sustainalytics_raw_score = df.loc[df['RATING_AGENCY'] == 'SUSTAINALYTICS', 'ESG_RATING'].values
    lseg_raw_score = df.loc[df['RATING_AGENCY'] == 'LSEG', 'ESG_RATING'].values
    msci_raw_score = df.loc[df['RATING_AGENCY'] == 'MSCI', 'ESG_RATING'].values

    sustainalytics_score = normalize_sustainalytics(pd.to_numeric(sustainalytics_raw_score[0], errors='coerce')) if len(sustainalytics_raw_score) > 0 else None
    lseg_score = normalize_lseg(pd.to_numeric(lseg_raw_score[0], errors='coerce')) if len(lseg_raw_score) > 0 else None
    msci_score = normalize_msci(msci_raw_score[0]) if len(msci_raw_score) > 0 else None

    esg_data = {
        'Company Name': company_name,
        'Sustainalytics_RAW': sustainalytics_raw_score[0] if len(sustainalytics_raw_score) > 0 else None,
        'LSEG_RAW': lseg_raw_score[0] if len(lseg_raw_score) > 0 else None,
        'MSCI_RAW': msci_raw_score[0] if len(msci_raw_score) > 0 else None,
        'Sustainalytics': sustainalytics_score,
        'LSEG': lseg_score,
        'MSCI': msci_score,
    }

    ratings = [s for s in [sustainalytics_score, lseg_score, msci_score] if s is not None]
    esg_data['Average ESG (1-10)'] = round(np.mean(ratings), 2) if ratings else np.nan

    return esg_data

def load_sentiment(txt_file):
    with open(txt_file, 'r') as file:
        line = file.readline()
        try:
            sentiment_score = float(line.split(':')[1].strip())
        except (IndexError, ValueError):
            sentiment_score = 0.0
    return sentiment_score

def recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, output_file,
                              lseg_weight=0.4, sustainalytics_weight=0.3, msci_weight=0.3,
                              industry_benchmark=75):
    msci_boost = 10 if esg_data['MSCI'] is not None and esg_data['MSCI'] >= 8.5 else 5
    overall_score = (
        (esg_data['LSEG'] or 0) * lseg_weight +
        (esg_data['Sustainalytics'] or 0) * sustainalytics_weight +
        msci_boost * msci_weight
    )
    sentiment_impact = sentiment_score * 15
    overall_score += sentiment_impact

    bond_data['Overall Score'] = overall_score
    bond_data['Sentiment Impact'] = sentiment_impact
    bond_data['Quality Score'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 0.5
    bond_data['SECTOR_SCORE'] = pd.factorize(bond_data['ISSUER_SECTOR'])[0]
    bond_data['REVIEWER_SCORE'] = pd.factorize(bond_data['EXTERNAL_REVIEWER'])[0]

    features = ['Overall Score', 'Sentiment Impact', 'Quality Score', 'SECTOR_SCORE', 'REVIEWER_SCORE']
    X = bond_data[features]
    y = bond_data['Recommended Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'Model Performance: MSE = {mean_squared_error(y_test, y_pred):.2f}, R2 = {r2_score(y_test, y_pred):.2f}')

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

        bond_recommendations.append({
            'Company Name': esg_data['Company Name'],
            'Issuer Name': row['ISSUER_NAME'].title(),
            'Recommendation': recommendation if recommendation != 'Hold' else 'Strategic Buy',
            'Score': round(overall_score, 2),
            'Sentiment Impact': round(sentiment_impact, 2)
        })

    top_recommendations = sorted(bond_recommendations, key=lambda x: x['Score'], reverse=True)[:5]
    recommendations_df = pd.DataFrame(top_recommendations)

    buffer1 = esg_data['Average ESG (1-10)']
    recommendations_df['Buffer 1'] = buffer1
    recommendations_df['Buffer 2'] = '8 to 10'
    recommendations_df['Buffer 3'] = '6 to 8'
    recommendations_df['Buffer 4'] = '5 to 6'
    recommendations_df['Buffer 5'] = '4 to 5'

    if buffer1 < 6:
        recommendations_df.loc[2:, 'Buffer 4'] = ''
        recommendations_df.loc[4:, 'Buffer 3'] = ''
    elif 6 <= buffer1 < 7:
        recommendations_df['Buffer 4'] = ''
        recommendations_df.loc[2:, 'Buffer 3'] = ''
    elif buffer1 > 8:
        recommendations_df['Buffer 3'] = ''
        recommendations_df['Buffer 4'] = ''
        recommendations_df['Buffer 5'] = ''

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        esg_df = pd.DataFrame([{
            'Company Name': esg_data['Company Name'],
            'Sustainalytics ESG Score': esg_data.get('Sustainalytics_RAW'),
            'LSEG ESG Score': esg_data.get('LSEG_RAW'),
            'MSCI ESG Rating': esg_data.get('MSCI_RAW'),
            'EsgBuffer1': esg_data.get('Average ESG (1-10)')
        }])

        esg_df['EsgBuffer2'] = esg_df['EsgBuffer1'].apply(lambda x: 
            'Excellent' if x >= 8 else 
            'Good' if x >= 6 else 
            'Average' if x >= 4 else 
            'Poor' if x >= 2 else 
            'Very Poor')

        esg_df['Recommendation'] = esg_df['EsgBuffer1'].apply(lambda x: 
            'Maintain or Increase Green Investments' if x >= 8 else 
            'Consider Strategic Investments' if x >= 6 else 
            'Focus on ESG Improvement')
        
        esg_df['Anomalies Detected'] = 'NO' 

        esg_df.to_excel(writer, sheet_name='Esgdata', index=False)
        recommendations_df.to_excel(writer, sheet_name='Data', index=False)

    print(f'Recommendations saved to {output_file}')

company_name = 'Allianz SE'
esg_data = load_esg_data('Esg_data.csv', company_name)
sentiment_score = load_sentiment('average_sentiment.txt')
bond_data = pd.read_csv('bond_data.csv')

bond_data.columns = bond_data.columns.str.strip().str.replace(' ', '_').str.upper().str.replace('__', '_')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = pd.to_numeric(
    bond_data['AMOUNT_ISSUED_(USD_BN.)'].astype(str).str.replace(',', '').str.replace('$', ''),
    errors='coerce')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'].fillna(0)
bond_data['Recommended Quantity'] = np.round(bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 10)
bond_data['Recommended Quantity'] = bond_data['Recommended Quantity'].fillna(0)

recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, 'ml_based_recommendations13.xlsx')
