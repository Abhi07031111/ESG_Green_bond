import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl import Workbook

# Load ESG Data
def load_esg_data(csv_file, company_name):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()
    df['RATING_AGENCY'] = df['RATING_AGENCY'].str.upper()
    df['ESG_RATING'] = pd.to_numeric(df['ESG_RATING'], errors='coerce')
    
    sustainalytics = df.loc[df['RATING_AGENCY'] == 'SUSTAINALYTICS', 'ESG_RATING'].values
    lseg = df.loc[df['RATING_AGENCY'] == 'LSEG', 'ESG_RATING'].values
    msci = df.loc[df['RATING_AGENCY'] == 'MSCI', 'ESG_RATING'].values
    
    esg_data = {
        'Company Name': company_name,
        'Sustainalytics': sustainalytics[0] if len(sustainalytics) > 0 else np.nan,
        'LSEG': lseg[0] if len(lseg) > 0 else np.nan,
        'MSCI': msci[0] if len(msci) > 0 else np.nan
    }
    
    # Calculate Average ESG Score on a scale of 1-10
    ratings = [esg_data['Sustainalytics'], esg_data['LSEG'], esg_data['MSCI']]
    ratings = [r for r in ratings if not np.isnan(r)]
    if ratings:
        average_esg = np.mean(ratings)
        esg_data['Average ESG (1-10)'] = round((average_esg / 100) * 10, 2)
    else:
        esg_data['Average ESG (1-10)'] = np.nan
        
    return esg_data

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

    bond_data['Overall Score'] = overall_score
    bond_data['Sentiment Impact'] = sentiment_impact
    bond_data['Quality Score'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 0.5
    bond_data['SECTOR_SCORE'] = pd.factorize(bond_data['ISSUER_SECTOR'])[0]
    bond_data['REVIEWER_SCORE'] = pd.factorize(bond_data['EXTERNAL_REVIEWER'])[0]
    X = bond_data[['Overall Score', 'Sentiment Impact', 'Quality Score', 'SECTOR_SCORE', 'REVIEWER_SCORE']]
    y = bond_data['Recommended Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Model Performance: MSE={mse:.2f}, R2={r2:.2f}')

    bond_recommendations = []
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
        
        potential_esg_increase = 0.1
        potential_esg_score = esg_data['Average ESG (1-10)'] + potential_esg_increase
        potential_esg_score = min(potential_esg_score, 10) 

        bond_recommendations.append({
            'Company Name': esg_data['Company Name'],
            'Issuer Name': row['ISSUER_NAME'].title(),
            'Recommendation': recommendation.replace('Hold', 'Strategic Buy'),
            'Potential ESG Score (1-10)': round(potential_esg_score, 2),
            'Comments': 'High Priority Purchase' if recommendation in ['Strong Buy', 'Buy'] else 'Strategic Investment',
            'Score': round(overall_score, 2),
            'Sentiment Impact': round(sentiment_impact, 2)
        })

    bond_recommendations = sorted(bond_recommendations, key=lambda x: x['Score'], reverse=True)[:5]
    recommendations_df = pd.DataFrame(bond_recommendations)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        esg_df = pd.DataFrame([esg_data])
        
        esg_df['Comment'] = esg_df['Average ESG (1-10)'].apply(lambda x: 
            'Excellent' if x >= 8 else 
            'Good' if x >= 6 else 
            'Average' if x >= 4 else 
            'Poor' if x >= 2 else 
            'Very Poor'
        )
        
        esg_df['Recommendation'] = esg_df['Average ESG (1-10)'].apply(lambda x: 
            'Maintain or Increase Green Investments' if x >= 8 else 
            'Consider Strategic Investments' if x >= 6 else 
            'Focus on ESG Improvement'
        )
        
        esg_df.to_excel(writer, sheet_name='Company ESG Ratings', index=False)
        recommendations_df.to_excel(writer, sheet_name='Top Recommendations', index=False)

    print(f'Recommendations saved to {output_file}')

company_name = 'Allianz SE'
esg_data = load_esg_data('Esg_data.csv', company_name)
sentiment_score = load_sentiment('average_sentiment.txt')
bond_data = pd.read_csv('bond_data.csv')
bond_data.columns = bond_data.columns.str.strip().str.replace(' ', '_').str.upper().str.replace('__', '_')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = pd.to_numeric(bond_data['AMOUNT_ISSUED_(USD_BN.)'].str.replace(',', '').str.replace('$', ''), errors='coerce')
bond_data['AMOUNT_ISSUED_(USD_BN.)'] = bond_data['AMOUNT_ISSUED_(USD_BN.)'].fillna(0)
bond_data['Recommended Quantity'] = np.round(bond_data['AMOUNT_ISSUED_(USD_BN.)'] * 10)
bond_data['Recommended Quantity'] = bond_data['Recommended Quantity'].fillna(0)

recommend_green_bonds_ml(esg_data, sentiment_score, bond_data, 'ml_based_recommendations13.xlsx')
