# ESG_Green_bond


The recommendations are calculated using the following method:

Score Calculation:

The score for each green bond is calculated as:
Score
=
(
LSEG
×
0.5
)
+
(
Sustainalytics
×
0.3
)
+
(
MSCI Bonus
)
+
(
Sentiment Score
×
10
)
Score=(LSEG×0.5)+(Sustainalytics×0.3)+(MSCI Bonus)+(Sentiment Score×10)
Here:
LSEG is multiplied by 0.5.
Sustainalytics is multiplied by 0.3.
MSCI Bonus:
If MSCI is AA or AAA, it adds 100 to the score.
Otherwise, it adds 50.
Sentiment Score is multiplied by 10.
Recommendation Criteria:

Based on the calculated score:
Score > 75 → Recommendation: "Strong Buy"
Score > 50 → Recommendation: "Buy"
Otherwise → Recommendation: "Hold"
Filtering & Limiting Recommendations:

Only green bonds (row["Bond type"].lower() == "green") are considered.
Recommendations are sorted by score in descending order.
The top 5 recommendations are selected.
Output:

The recommendations are saved to a CSV file, with the Sustainalytics, LSEG, and MSCI scores shown only in the header.
