from transformers import pipeline

# Load sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text, chunk_size=512):
    """Analyzes sentiment of UBS ESG-related text in chunks and calculates average sentiment."""
    # Split text into chunks of max 'chunk_size' tokens
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    sentiments = []
    for chunk in text_chunks:
        result = sentiment_analyzer(chunk)[0]
        sentiment_score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        sentiments.append(sentiment_score)
    
    # Compute average sentiment score
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    return sentiments, avg_sentiment

# Read UBS ESG report
with open("ubs_esg_report.txt", "r", encoding="utf-8") as file:
    ubs_report_text = file.read()

# Analyze sentiment of the full report in chunks
sentiment_results, avg_sentiment = analyze_sentiment(ubs_report_text)

# Print sentiment results
for i, score in enumerate(sentiment_results, start=1):
    label = "POSITIVE" if score > 0 else "NEGATIVE"
    print(f"Chunk {i}: Sentiment = {label}, Score = {abs(score):.4f}")

print(f"\nAverage Sentiment Score: {avg_sentiment:.4f}")

# Save average sentiment score to a text file
with open("average_sentiment.txt", "w", encoding="utf-8") as file:
    file.write(f"Average Sentiment Score: {avg_sentiment:.4f}\n")

print("Average sentiment score saved to 'average_sentiment.txt'")
