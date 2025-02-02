from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- Downloads (Run these once) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)

# Load the trained model, TF-IDF vectorizer, and scaler
model = joblib.load("./models/bot_detection_model.pkl")
tfidf = joblib.load("./models/tfidf.pkl")

# Initialize TruncatedSVD (without fitting)
svd = TruncatedSVD(n_components=50, random_state=42)

# --- Text Cleaning Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words).strip()

# --- Preprocess Input Data ---
def preprocess_input(data):
    # Convert input data to a DataFrame
    df = pd.DataFrame([data])

    # Clean text
    df['cleaned_text'] = df['Tweet'].apply(clean_text)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['Tweet'].apply(lambda text: analyzer.polarity_scores(text)['compound'] if isinstance(text, str) else 0)

    # Feature Engineering
    df['Tweet Length'] = df['Tweet'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['Hashtag Count'] = df['Hashtags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    df['Follower to Following Ratio'] = df['Follower Count'] / (df['Follower Count'] + 1)

    # Handle missing values
    df.fillna(0, inplace=True)  # Fill NaN values with 0

    # TF-IDF Features
    text_features_tfidf = tfidf.transform(df['cleaned_text']).toarray()

    # Combine Features
    text_features = np.hstack((text_features_tfidf, df[['vader_sentiment']].values.reshape(-1, 1)))
    behavioral_features = df[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Tweet Length', 'Hashtag Count', 'Follower to Following Ratio']].values
    X_combined = np.hstack((text_features, behavioral_features))

    # Ensure the number of features matches the model's expectations (50)
    if X_combined.shape[1] < 50:
        # Add zeros for missing features
        X_combined = np.hstack((X_combined, np.zeros((X_combined.shape[0], 50 - X_combined.shape[1]))))
    elif X_combined.shape[1] > 50:
        # Truncate extra features
        X_combined = X_combined[:, :50]

    # Dimensionality Reduction (skip if svd is not fitted)
    try:
        X_svd = svd.transform(X_combined)
    except:
        X_svd = X_combined  # Skip SVD if not fitted

    return X_svd

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json
        print("Input Data:", input_data)  # Log input data

        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        print("Processed Data Shape:", processed_data.shape)  # Log processed data shape
        print("Processed Data Sample:", processed_data[0])  # Log sample processed data

        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        print("Prediction:", prediction)  # Log prediction
        print("Probabilities:", prediction_proba)  # Log probabilities

        # Return the result
        return jsonify({
            'prediction': int(prediction[0]),
            'probabilities': prediction_proba.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
