import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Data ---
try:
    data = pd.read_csv(".//data//cleaned_bot_data.csv")  # Load the CLEANED data
    logger.info("Data loaded successfully.")
except FileNotFoundError:
    logger.error("Error: cleaned_bot_data.csv not found. Run preprocess_data.py first.")
    exit()
except Exception as e:
    logger.error(f"Error loading data: {e}")
    exit()

# --- Feature Engineering ---
# 1. TF-IDF Features
try:
    tfidf = joblib.load(".//models//tfidf.pkl")  # Load TF-IDF vectorizer
    text_features_tfidf = tfidf.transform(data['cleaned_text']).toarray()
    logger.info("TF-IDF features extracted successfully.")
except Exception as e:
    logger.error(f"Error loading TF-IDF vectorizer: {e}")
    exit()

# 2. Behavioral Features
data['Tweet Length'] = data['Tweet'].apply(lambda x: len(x) if isinstance(x, str) else 0)
data['Hashtag Count'] = data['Hashtags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
data['Mention Count'] = data['Mention Count'].astype(int)
data['Retweet Count'] = data['Retweet Count'].astype(int)
data['Follower Count'] = data['Follower Count'].astype(int)
data['Follower to Following Ratio'] = data['Follower Count'] / (data['Follower Count'] + 1)  # Avoid division by zero
logger.info("User behavioral features extracted successfully.")

# 3. Temporal Features
if 'Created At' in data.columns:
    data['Created At'] = pd.to_datetime(data['Created At'])
    data['Hour of Day'] = data['Created At'].dt.hour
    data['Day of Week'] = data['Created At'].dt.dayofweek
    data['Month'] = data['Created At'].dt.month
    logger.info("Temporal features extracted successfully.")

# 4. Verified Status
if 'Verified' in data.columns:
    data['Verified'] = data['Verified'].astype(int)
    logger.info("Verified status extracted successfully.")

# --- Combine Features ---
# Text Features: TF-IDF + VADER Sentiment
text_features = np.hstack((text_features_tfidf, data[['vader_sentiment']].values.reshape(-1, 1)))  # 300 + 1

# Behavioral Features
behavioral_features = data[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Tweet Length', 'Hashtag Count', 'Follower to Following Ratio']].values  # 7

# Temporal Features
temporal_features = data[['Hour of Day', 'Day of Week', 'Month']].values if 'Created At' in data.columns else np.zeros((len(data), 3))  # 3

# --- Total Features: 300 + 1 + 7 + 3 = 311 ---
X_combined = np.hstack((text_features, behavioral_features, temporal_features))
logger.info(f"Number of features in X_combined: {X_combined.shape[1]}")

# --- Scaling ---
try:
    scaler = joblib.load(".//models//scaler.pkl")  # Load the scaler
    X_scaled = scaler.transform(X_combined)  # Apply the scaler to the combined features
    logger.info("Features scaled successfully.")
except Exception as e:
    logger.error(f"Error loading or applying scaler: {e}")
    exit()

# --- Save Features ---
try:
    np.save(".//data//X.npy", X_scaled.astype(np.float32))  # Save scaled features
    np.save(".//data//y.npy", data['Bot Label'].values.astype(int))  # Save labels
    logger.info("Feature extraction complete. Features saved to disk.")
except Exception as e:
    logger.error(f"Error saving features: {e}")
    exit()
