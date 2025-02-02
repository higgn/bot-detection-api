import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Downloads (Run these once) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# --- Load Data ---
try:
    data = pd.read_csv(".//data//bot_detection_data.csv")
except FileNotFoundError:
    print("Error: bot_detection_data.csv not found. Make sure the file exists in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- NaN Handling (Robust) ---
# 1. Handle 'Hashtags' FIRST (Crucial!)
if 'Hashtags' in data.columns:
    data['Hashtags'] = data['Hashtags'].fillna("")  # Fill with empty string

# 2. Identify NUMERIC and CATEGORICAL columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()

# 3. Impute NUMERIC columns using MEDIAN
if numeric_cols:
    imputer_numeric = SimpleImputer(strategy='median')
    data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])

# 4. Impute CATEGORICAL columns using MODE
if categorical_cols:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

# --- NaN Hunting ---
print("NaN Count after Imputation:", data.isna().sum().sum())  # This MUST be 0

# --- Text Cleaning ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""  # Or " " if you prefer a space
    text = re.sub(r'http\S+|www\.\S+', '', text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words).strip()

data['cleaned_text'] = data['Tweet'].apply(clean_text)

# --- Sentiment Analysis (VADER) ---
analyzer = SentimentIntensityAnalyzer()
data['vader_sentiment'] = data['Tweet'].apply(lambda text: analyzer.polarity_scores(text)['compound'] if isinstance(text, str) else 0)
print("VADER sentiment analysis completed.")

# --- Feature Engineering (After NaN Handling) ---
# 1. Temporal Features
if 'Created At' in data.columns:
    data['Created At'] = pd.to_datetime(data['Created At'])
    data['Hour of Day'] = data['Created At'].dt.hour
    data['Day of Week'] = data['Created At'].dt.dayofweek
    data['Month'] = data['Created At'].dt.month  # Add month feature

# 2. User Metadata Features
if 'Follower Count' in data.columns:
    data['Follower to Following Ratio'] = data['Follower Count'] / (data['Follower Count'] + 1)  # Avoid division by zero

# 3. Verified Status
if 'Verified' in data.columns:
    data['Verified'] = data['Verified'].astype(int)

# 4. Tweet-Specific Features
data['Tweet Length'] = data['Tweet'].apply(lambda x: len(x) if isinstance(x, str) else 0)
data['Hashtag Count'] = data['Hashtags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
data['Mention Count'] = data['Mention Count'].astype(int)
data['Retweet Count'] = data['Retweet Count'].astype(int)
data['Follower Count'] = data['Follower Count'].astype(int)

# --- NaN Hunting (Final Check) ---
print("NaN Count after Feature Engineering:", data.isna().sum().sum())  # This MUST be 0

# --- TF-IDF (Create and Save) ---
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))  # Reduced features
text_features_tfidf = tfidf.fit_transform(data['cleaned_text']).toarray()
joblib.dump(tfidf, ".//models//tfidf.pkl")  # Save the TF-IDF vectorizer

# --- Combine All Features ---
text_features = np.hstack((text_features_tfidf, data[['vader_sentiment']].values.reshape(-1, 1)))  # 300 + 1
behavioral_features = data[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Tweet Length', 'Hashtag Count', 'Follower to Following Ratio']].values  # 7
temporal_features = data[['Hour of Day', 'Day of Week', 'Month']].values if 'Created At' in data.columns else np.zeros((len(data), 3))  # 3

# --- Total Features: 300 + 1 + 7 + 3 = 311 ---
X_combined = np.hstack((text_features, behavioral_features, temporal_features))

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
joblib.dump(scaler, ".//models//scaler.pkl")  # Save the scaler

# --- Memory Optimization ---
for col in data.select_dtypes(include=['float64']):
    data[col] = data[col].astype(np.float32)
for col in data.select_dtypes(include=['int64']):
    data[col] = data[col].astype(np.int32)

# --- Debugging: Print Feature Counts ---
print("Number of TF-IDF features:", text_features_tfidf.shape[1])
print("Number of behavioral features:", behavioral_features.shape[1])
print("Number of temporal features:", temporal_features.shape[1])
print("Total number of features in X_combined:", X_combined.shape[1])

# --- Save Cleaned Data ---
try:
    data.to_csv(".//data//cleaned_bot_data.csv", index=False)
    print("Preprocessing complete. Cleaned data saved to cleaned_bot_data.csv")
except Exception as e:
    print(f"Error saving preprocessed data: {e}")
    exit()