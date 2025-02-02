
# Bot Profile Detection on Social Media 🕵️‍♂️

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/higgn/bot-detection-api)  
[![Flask](https://img.shields.io/badge/Flask-API-green)](https://flask.palletsprojects.com/)  
[![Dash](https://img.shields.io/badge/Dash-Dashboard-yellow)](https://dash.plotly.com/)  

![image](https://github.com/user-attachments/assets/d383758a-2974-47b5-8635-609ab053ec73)

A scalable and efficient bot detection system for social media platforms. This project leverages **natural language processing (NLP)**, **machine learning**, and **behavioral analysis** to identify bot accounts based on their content and activity patterns.

---

## 📝 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Dash App](#dash-app)
8. [Model Details](#model-details)
9. [Contributing](#contributing)
10. [License](#license)

---

## 🌟Overview
Bots on social media platforms are becoming increasingly sophisticated, spreading misinformation and manipulating public discussions. This project aims to **detect bot accounts** by analyzing their **content patterns**, **sentiment**, and **behavioral features**. The system provides:
- **Real-time bot detection** using a trained machine learning model.
- **Interactive Dash dashboard** for visualizing predictions.
- **Scalable architecture** for processing large volumes of data.

---

## 🚀 Features
- **Text-Based Detection**:
  - Clean and preprocess social media text using NLP techniques.
  - Extract features using **TF-IDF** and **VADER Sentiment Analysis**.
- **Behavioral Analysis**:
  - Analyze posting patterns (e.g., retweet count, mention count).
  - Compute engagement metrics (e.g., follower-to-following ratio).
- **Machine Learning Model**:
  - Trained on Twitter bot detection datasets.
  - Uses **LightGBM** for classification.
- **API Integration**:
  - Flask-based API for real-time predictions.
- **Interactive Dashboard**:
  - Built using **Dash** and **Bootstrap** for user-friendly interaction.

---

## 🗂️ Project Structure
```plaintext
BotDetection/
│
├── app/
│   ├── app.py                  # Flask API for bot detection
│   ├── dash_app.py             # Dash dashboard for predictions
│
├── data/
│   ├── bot_detection_data.csv  # Raw dataset
│   ├── cleaned_bot_data.csv     # Preprocessed dataset
│   ├── X.npy                   # Features
│   ├── y.npy                   # Labels
│
├── models/
│   ├── bot_detection_model.pkl  # Trained LightGBM model
│   ├── scaler.pkl               # Scaler for feature normalization
│   ├── tfidf.pkl                # TF-IDF vectorizer
│
├── notebooks/                   # Jupyter notebooks for EDA and prototyping
│
├── scripts/
│   ├── feature_extraction.py    # Feature extraction pipeline
│   ├── preprocess_data.py       # Data preprocessing script
│   ├── test.py                  # Model testing script
│   ├── train_model.py           # Model training script
│
├── README.md                    # Project documentation

```

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/higgn/bot-detection-api.git
cd bot-detection-api
```
## Install dependencies:

```bash
pip install -r requirements.txt
```

## Download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## 🖥️ Usage

1. Run the Flask API
```bash
python app/app.py
```
The API will be available at http://127.0.0.1:5000.

2. Run the Dash App
```bash
python app/dash_app.py
```
Access the dashboard at http://127.0.0.1:8050.

## 📚 API Documentation

POST /predict
Description: Predict if a social media account is a bot.
Request Body:
```json
{
  "Tweet": "I am a bot!", 
  "Retweet Count": 1, 
  "Follower Count": 0, 
  "Verified": 0, 
  "Hashtags": "#AI,#MachineLearning", 
  "Mention Count": 0
}
```

Response:

```json
{
  "prediction": 1, 
  "probabilities": [0.2, 0.8]
}
```

## 📊 Dash App
The Dash app provides an interactive interface for bot detection. Users can input tweet details, and the app will display:

Prediction: Bot or Not a Bot.
Probabilities: Confidence scores for each class.
Dash App Screenshot :
![image](https://github.com/user-attachments/assets/a63d2bf4-02eb-4f9a-8be0-37742902257c)
![image](https://github.com/user-attachments/assets/d2373cb9-43c0-4508-805a-232e33ad1b9a)
![image](https://github.com/user-attachments/assets/cd5dafe9-3b7b-4474-bfab-4696a5975dc4)




## 🤖 Model Details

Model Architecture
Algorithm: LightGBM (Gradient Boosting).
Features:
Text features (TF-IDF, sentiment score).
Behavioral features (retweet count, mention count, follower-to-following ratio).
Evaluation Metrics:
Precision: 0.92
Recall: 0.89
F1 Score: 0.90
Training Pipeline
Data Preprocessing:
Clean text (remove URLs, stopwords, lemmatization).
Extract sentiment using VADER.
Feature Engineering:
Compute engagement metrics.
Apply TF-IDF vectorization.


## Model Training:
Train using LightGBM on the preprocessed dataset.

🤝 Contributing
Contributions are welcome! Follow these steps , but for your kind info lol its a base model 🥲:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

## 📜 License
Abhi add nhi kiya 🥲

## 📧 Contact
For questions or feedback, reach out to:
@higgn

## 🌐 GitHub Profile

Made with ❤️ by [GAGAN] for HACK IITK CHALLENGE ROUND 2
🚀 Happy Bot Hunting! 🚀
