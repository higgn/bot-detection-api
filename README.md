
# Bot Profile Detection on Social Media 🕵️‍♂️

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/higgn/bot-detection-api)  
[![Flask](https://img.shields.io/badge/Flask-API-green)](https://flask.palletsprojects.com/)  
[![Dash](https://img.shields.io/badge/Dash-Dashboard-yellow)](https://dash.plotly.com/)  

![image](https://github.com/user-attachments/assets/d383758a-2974-47b5-8635-609ab053ec73)

A scalable and efficient bot detection system for social media platforms. This project leverages **natural language processing (NLP)**, **machine learning**, and **behavioral analysis** to identify bot accounts based on their content and activity patterns.


## LIVE URL ❤️ http://13.251.44.12:8050/  & http://13.251.44.12:5000/predict 

## ☁️ AWS Deployment

This section describes how to deploy the bot detection system on AWS.

### Prerequisites

*   An AWS account.
*   An EC2 instance (Amazon Linux recommended).
*   Basic knowledge of SSH and Linux command line.

### Steps

1.  **Connect to your EC2 instance:** Use SSH to connect to your EC2 instance.

2.  **Clone the repository:**

    ```bash
    git clone [https://github.com/higgn/bot-detection-api.git](https://github.com/higgn/bot-detection-api.git)
    cd bot-detection-api
    ```

3.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install gunicorn
    ```

5.  **Download NLTK data:**

    ```python
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    ```

6.  **Run the Flask API (using Gunicorn in production):**

    ```bash
    cd app
    gunicorn --bind 0.0.0.0:5000 app:app &  # Run in the background
    ```

7.  **Run the Dash App (using Gunicorn in production):**

    ```bash
    cd app
    gunicorn --bind 0.0.0.0:8050 dash_app:server & # Run in the background
    ```

8.  **Configure Security Groups:**

    *   Open the EC2 console and navigate to "Security groups."
    *   Select the security group associated with your EC2 instance.
    *   Add inbound rules:
        *   **Custom TCP Rule, TCP, Port 5000, Source 0.0.0.0/0 (or restricted IP)** - For the Flask API.
        *   **Custom TCP Rule, TCP, Port 8050, Source 0.0.0.0/0 (or restricted IP)** - For the Dash app.

9.  **Access the applications:**

    *   **Flask API:** `http://<your_ec2_public_ip>:5000/predict`
    *   **Dash App:** `http://<your_ec2_public_ip>:8050`

10. **Keep applications running persistently (using tmux):**

    *   Install tmux: `sudo yum install tmux` (or `sudo apt install tmux` on Debian/Ubuntu).
    *   Start a tmux session: `tmux new -s bot_detection`
    *   Inside the tmux session, create two windows (Ctrl+b c) and run the Flask API and Dash app in separate windows (following steps 6 and 7).
    *   Detach from the tmux session (Ctrl+b d). The applications will continue running in the background.

**Important Notes:**

*   **Replace placeholders:** Replace `<your_ec2_public_ip>` with the actual public IP address or DNS name of your EC2 instance.
*   **Security:** Restricting the source IP in your security group rules is highly recommended for production.  Use HTTPS for production deployments to encrypt traffic.
*   **Process managers (systemd, pm2, supervisor):** For a more robust production deployment, consider using a process manager like `systemd`, `pm2`, or `supervisor` to manage your Flask API and Dash app processes. This will ensure that the applications restart automatically if the server reboots or if there are any unexpected errors. The provided instructions use `tmux` for simplicity, but a process manager is best for production.
*   **Load Balancer (for scale and HA):** For production-level scalability and high availability, consider using a load balancer in front of multiple EC2 instances running your applications.
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
