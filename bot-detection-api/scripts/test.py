import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- Load your data ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJECT_ROOT, "data", "cleaned_bot_data.csv") # Correct path
data = pd.read_csv(data_path)

# 1. Check for Non-String Values in 'Tweet' Column
non_string_tweets = data[~data['Tweet'].apply(lambda x: isinstance(x, str))]

if not non_string_tweets.empty:
    print("Found non-string values in 'Tweet' column:")
    print(non_string_tweets)  # Print the rows with non-string values

# 2. If non-string values exist, handle them:
#    a. Option 1: Convert to string (if appropriate):
#    data['Tweet'] = data['Tweet'].astype(str)

#    b. Option 2 (Recommended): Replace with empty string or NaN:
#    data['Tweet'] = data['Tweet'].apply(lambda x: str(x) if not pd.isna(x) else "") # Convert to str if not NaN, else empty str
# OR
#    data['Tweet'] = data['Tweet'].apply(lambda x: str(x) if isinstance(x, str) else np.nan) # Convert to str if string, else NaN
#    data.dropna(subset=['Tweet'], inplace=True) # Drop rows with NaN in 'Tweet' column

# 3. BERT Embedding Function (Simplified)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()[0]

# 4. Apply BERT Embeddings (Now should work)
data['bert_embeddings'] = data['Tweet'].apply(get_bert_embedding)

# 5. Print Shape
print("Shape of bert_embeddings:", data['bert_embeddings'].shape)

# ... (Rest of your code)