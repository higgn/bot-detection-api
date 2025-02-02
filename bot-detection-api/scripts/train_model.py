import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import gc

# 1. Data Loading (memmap and checks)
try:
    print("Loading X (memmap)...")
    X = np.load(".//data//X.npy", mmap_mode='r')
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)

    print("Loading y (memmap)...")
    y = np.load(".//data//y.npy", mmap_mode='r')
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

except FileNotFoundError:
    print("Error: X.npy or y.npy not found. Run feature extraction first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Handle Rare Classes (crucial)
class_counts = pd.Series(y).value_counts()
print("Initial Class Distribution:\n", class_counts)
rare_classes = class_counts[class_counts < 10].index.tolist()
y_modified = np.where(np.isin(y, rare_classes), 2, y)  # Combine rare classes into class 2
print("Updated Class Distribution (Rare classes combined):\n", pd.Series(y_modified).value_counts())

# 3. Data Splitting (stratified - using y_modified)
try:
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_modified, test_size=0.2, random_state=42, stratify=y_modified)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

# 4. Handle Class Imbalance (SMOTE)
try:
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled X_train shape:", X_train_resampled.shape)
    print("Resampled y_train shape:", y_train_resampled.shape)
except Exception as e:
    print(f"Error in SMOTE: {e}")
    exit()

# 5. Scaling and Dimensionality Reduction (using the saved scaler)
try:
    print("Scaling and reducing dimensionality...")
    scaler = joblib.load(".//models//scaler.pkl")  # Load the saved scaler
    X_train_scaled = scaler.transform(X_train_resampled)  # Use the loaded scaler
    X_test_scaled = scaler.transform(X_test)  # Use the loaded scaler

    svd = TruncatedSVD(n_components=50, random_state=42)  # Reduce n_components further
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)

    print("X_train shape (after SVD):", X_train_svd.shape)
    print("X_test shape (after SVD):", X_test_svd.shape)
except Exception as e:
    print(f"Error in scaling or SVD: {e}")
    exit()

# 6. LightGBM Model (optimized for performance)
model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'num_leaves': [20, 30, 40]
}

# 7. Hyperparameter Tuning (with error handling)
try:
    print("Tuning LightGBM...")
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_svd, y_train_resampled)
    best_model = grid_search.best_estimator_
    print(f"Best LightGBM Parameters:", grid_search.best_params_)
except Exception as e:
    print(f"Error in hyperparameter tuning: {e}")
    exit()

# 8. Save the Best Model (with error handling)
try:
    print("Saving the best model...")
    joblib.dump(best_model, ".//models//bot_detection_model.pkl")
    print("Best model saved as bot_detection_model.pkl!")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

# 9. Evaluation (with error handling and comprehensive metrics)
try:
    print("Evaluating...")
    y_pred = best_model.predict(X_test_svd)
    accuracy = accuracy_score(y_test, y_pred)
    print("LightGBM Model Accuracy:", accuracy)

    # Comprehensive evaluation metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)  # Get confusion matrix for plotting
    print("Confusion Matrix:\n", cm)  # Print confusion matrix
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))  # multi_class added

    # Plot confusion matrix (improved)
    plt.figure(figsize=(8, 6))  # Adjust figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)  # cbar=False for cleaner look
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

except Exception as e:
    print(f"Error in evaluation: {e}")
    exit()

# 10. Feature Importance Analysis (optional)
try:
    print("Feature Importances:")
    importances = best_model.feature_importances_
    print(importances)
except Exception as e:
    print(f"Error in feature importance analysis: {e}")
    exit()

print("Script completed.")