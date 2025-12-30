from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
import logging

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- NLTK SETUP (RENDER SAFE) ----------------------
nltk.data.path.append("/tmp")

def download_nltk_data():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"NLTK download failed: {e}")

download_nltk_data()

# ---------------------- FLASK APP ----------------------
app = Flask(__name__)
CORS(app)

# ---------------------- LOAD DATASET ----------------------
try:
    data = pd.read_csv("data/spam.csv", encoding="latin-1")
    data = data[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    data["label"] = data["label"].map({"safe": 0, "spam": 1})
    data.dropna(inplace=True)
    logger.info(f"Dataset loaded: {len(data)} rows")
except Exception as e:
    logger.error(f"Dataset loading failed: {e}")
    raise

# ---------------------- SPLIT DATA ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ---------------------- TEXT PREPROCESSING ----------------------
custom_stopwords = set(stopwords.words("english")).union(
    {"subject", "http", "www", "com"}
)

lemmatizer = WordNetLemmatizer()

def preprocess_message(message: str) -> str:
    message = message.lower()
    message = re.sub(r"http\S+|www\S+|@\S+|\d+", "", message)
    message = re.sub(r"[^\w\s]", "", message)
    words = message.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in custom_stopwords]
    return " ".join(words)

# ---------------------- VECTORIZATION ----------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------- HANDLE IMBALANCE ----------------------
smote = SMOTE(random_state=42)
X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

# ---------------------- CLASS WEIGHTS ----------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ---------------------- MODEL TRAINING ----------------------
model = RandomForestClassifier(
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_vec, y_train)
model = grid_search.best_estimator_

logger.info("Model training completed")

# ---------------------- EVALUATION ----------------------
y_pred = model.predict(X_test_vec)

logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
logger.info("\n" + classification_report(y_test, y_pred))

# ---------------------- ROUTES ----------------------
@app.route("/")
def home():
    return "Email Spam Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        processed = preprocess_message(message)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        return jsonify({
            "message": message,
            "prediction": "spam" if prediction == 1 else "safe"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ---------------------- RUN APP (LOCAL ONLY) ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
