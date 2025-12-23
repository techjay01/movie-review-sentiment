import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# Load Dataset
# --------------------------------
data = pd.read_csv("imdb_reviews.csv")  

# --------------------------------
# Select Required Columns
# --------------------------------
data = data[['id', 'text', 'sentiment']]

# --------------------------------
# Encode Target Variable
# GOOD -> 1, BAD -> 0
# --------------------------------
data['sentiment'] = data['sentiment'].map({'pos': 1, 'neg': 0})

# --------------------------------
# Text Preprocessing Function
# --------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

data['text'] = data['text'].apply(preprocess_text)

# --------------------------------
# Train-Test Split
# --------------------------------
X = data['text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# TF-IDF Vectorization
# --------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --------------------------------
# Model Training
# --------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# --------------------------------
# Evaluation
# --------------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------------------------------
# Save Model and Vectorizer
# --------------------------------
joblib.dump(model, "good_bad_sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and Vectorizer saved successfully.")
