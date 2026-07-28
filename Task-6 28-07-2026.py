import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Choose one model
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Download stopwords
nltk.download('stopwords')

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

print(data.head())

# Stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text Preprocessing Function
def preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
data["Clean_Review"] = data["review"].apply(preprocess)

print(data[["review", "Clean_Review"]].head())

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data["Clean_Review"])
y = data["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = MultinomialNB()

# Logistic Regression
# model = LogisticRegression(max_iter=1000)

# Support Vector Machine
# model = LinearSVC()

# Train
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Evaluation
print("\nAccuracy")
print(accuracy_score(y_test, prediction))

print("\nPrecision")
print(precision_score(y_test, prediction, pos_label="positive"))

print("\nRecall")
print(recall_score(y_test, prediction, pos_label="positive"))

print("\nF1 Score")
print(f1_score(y_test, prediction, pos_label="positive"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

print("\nClassification Report")
print(classification_report(y_test, prediction))

# Predict New Reviews
new_reviews = [
    "This movie was fantastic. I loved every scene.",
    "Worst movie ever. Completely wasted my time."
]

new_reviews_clean = [preprocess(review) for review in new_reviews]

new_reviews_vector = tfidf.transform(new_reviews_clean)

result = model.predict(new_reviews_vector)

print("\nNew Review Predictions")

for review, sentiment in zip(new_reviews, result):
    print("Review :", review)
    print("Sentiment :", sentiment)
    print()
