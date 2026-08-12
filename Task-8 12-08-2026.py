import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords


# 1. LOAD DATASET
df = pd.read_csv("labeled_final_test.csv")

print("Input: Student assignment/text samples")
print("Dataset shape:", df.shape)

# 2. SELECT TEXT
documents = df["sentence1"].dropna().head(20).tolist()

document_names = [
    f"Assignment {i+1}"
    for i in range(len(documents))
]

# 3. CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


cleaned_documents = [
    clean_text(text)
    for text in documents
]

# 4. TOKENIZATION + STOPWORD REMOVAL
stop_words = set(stopwords.words("english"))

def tokenize(text):
    words = text.split()
    return [
        word for word in words
        if word not in stop_words
    ]

tokenized_documents = [
    tokenize(text)
    for text in cleaned_documents
]

document_texts = [
    " ".join(words)
    for words in tokenized_documents
]

print("\nText cleaning and tokenization completed.")


# 5. SHOW ORIGINAL VS CLEANED
print("\nOriginal vs Cleaned:")

for i in range(min(3, len(documents))):
    print("\nOriginal:", documents[i])
    print("Cleaned :", document_texts[i])


# 6. TF-IDF
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    document_texts
)

print("\nTF-IDF conversion completed.")
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# 7. COSINE SIMILARITY
similarity_matrix = cosine_similarity(tfidf_matrix)

similarity_percentage = similarity_matrix * 100


# 8. COMPARE DOCUMENT PAIRS
threshold = 70

results = []

for i in range(len(document_names)):
    for j in range(i + 1, len(document_names)):

        score = similarity_percentage[i][j]

        if score >= threshold:
            status = "Possible Plagiarism"
        else:
            status = "Low Similarity"

        results.append([
            document_names[i],
            document_names[j],
            score,
            status
        ])


# 9. RANK RESULTS
results.sort(
    key=lambda x: x[2],
    reverse=True
)


# 10. DISPLAY TOP 10
print("\n" + "=" * 60)
print("RANKED PLAGIARISM REPORT")
print("=" * 60)

for doc1, doc2, score, status in results[:10]:

    print(
        f"{doc1} vs {doc2}: "
        f"{score:.2f}% → {status}"
    )


# 11. SAVE REPORT
report = pd.DataFrame(
    results,
    columns=[
        "Document 1",
        "Document 2",
        "Similarity (%)",
        "Status"
    ]
)

report.to_csv(
    "Plagiarism_Similarity_Report.csv",
    index=False
)

print("\nReport saved as:")
print("Plagiarism_Similarity_Report.csv")
