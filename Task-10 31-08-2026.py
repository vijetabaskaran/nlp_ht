# ============================================================
# ASSIGNMENT SIMILARITY / PLAGIARISM DETECTOR
# ============================================================

import re
import itertools
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. SETTINGS
# ============================================================

# Similarity percentage above this value
# will be considered potentially copied
THRESHOLD = 70


# ============================================================
# 2. SAMPLE ASSIGNMENT DOCUMENTS
# ============================================================
#
# For your actual project, you can replace these texts
# with your assignment documents.
#

documents = {

    "Assignment A": """
    Machine learning is a branch of artificial intelligence
    that enables computers to learn from data and make predictions.
    Machine learning algorithms identify patterns in data and use
    these patterns to make decisions without being explicitly programmed.
    """

    ,

    "Assignment B": """
    Machine learning is a branch of artificial intelligence
    that allows computers to learn from data and make predictions.
    Machine learning algorithms find patterns in data and use
    these patterns to make decisions without explicit programming.
    """

    ,

    "Assignment C": """
    Artificial intelligence is used in many modern applications.
    It allows computer systems to perform tasks that normally require
    human intelligence. These applications include speech recognition,
    computer vision, recommendation systems and robotics.
    """

    ,

    "Assignment D": """
    Data science combines statistics, programming and machine learning
    to extract useful information from large datasets. Data scientists
    clean data, analyze patterns and build predictive models to solve
    real world problems.
    """
}


# ============================================================
# 3. CLEAN AND NORMALIZE TEXT
# ============================================================

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


cleaned_documents = {}

for name, text in documents.items():

    cleaned_documents[name] = clean_text(text)


print("=" * 70)
print("CLEANED DOCUMENTS")
print("=" * 70)

for name, text in cleaned_documents.items():

    print("\n", name)
    print(text)


# ============================================================
# 4. TOKENIZATION
# ============================================================

tokenized_documents = {}

for name, text in cleaned_documents.items():

    tokens = text.split()

    tokenized_documents[name] = tokens


print("\n" + "=" * 70)
print("TOKENIZED DOCUMENTS")
print("=" * 70)

for name, tokens in tokenized_documents.items():

    print("\n", name)
    print(tokens)


# ============================================================
# 5. PREPARE DOCUMENT TEXT FOR TF-IDF
# ============================================================

document_names = list(cleaned_documents.keys())

document_texts = list(cleaned_documents.values())


# ============================================================
# 6. CONVERT DOCUMENTS INTO TF-IDF VECTORS
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF-IDF VECTORS")
print("=" * 70)


vectorizer = TfidfVectorizer(
    stop_words="english"
)


tfidf_matrix = vectorizer.fit_transform(
    document_texts
)


print("TF-IDF matrix created successfully!")

print(
    "Matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 7. DISPLAY TF-IDF FEATURES
# ============================================================

feature_names = vectorizer.get_feature_names_out()

print("\nNumber of TF-IDF features:", len(feature_names))

print("\nSample TF-IDF features:")

print(feature_names[:20])


# ============================================================
# 8. CALCULATE COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING COSINE SIMILARITY")
print("=" * 70)


similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# Convert similarity values to percentages
similarity_percentage = similarity_matrix * 100


# ============================================================
# 9. CREATE SIMILARITY MATRIX
# ============================================================

similarity_df = pd.DataFrame(
    similarity_percentage,
    index=document_names,
    columns=document_names
)


print("\nSimilarity Matrix (%):")

print(
    similarity_df.round(2)
)


# ============================================================
# 10. COMPARE DOCUMENT PAIRS
# ============================================================

print("\n" + "=" * 70)
print("DOCUMENT PAIR ANALYSIS")
print("=" * 70)


results = []


# Generate every unique document pair
for i, j in itertools.combinations(
    range(len(document_names)),
    2
):

    doc1 = document_names[i]
    doc2 = document_names[j]

    score = similarity_percentage[i][j]


    # Determine plagiarism status
    if score >= THRESHOLD:

        status = "Possible Plagiarism"

    else:

        status = "Low Similarity"


    results.append({

        "Document 1": doc1,

        "Document 2": doc2,

        "Similarity (%)": round(score, 2),

        "Status": status

    })


# ============================================================
# 11. CREATE RANKED REPORT
# ============================================================

results_df = pd.DataFrame(results)


# Sort from highest similarity to lowest
results_df = results_df.sort_values(
    by="Similarity (%)",
    ascending=False
).reset_index(drop=True)


print("\nRanked Similarity Report:\n")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 12. DISPLAY SUSPICIOUS DOCUMENT PAIRS
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS DOCUMENT PAIRS")
print("=" * 70)


suspicious_pairs = results_df[
    results_df["Similarity (%)"] >= THRESHOLD
]


if len(suspicious_pairs) > 0:

    for index, row in suspicious_pairs.iterrows():

        print(
            f"\n{row['Document 1']} vs "
            f"{row['Document 2']}"
        )

        print(
            f"Similarity: "
            f"{row['Similarity (%)']}%"
        )

        print(
            "Status: Possible Plagiarism"
        )

else:

    print("\nNo suspicious document pairs found.")


# ============================================================
# 13. DISPLAY LOW-SIMILARITY PAIRS
# ============================================================

print("\n" + "=" * 70)
print("LOW SIMILARITY DOCUMENT PAIRS")
print("=" * 70)


low_similarity_pairs = results_df[
    results_df["Similarity (%)"] < THRESHOLD
]


if len(low_similarity_pairs) > 0:

    for index, row in low_similarity_pairs.iterrows():

        print(
            f"\n{row['Document 1']} vs "
            f"{row['Document 2']}"
        )

        print(
            f"Similarity: "
            f"{row['Similarity (%)']}%"
        )

        print(
            "Status: Low Similarity"
        )


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PLAGIARISM REPORT")
print("=" * 70)

print(
    "\nNumber of documents:",
    len(documents)
)

print(
    "Similarity threshold:",
    THRESHOLD,
    "%"
)

print(
    "Total document pairs:",
    len(results_df)
)

print(
    "Suspicious pairs:",
    len(suspicious_pairs)
)


# ============================================================
# 15. EXPORT REPORT TO CSV
# ============================================================

results_df.to_csv(
    "plagiarism_similarity_report.csv",
    index=False
)


print(
    "\nReport saved as: "
    "plagiarism_similarity_report.csv"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)


for index, row in results_df.iterrows():

    print(
        f"\n{row['Document 1']} vs "
        f"{row['Document 2']}"
    )

    print(
        f"Similarity: "
        f"{row['Similarity (%)']}%"
    )

    print(
        f"Result: "
        f"{row['Status']}"
    )


print("\nPlagiarism detection completed!")
