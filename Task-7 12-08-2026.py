import pandas as pd
import re
import nltk

# ==========================================
# NLTK RESOURCES
# ==========================================

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("IMDB_Dataset_CLEANED.csv")

print("Input: Raw movie/customer reviews")
print("Dataset shape:", df.shape)


# ==========================================
# 2. CHECK MISSING VALUES AND DUPLICATES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate reviews:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()


# ==========================================
# 3. CONVERT TO LOWERCASE
# ==========================================

df['cleaned_review'] = df['review'].astype(str).str.lower()

print("\nStep 3: Lowercase conversion completed.")

print("\nOriginal vs Lowercase:")
print(df[['review', 'cleaned_review']].head())


# ==========================================
# 4. REMOVE HTML, URLs, NUMBERS,
#    PUNCTUATION AND SPECIAL CHARACTERS
# ==========================================

def clean_text(text):

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df['cleaned_review'] = df['cleaned_review'].apply(clean_text)

print("\nStep 4: Regex cleaning completed.")


# ==========================================
# 5. REMOVE STOPWORDS
# ==========================================

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return ' '.join(words)


df['cleaned_review'] = df['cleaned_review'].apply(remove_stopwords)

print("Step 5: Stopword removal completed.")


# ==========================================
# 6. TOKENIZATION
# ==========================================

df['tokens'] = df['cleaned_review'].apply(word_tokenize)

print("Step 6: Tokenization completed.")


# ==========================================
# 7. LEMMATIZATION
# ==========================================

lemmatizer = WordNetLemmatizer()

def lemmatize_words(tokens):

    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


df['tokens'] = df['tokens'].apply(lemmatize_words)

print("Step 7: Lemmatization completed.")


# ==========================================
# 8. CREATE FINAL CLEANED REVIEW
# ==========================================

df['cleaned_review'] = df['tokens'].apply(
    lambda words: ' '.join(words)
)

# Remove temporary tokens column
df = df.drop(columns=['tokens'])

print("Step 8: Cleaned review column created.")


# ==========================================
# 9. DISPLAY ORIGINAL VS CLEANED
# ==========================================

print("\n" + "=" * 70)
print("ORIGINAL VS CLEANED REVIEWS")
print("=" * 70)

for i in range(5):

    print("\nOriginal:")
    print(df['review'].iloc[i])

    print("\nCleaned:")
    print(df['cleaned_review'].iloc[i])

    print("-" * 70)


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\nOutput: Cleaned text dataset")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df[['review', 'sentiment', 'cleaned_review']].head())


# ==========================================
# 11. EXPORT CSV
# ==========================================

df.to_csv(
    "IMDB_Dataset_Preprocessed.csv",
    index=False
)

print("\nPreprocessed dataset saved successfully!")
print("File: IMDB_Dataset_Preprocessed.csv")
