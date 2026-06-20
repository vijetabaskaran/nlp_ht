import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist

# Download required resources (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
# Input resume text
text = input("Enter resume text: ")

# 1. Tokenization
tokens = word_tokenize(text)

# Keep only alphabetic words and convert to lowercase
words = [word.lower() for word in tokens if word.isalpha()]

print("Tokens:")
print(words)

# 2. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered_words)

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

print("\nStemmed Words:")
print(stemmed_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print("\nLemmatized Words:")
print(lemmatized_words)

# 5. POS Tagging
pos_tags = pos_tag(lemmatized_words)

print("\nPOS Tags:")
print(pos_tags)

# 6. Frequency Distribution
freq_dist = FreqDist(lemmatized_words)

print("\nFrequency Distribution:")
for word, freq in freq_dist.items():
    print(word, ":", freq)
