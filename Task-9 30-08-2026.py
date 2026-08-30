# ============================================================
# SMART NEXT-WORD PREDICTOR
# Using WikiText-2 + Unigram, Bigram and Trigram Model
# ============================================================

# If datasets is not installed, uncomment the next line:
# !pip install datasets


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import re
from collections import Counter
from datasets import load_dataset


print("=" * 60)
print("        SMART NEXT-WORD PREDICTOR")
print("=" * 60)


# ============================================================
# 2. LOAD WIKITEXT-2 DATASET
# ============================================================

print("\nLoading WikiText-2 dataset...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

train_data = dataset["train"]

print("Dataset loaded successfully!")
print("Training rows:", len(train_data))


# ============================================================
# 3. CLEAN AND TOKENIZE TEXT
# ============================================================

def clean_and_tokenize(text):

    # Convert to lowercase
    text = text.lower()

    # Keep letters, numbers, spaces and apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()

    return tokens


print("\nCleaning and tokenizing the dataset...")

all_tokens = []

for row in train_data:

    text = row["text"]

    if text.strip():

        tokens = clean_and_tokenize(text)

        all_tokens.extend(tokens)


print("Tokenization completed!")
print("Total tokens:", len(all_tokens))


# ============================================================
# 4. BUILD UNIGRAM FREQUENCY TABLE
# ============================================================

print("\nBuilding unigram frequency table...")

unigram_counts = Counter(all_tokens)

print("Unique unigrams:", len(unigram_counts))


# ============================================================
# 5. BUILD BIGRAM FREQUENCY TABLE
# ============================================================

print("Building bigram frequency table...")

bigram_counts = Counter()

for i in range(len(all_tokens) - 1):

    bigram = (
        all_tokens[i],
        all_tokens[i + 1]
    )

    bigram_counts[bigram] += 1


print("Unique bigrams:", len(bigram_counts))


# ============================================================
# 6. BUILD TRIGRAM FREQUENCY TABLE
# ============================================================

print("Building trigram frequency table...")

trigram_counts = Counter()

for i in range(len(all_tokens) - 2):

    trigram = (
        all_tokens[i],
        all_tokens[i + 1],
        all_tokens[i + 2]
    )

    trigram_counts[trigram] += 1


print("Unique trigrams:", len(trigram_counts))


# ============================================================
# 7. DISPLAY SAMPLE FREQUENCY TABLES
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE FREQUENCY TABLES")
print("=" * 60)


print("\nTop 10 Unigrams:")

for word, count in unigram_counts.most_common(10):

    print(f"{word:<15} : {count}")


print("\nTop 10 Bigrams:")

for bigram, count in bigram_counts.most_common(10):

    print(f"{bigram} : {count}")


print("\nTop 10 Trigrams:")

for trigram, count in trigram_counts.most_common(10):

    print(f"{trigram} : {count}")


# ============================================================
# 8. CALCULATE PROBABILITIES
# ============================================================

total_words = len(all_tokens)


def unigram_probability(word):

    """
    P(word) = Count(word) / Total words
    """

    return unigram_counts[word] / total_words


def bigram_probability(word1, word2):

    """
    P(word2 | word1)

    = Count(word1, word2) / Count(word1)
    """

    previous_count = unigram_counts[word1]

    if previous_count == 0:

        return 0

    return bigram_counts[
        (word1, word2)
    ] / previous_count


def trigram_probability(word1, word2, word3):

    """
    P(word3 | word1, word2)

    = Count(word1, word2, word3)
      / Count(word1, word2)
    """

    previous_count = bigram_counts[
        (word1, word2)
    ]

    if previous_count == 0:

        return 0

    return trigram_counts[
        (word1, word2, word3)
    ] / previous_count


print("\nProbability functions created successfully!")


# ============================================================
# 9. CREATE FAST LOOKUP TABLES
# ============================================================

# These dictionaries make prediction faster.

bigram_next_words = {}

for (word1, word2), count in bigram_counts.items():

    if word1 not in bigram_next_words:

        bigram_next_words[word1] = []

    bigram_next_words[word1].append(
        (word2, count)
    )


trigram_next_words = {}

for (word1, word2, word3), count in trigram_counts.items():

    key = (word1, word2)

    if key not in trigram_next_words:

        trigram_next_words[key] = []

    trigram_next_words[key].append(
        (word3, count)
    )


print("Prediction lookup tables created!")


# ============================================================
# 10. PREDICT NEXT WORDS
# ============================================================

def predict_next_words(sentence, top_n=5):

    # Clean and tokenize user sentence
    tokens = clean_and_tokenize(sentence)

    if len(tokens) == 0:

        return []


    candidates = []


    # --------------------------------------------------------
    # TRIGRAM MODEL
    # --------------------------------------------------------

    if len(tokens) >= 2:

        word1 = tokens[-2]
        word2 = tokens[-1]

        key = (word1, word2)

        if key in trigram_next_words:

            for next_word, count in trigram_next_words[key]:

                probability = trigram_probability(
                    word1,
                    word2,
                    next_word
                )

                candidates.append(
                    (
                        next_word,
                        probability,
                        "Trigram"
                    )
                )


    # --------------------------------------------------------
    # BIGRAM MODEL
    # --------------------------------------------------------

    if len(candidates) == 0:

        previous_word = tokens[-1]

        if previous_word in bigram_next_words:

            for next_word, count in bigram_next_words[
                previous_word
            ]:

                probability = bigram_probability(
                    previous_word,
                    next_word
                )

                candidates.append(
                    (
                        next_word,
                        probability,
                        "Bigram"
                    )
                )


    # --------------------------------------------------------
    # UNIGRAM MODEL
    # --------------------------------------------------------

    if len(candidates) == 0:

        for word, count in unigram_counts.most_common(
            top_n
        ):

            probability = unigram_probability(word)

            candidates.append(
                (
                    word,
                    probability,
                    "Unigram"
                )
            )


    # --------------------------------------------------------
    # SORT BY PROBABILITY
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # Return Top-N predictions
    return candidates[:top_n]


# ============================================================
# 11. DISPLAY PREDICTIONS
# ============================================================

def display_predictions(sentence, top_n=5):

    print("\n" + "=" * 60)

    print("Input:", sentence)

    print("=" * 60)

    predictions = predict_next_words(
        sentence,
        top_n
    )


    if len(predictions) == 0:

        print("\nNo predictions found.")

        return


    print("\nTop predicted next words:\n")


    for i, (word, probability, method) in enumerate(
        predictions,
        start=1
    ):

        print(
            f"{i}. {word:<15} "
            f"- Probability: {probability:.4f} "
            f"[{method}]"
        )


# ============================================================
# 12. TEST THE PREDICTOR
# ============================================================

print("\n")
print("=" * 60)
print("TESTING THE PREDICTOR")
print("=" * 60)


test_sentences = [

    "the united",

    "one of the",

    "in the",

    "the first",

    "machine learning"

]


for sentence in test_sentences:

    display_predictions(
        sentence,
        top_n=5
    )


# ============================================================
# 13. INTERACTIVE USER INPUT
# ============================================================

print("\n")
print("=" * 60)
print("INTERACTIVE NEXT-WORD PREDICTOR")
print("=" * 60)

print("\nEnter a sentence to predict the next word.")
print("Type 'exit' to stop.\n")


while True:

    user_input = input(
        "Enter sentence: "
    )


    # Exit condition
    if user_input.lower().strip() == "exit":

        print("\nThank you for using Smart Next-Word Predictor!")

        break


    # Empty input
    if user_input.strip() == "":

        print("Please enter a sentence.")

        continue


    # Display predictions
    display_predictions(
        user_input,
        top_n=5
    )
