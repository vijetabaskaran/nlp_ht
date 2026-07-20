import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("pos_tags.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==================================================
# CREATE SENTENCES
# ==================================================

sentences = []
temp = []

for _, row in df.iterrows():

    word = str(row["word"])
    tag = str(row["tag"])

    temp.append((word, tag))

    if len(temp) == 20:
        sentences.append(temp)
        temp = []

if len(temp) > 0:
    sentences.append(temp)

print("Total Sentences:", len(sentences))

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

train_data, test_data = train_test_split(
    sentences,
    test_size=0.2,
    random_state=42
)

# ==================================================
# VOCABULARY AND TAGS
# ==================================================

vocab = set()
tag_set = set()

for sentence in train_data:

    for word, tag in sentence:

        vocab.add(word.lower())
        tag_set.add(tag)

vocab = sorted(list(vocab))
tag_set = sorted(list(tag_set))

word_to_idx = {
    word: i
    for i, word in enumerate(vocab)
}

tag_to_idx = {
    tag: i
    for i, tag in enumerate(tag_set)
}

idx_to_tag = {
    i: tag
    for tag, i in tag_to_idx.items()
}

V = len(vocab)
N_TAGS = len(tag_set)

print("Vocabulary Size:", V)
print("Number of Tags:", N_TAGS)

# ==================================================
# HMM MATRICES WITH LAPLACE SMOOTHING
# ==================================================

initial_counts = np.ones(N_TAGS)

transition_counts = np.ones(
    (N_TAGS, N_TAGS)
)

emission_counts = np.ones(
    (N_TAGS, V)
)

# ==================================================
# COUNT INITIAL, TRANSITION, EMISSION
# ==================================================

for sentence in train_data:

    first_tag = sentence[0][1]

    initial_counts[
        tag_to_idx[first_tag]
    ] += 1

    for i, (word, tag) in enumerate(sentence):

        word = word.lower()

        tag_idx = tag_to_idx[tag]

        emission_counts[
            tag_idx,
            word_to_idx[word]
        ] += 1

        if i > 0:

            prev_tag = sentence[i - 1][1]

            transition_counts[
                tag_to_idx[prev_tag],
                tag_idx
            ] += 1

# ==================================================
# CONVERT COUNTS TO PROBABILITIES
# ==================================================

initial_prob = (
    initial_counts /
    initial_counts.sum()
)

transition_prob = (
    transition_counts /
    transition_counts.sum(
        axis=1,
        keepdims=True
    )
)

emission_prob = (
    emission_counts /
    emission_counts.sum(
        axis=1,
        keepdims=True
    )
)

# ==================================================
# LOG SPACE
# ==================================================

log_initial = np.log(initial_prob)
log_transition = np.log(transition_prob)
log_emission = np.log(emission_prob)

# ==================================================
# VECTORISED VITERBI ALGORITHM
# ==================================================

def viterbi(words):

    length = len(words)

    dp = np.full(
        (N_TAGS, length),
        -np.inf
    )

    backpointer = np.zeros(
        (N_TAGS, length),
        dtype=int
    )

    first_word = words[0].lower()

    if first_word in word_to_idx:

        emission = log_emission[
            :,
            word_to_idx[first_word]
        ]

    else:

        emission = np.log(
            np.ones(N_TAGS) * 1e-10
        )

    dp[:, 0] = (
        log_initial + emission
    )

    for t in range(1, length):

        word = words[t].lower()

        if word in word_to_idx:

            emission = log_emission[
                :,
                word_to_idx[word]
            ]

        else:

            emission = np.log(
                np.ones(N_TAGS) * 1e-10
            )

        scores = (
            dp[:, t - 1][:, None]
            + log_transition
        )

        dp[:, t] = (
            np.max(scores, axis=0)
            + emission
        )

        backpointer[:, t] = np.argmax(
            scores,
            axis=0
        )

    best_last_tag = np.argmax(
        dp[:, -1]
    )

    best_path = [best_last_tag]

    for t in range(length - 1, 0, -1):

        best_last_tag = backpointer[
            best_last_tag,
            t
        ]

        best_path.append(best_last_tag)

    best_path.reverse()

    predicted_tags = [
        idx_to_tag[i]
        for i in best_path
    ]

    return predicted_tags

# ==================================================
# EVALUATION
# ==================================================

actual_tags = []
predicted_tags = []

for sentence in test_data:

    words = [
        word
        for word, tag in sentence
    ]

    true_tags = [
        tag
        for word, tag in sentence
    ]

    pred_tags = viterbi(words)

    actual_tags.extend(true_tags)
    predicted_tags.extend(pred_tags)

# ==================================================
# METRICS
# ==================================================

print("\nAccuracy")
print(
    accuracy_score(
        actual_tags,
        predicted_tags
    )
)

print("\nPrecision")
print(
    precision_score(
        actual_tags,
        predicted_tags,
        average="weighted",
        zero_division=0
    )
)

print("\nRecall")
print(
    recall_score(
        actual_tags,
        predicted_tags,
        average="weighted",
        zero_division=0
    )
)

print("\nF1 Score")
print(
    f1_score(
        actual_tags,
        predicted_tags,
        average="weighted",
        zero_division=0
    )
)

print("\nClassification Report")
print(
    classification_report(
        actual_tags,
        predicted_tags,
        zero_division=0
    )
)

# ==================================================
# UNSEEN SENTENCES
# ==================================================

unseen_sentences = [

    "The dog is running",

    "She reads a book",

    "Students are studying NLP",

    "Birds fly in the sky",

    "Artificial intelligence improves healthcare"

]

print("\nUNSEEN SENTENCE TESTING")
print("=" * 50)

for sent in unseen_sentences:

    words = sent.split()

    tags = viterbi(words)

    print("\nSentence:", sent)

    for word, tag in zip(words, tags):

        print(f"{word:15} -> {tag}")
