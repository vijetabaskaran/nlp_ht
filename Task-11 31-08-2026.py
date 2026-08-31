# ============================================================
# HMM POS TAGGER
# Hidden Markov Model + Viterbi Algorithm
# Dataset: Universal Dependencies English EWT
# ============================================================

# If required, install the libraries:
# !pip install datasets pandas scikit-learn


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import math
import re
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset


print("=" * 70)
print("             HMM POS TAGGER")
print("=" * 70)


# ============================================================
# 2. LOAD UNIVERSAL DEPENDENCIES ENGLISH EWT DATASET
# ============================================================

print("\nLoading Universal Dependencies English EWT dataset...")

dataset = load_dataset(
    "universal-dependencies/universal_dependencies",
    "en_ewt"
)

print("\nDataset loaded successfully!")

print(dataset)

print("\nDataset sizes:")
print("Train :", len(dataset["train"]))
print("Dev   :", len(dataset["dev"]))
print("Test  :", len(dataset["test"]))

# ============================================================
# 3. EXTRACT WORDS AND POS TAGS
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING WORDS AND POS TAGS")
print("=" * 70)


# Universal Dependencies uses UPOS tags
# Examples:
# DET, NOUN, VERB, ADJ, ADV, PRON, etc.

train_data = dataset["train"]
test_data = dataset["test"]


# ============================================================
# 4. DISPLAY SAMPLE DATA
# ============================================================

print("\nSample training sentence:\n")

sample_sentence = train_data[0]

print("Words:")
print(sample_sentence["tokens"])

print("\nPOS Tags:")
print(sample_sentence["upos"])


# ============================================================
# 5. CREATE TRANSITION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING TRANSITION PROBABILITIES")
print("=" * 70)


transition_counts = defaultdict(Counter)

tag_counts = Counter()


# START and END states
START = "<START>"
END = "<END>"


for sentence in train_data:

    words = sentence["tokens"]
    tags = sentence["upos"]

    if len(words) == 0:
        continue

    previous_tag = START

    for tag in tags:

        transition_counts[previous_tag][tag] += 1

        tag_counts[tag] += 1

        previous_tag = tag

    # Transition from last tag to END
    transition_counts[previous_tag][END] += 1


print("Transition counts calculated!")


# ============================================================
# 6. CALCULATE TRANSITION PROBABILITIES
# ============================================================

transition_probabilities = defaultdict(dict)


for previous_tag in transition_counts:

    total = sum(
        transition_counts[previous_tag].values()
    )

    for current_tag in transition_counts[previous_tag]:

        transition_probabilities[
            previous_tag
        ][current_tag] = (
            transition_counts[
                previous_tag
            ][current_tag] / total
        )


print("Transition probabilities calculated!")


# ============================================================
# 7. CREATE EMISSION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING EMISSION PROBABILITIES")
print("=" * 70)


emission_counts = defaultdict(Counter)


for sentence in train_data:

    words = sentence["tokens"]
    tags = sentence["upos"]

    for word, tag in zip(words, tags):

        # Convert word to lowercase
        word = word.lower()

        emission_counts[tag][word] += 1


print("Emission counts calculated!")


# ============================================================
# 8. CALCULATE EMISSION PROBABILITIES
# ============================================================

emission_probabilities = defaultdict(dict)


for tag in emission_counts:

    total = sum(
        emission_counts[tag].values()
    )

    for word in emission_counts[tag]:

        emission_probabilities[tag][word] = (
            emission_counts[tag][word] / total
        )


print("Emission probabilities calculated!")


# ============================================================
# 9. GET ALL POS TAGS
# ============================================================

tags_list = sorted(tag_counts.keys())

print("\nPOS Tags found in dataset:")

print(tags_list)

print("\nNumber of POS tags:", len(tags_list))


# ============================================================
# 10. HANDLE UNKNOWN WORDS
# ============================================================

def get_emission_probability(
    word,
    tag
):

    word = word.lower()

    # Known word
    if word in emission_probabilities[tag]:

        return emission_probabilities[tag][word]


    # Unknown word
    # Use a small probability
    return 1e-6


# ============================================================
# 11. GET TRANSITION PROBABILITY
# ============================================================

def get_transition_probability(
    previous_tag,
    current_tag
):

    if current_tag in transition_probabilities[
        previous_tag
    ]:

        return transition_probabilities[
            previous_tag
        ][current_tag]


    # Small probability for unseen transition
    return 1e-6


# ============================================================
# 12. VITERBI ALGORITHM
# ============================================================

def viterbi(words):

    words = [
        word.lower()
        for word in words
    ]


    if len(words) == 0:

        return []


    # --------------------------------------------------------
    # VITERBI TABLE
    # --------------------------------------------------------

    viterbi_table = []

    backpointer = []


    # ========================================================
    # FIRST WORD
    # ========================================================

    first_word = words[0]

    first_scores = {}
    first_backpointer = {}


    for tag in tags_list:

        transition_prob = get_transition_probability(
            START,
            tag
        )

        emission_prob = get_emission_probability(
            first_word,
            tag
        )


        # Log probabilities prevent underflow
        score = (
            math.log(transition_prob)
            +
            math.log(emission_prob)
        )


        first_scores[tag] = score

        first_backpointer[tag] = START


    viterbi_table.append(first_scores)

    backpointer.append(first_backpointer)


    # ========================================================
    # REMAINING WORDS
    # ========================================================

    for position in range(
        1,
        len(words)
    ):

        current_word = words[position]

        current_scores = {}
        current_backpointer = {}


        for current_tag in tags_list:

            emission_prob = get_emission_probability(
                current_word,
                current_tag
            )

            emission_log = math.log(
                emission_prob
            )


            best_score = float("-inf")

            best_previous_tag = None


            for previous_tag in tags_list:

                transition_prob = (
                    get_transition_probability(
                        previous_tag,
                        current_tag
                    )
                )


                score = (
                    viterbi_table[position - 1][
                        previous_tag
                    ]
                    +
                    math.log(
                        transition_prob
                    )
                    +
                    emission_log
                )


                if score > best_score:

                    best_score = score

                    best_previous_tag = (
                        previous_tag
                    )


            current_scores[current_tag] = best_score

            current_backpointer[current_tag] = (
                best_previous_tag
            )


        viterbi_table.append(
            current_scores
        )

        backpointer.append(
            current_backpointer
        )


    # ========================================================
    # FIND BEST FINAL TAG
    # ========================================================

    last_position = len(words) - 1

    best_final_score = float("-inf")

    best_final_tag = None


    for tag in tags_list:

        end_probability = get_transition_probability(
            tag,
            END
        )


        score = (
            viterbi_table[last_position][tag]
            +
            math.log(end_probability)
        )


        if score > best_final_score:

            best_final_score = score

            best_final_tag = tag


    # ========================================================
    # BACKTRACK
    # ========================================================

    predicted_tags = [
        best_final_tag
    ]


    for position in range(
        len(words) - 1,
        0,
        -1
    ):

        previous_tag = backpointer[position][
            predicted_tags[-1]
        ]

        predicted_tags.append(
            previous_tag
        )


    # Reverse the sequence
    predicted_tags.reverse()


    return predicted_tags


# ============================================================
# 13. TEST CUSTOM SENTENCE
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM SENTENCE POS TAGGING")
print("=" * 70)


sentence = "The student reads a book."


# Tokenize sentence
words = re.findall(
    r"\b[\w']+\b",
    sentence
)


predicted_tags = viterbi(words)


print("\nInput sentence:")
print(sentence)


print("\nPredicted POS tags:\n")


for word, tag in zip(
    words,
    predicted_tags
):

    print(
        f"{word:<15} -> {tag}"
    )


# ============================================================
# 14. EXPECTED EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED POS TAGGING")
print("=" * 70)


print("""
The       -> DET
student   -> NOUN
reads     -> VERB
a         -> DET
book      -> NOUN
""")


# ============================================================
# 15. EVALUATE ON TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON TEST DATASET")
print("=" * 70)


total_words = 0
correct_words = 0


evaluation_results = []


for sentence_index, sentence_data in enumerate(
    test_data
):

    words = sentence_data["tokens"]
    actual_tags = sentence_data["upos"]


    if len(words) == 0:
        continue


    # Predict tags
    predicted_tags = viterbi(words)


    # Compare tags
    for word, actual, predicted in zip(
        words,
        actual_tags,
        predicted_tags
    ):

        total_words += 1

        if actual == predicted:

            correct_words += 1


        evaluation_results.append({

            "Word": word,

            "Actual POS": actual,

            "Predicted POS": predicted,

            "Correct":
                actual == predicted

        })


    # Progress
    if (sentence_index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{sentence_index + 1} test sentences..."
        )


# ============================================================
# 16. CALCULATE ACCURACY
# ============================================================

accuracy = (
    correct_words / total_words
) * 100


print("\n" + "=" * 70)
print("POS TAGGING ACCURACY")
print("=" * 70)


print(
    f"\nTotal words tested : {total_words}"
)

print(
    f"Correct predictions: {correct_words}"
)

print(
    f"Incorrect predictions: "
    f"{total_words - correct_words}"
)

print(
    f"\nPOS Tagging Accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# 17. CREATE EVALUATION REPORT
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_results
)


print("\n" + "=" * 70)
print("EVALUATION REPORT")
print("=" * 70)


print(
    evaluation_df.head(20)
)


# ============================================================
# 18. TAG-WISE ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("TAG-WISE ACCURACY")
print("=" * 70)


tag_accuracy_results = []


for tag in tags_list:

    tag_data = evaluation_df[
        evaluation_df["Actual POS"] == tag
    ]


    if len(tag_data) == 0:
        continue


    correct = tag_data["Correct"].sum()

    total = len(tag_data)

    tag_accuracy = (
        correct / total
    ) * 100


    tag_accuracy_results.append({

        "POS Tag": tag,

        "Total": total,

        "Correct": correct,

        "Accuracy (%)":
            round(tag_accuracy, 2)

    })


tag_accuracy_df = pd.DataFrame(
    tag_accuracy_results
)


print(
    tag_accuracy_df.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE EVALUATION REPORT
# ============================================================

evaluation_df.to_csv(
    "hmm_pos_tagging_evaluation.csv",
    index=False
)


tag_accuracy_df.to_csv(
    "pos_tag_accuracy_report.csv",
    index=False
)


print("\nEvaluation reports saved!")

print(
    "1. hmm_pos_tagging_evaluation.csv"
)

print(
    "2. pos_tag_accuracy_report.csv"
)


# ============================================================
# 20. INTERACTIVE POS TAGGER
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE HMM POS TAGGER")
print("=" * 70)

print(
    "\nEnter a sentence to predict POS tags."
)

print(
    "Type 'exit' to stop."
)


while True:

    user_input = input(
        "\nEnter sentence: "
    )


    if user_input.lower().strip() == "exit":

        print(
            "\nHMM POS Tagger stopped."
        )

        break


    if user_input.strip() == "":

        print(
            "Please enter a sentence."
        )

        continue


    # Tokenize user sentence
    words = re.findall(
        r"\b[\w']+\b",
        user_input
    )


    # Predict POS tags
    predicted_tags = viterbi(words)


    print("\nPOS Tags:\n")


    for word, tag in zip(
        words,
        predicted_tags
    ):

        print(
            f"{word:<15} -> {tag}"
        )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("HMM POS TAGGING COMPLETED")
print("=" * 70)
