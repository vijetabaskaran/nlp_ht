# ============================================================
# LSTM CUSTOMER REVIEW SENTIMENT ANALYZER
# Dataset: IMDb 50K Movie Reviews
# Framework: PyTorch
# ============================================================

# If required, run this first:
# !pip install datasets torch scikit-learn matplotlib


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. SET RANDOM SEEDS
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 3. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("        LSTM CUSTOMER REVIEW SENTIMENT ANALYZER")
print("=" * 70)

print("\nDevice being used:", device)


# ============================================================
# 4. LOAD IMDb 50K DATASET
# ============================================================

print("\nLoading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

print("\nDataset loaded successfully!")

print(dataset)

print("\nTraining reviews:", len(dataset["train"]))
print("Testing reviews :", len(dataset["test"]))


# ============================================================
# 5. COMBINE DATA
# ============================================================

train_reviews = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"])

test_reviews = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

all_reviews = train_reviews + test_reviews
all_labels = train_labels + test_labels

print("\nTotal reviews:", len(all_reviews))


# ============================================================
# 6. CLEAN TEXT
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    # Keep only letters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning reviews...")


cleaned_reviews = [
    clean_text(review)
    for review in all_reviews
]


print("Cleaning completed!")


# ============================================================
# 7. TOKENIZATION
# ============================================================

def tokenize(text):

    return text.split()


tokenized_reviews = [
    tokenize(review)
    for review in cleaned_reviews
]


print("\nTokenization completed!")

print("\nSample tokens:")
print(tokenized_reviews[0][:30])


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

reviews_train, reviews_temp, labels_train, labels_temp = (
    train_test_split(
        tokenized_reviews,
        all_labels,
        test_size=0.30,
        random_state=42,
        stratify=all_labels
    )
)


reviews_val, reviews_test, labels_val, labels_test = (
    train_test_split(
        reviews_temp,
        labels_temp,
        test_size=0.50,
        random_state=42,
        stratify=labels_temp
    )
)


print("\nDataset split:")

print("Training   :", len(reviews_train))
print("Validation :", len(reviews_val))
print("Testing    :", len(reviews_test))


# ============================================================
# 9. CREATE VOCABULARY
# ============================================================

print("\nCreating vocabulary...")


MAX_VOCAB_SIZE = 20000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


word_counter = Counter()


for review in reviews_train:

    word_counter.update(review)


# Most common words
most_common_words = word_counter.most_common(
    MAX_VOCAB_SIZE - 2
)


vocab = {

    PAD_TOKEN: 0,

    UNK_TOKEN: 1

}


for word, count in most_common_words:

    if word not in vocab:

        vocab[word] = len(vocab)


print("Vocabulary size:", len(vocab))


# ============================================================
# 10. CONVERT WORDS TO INTEGER SEQUENCES
# ============================================================

def text_to_sequence(tokens):

    sequence = []

    for word in tokens:

        if word in vocab:

            sequence.append(
                vocab[word]
            )

        else:

            sequence.append(
                vocab[UNK_TOKEN]
            )

    return sequence


train_sequences = [
    text_to_sequence(review)
    for review in reviews_train
]


val_sequences = [
    text_to_sequence(review)
    for review in reviews_val
]


test_sequences = [
    text_to_sequence(review)
    for review in reviews_test
]


# ============================================================
# 11. PAD SEQUENCES
# ============================================================

MAX_LENGTH = 200


def pad_sequence(sequence):

    # Truncate long reviews
    if len(sequence) > MAX_LENGTH:

        sequence = sequence[
            :MAX_LENGTH
        ]


    # Pad short reviews
    if len(sequence) < MAX_LENGTH:

        sequence = sequence + [
            vocab[PAD_TOKEN]
        ] * (
            MAX_LENGTH - len(sequence)
        )


    return sequence


train_sequences = [
    pad_sequence(sequence)
    for sequence in train_sequences
]


val_sequences = [
    pad_sequence(sequence)
    for sequence in val_sequences
]


test_sequences = [
    pad_sequence(sequence)
    for sequence in test_sequences
]


print("\nSequences padded successfully!")

print(
    "Sequence length:",
    len(train_sequences[0])
)


# ============================================================
# 12. CREATE PYTORCH DATASET
# ============================================================

class ReviewDataset(Dataset):

    def __init__(
        self,
        sequences,
        labels
    ):

        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        return (
            self.sequences[index],
            self.labels[index]
        )


train_dataset = ReviewDataset(
    train_sequences,
    labels_train
)


val_dataset = ReviewDataset(
    val_sequences,
    labels_val
)


test_dataset = ReviewDataset(
    test_sequences,
    labels_test
)


# ============================================================
# 13. CREATE DATA LOADERS
# ============================================================

BATCH_SIZE = 64


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nDataLoaders created successfully!")


# ============================================================
# 14. BUILD LSTM MODEL
# ============================================================

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()


        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )


        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )


        # Fully connected layer
        self.fc = nn.Linear(
            hidden_dim,
            1
        )


        # Dropout
        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        # Convert integer sequences to embeddings
        embedded = self.embedding(x)


        # LSTM
        output, (hidden, cell) = self.lstm(
            embedded
        )


        # Take final hidden state
        hidden = hidden[-1]


        # Dropout
        hidden = self.dropout(hidden)


        # Fully connected layer
        output = self.fc(hidden)


        return output.squeeze(1)


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
)


model = model.to(device)


print("\n" + "=" * 70)
print("LSTM MODEL")
print("=" * 70)

print(model)


# ============================================================
# 16. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=5
):

    train_losses = []
    val_losses = []


    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        model.train()

        total_train_loss = 0


        for sequences, labels in train_loader:

            sequences = sequences.to(device)

            labels = labels.to(device)


            # Clear gradients
            optimizer.zero_grad()


            # Forward pass
            predictions = model(
                sequences
            )


            # Calculate loss
            loss = criterion(
                predictions,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            total_train_loss += (
                loss.item()
                * sequences.size(0)
            )


        average_train_loss = (
            total_train_loss
            /
            len(train_loader.dataset)
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0


        with torch.no_grad():

            for sequences, labels in val_loader:

                sequences = sequences.to(device)

                labels = labels.to(device)


                predictions = model(
                    sequences
                )


                loss = criterion(
                    predictions,
                    labels
                )


                total_val_loss += (
                    loss.item()
                    * sequences.size(0)
                )


        average_val_loss = (
            total_val_loss
            /
            len(val_loader.dataset)
        )


        train_losses.append(
            average_train_loss
        )

        val_losses.append(
            average_val_loss
        )


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.4f} "
            f"Validation Loss: {average_val_loss:.4f}"
        )


    return train_losses, val_losses


# ============================================================
# 18. TRAIN THE MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM MODEL")
print("=" * 70)


EPOCHS = 5


train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS
)


# ============================================================
# 19. PLOT TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 20. EVALUATE MODEL ON TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)


model.eval()


all_predictions = []
all_actual = []


with torch.no_grad():

    for sequences, labels in test_loader:

        sequences = sequences.to(device)


        outputs = model(
            sequences
        )


        probabilities = torch.sigmoid(
            outputs
        )


        predictions = (
            probabilities >= 0.5
        ).int()


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_actual.extend(
            labels.numpy().astype(int)
        )


# ============================================================
# 21. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_actual,
    all_predictions
)


precision = precision_score(
    all_actual,
    all_predictions,
    zero_division=0
)


recall = recall_score(
    all_actual,
    all_predictions,
    zero_division=0
)


f1 = f1_score(
    all_actual,
    all_predictions,
    zero_division=0
)


print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)


print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predictions
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


print(cm)


# ============================================================
# 24. PREDICT SENTIMENT FOR NEW REVIEW
# ============================================================

def predict_sentiment(
    review
):

    # Clean text
    cleaned = clean_text(
        review
    )


    # Tokenize
    tokens = tokenize(
        cleaned
    )


    # Convert to integers
    sequence = text_to_sequence(
        tokens
    )


    # Pad
    sequence = pad_sequence(
        sequence
    )


    # Convert to tensor
    tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)


    # Evaluation mode
    model.eval()


    with torch.no_grad():

        output = model(
            tensor
        )


        probability = torch.sigmoid(
            output
        ).item()


    # Determine sentiment
    if probability >= 0.5:

        sentiment = "Positive"

        confidence = probability

    else:

        sentiment = "Negative"

        confidence = 1 - probability


    return sentiment, confidence


# ============================================================
# 25. TEST WITH ASSIGNMENT EXAMPLE
# ============================================================

example_review = (
    "The movie was excellent and very enjoyable."
)


sentiment, confidence = predict_sentiment(
    example_review
)


print("\n" + "=" * 70)
print("SAMPLE SENTIMENT PREDICTION")
print("=" * 70)


print(
    "\nReview:",
    example_review
)


print(
    "\nPredicted Sentiment:",
    sentiment
)


print(
    f"Confidence: {confidence * 100:.2f}%"
)


# ============================================================
# 26. INTERACTIVE SENTIMENT ANALYZER
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE SENTIMENT ANALYZER")
print("=" * 70)


print(
    "\nEnter a movie review."
)

print(
    "Type 'exit' to stop."
)


while True:

    user_review = input(
        "\nEnter review: "
    )


    if user_review.lower().strip() == "exit":

        print(
            "\nSentiment Analyzer stopped."
        )

        break


    if user_review.strip() == "":

        print(
            "Please enter a review."
        )

        continue


    sentiment, confidence = (
        predict_sentiment(
            user_review
        )
    )


    print(
        "\nPredicted Sentiment:",
        sentiment
    )


    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


# ============================================================
# 27. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "imdb_lstm_sentiment_model.pth"
)


print(
    "\nModel saved as:"
    " imdb_lstm_sentiment_model.pth"
)


# ============================================================
# 28. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL REPORT")
print("=" * 70)


print(
    f"\nDataset           : IMDb 50K"
)

print(
    f"Vocabulary size   : {len(vocab)}"
)

print(
    f"Maximum sequence  : {MAX_LENGTH}"
)

print(
    f"Embedding size    : 128"
)

print(
    f"LSTM hidden size  : 128"
)

print(
    f"Number of epochs  : {EPOCHS}"
)

print(
    f"\nAccuracy          : {accuracy * 100:.2f}%"
)

print(
    f"Precision         : {precision * 100:.2f}%"
)

print(
    f"Recall            : {recall * 100:.2f}%"
)

print(
    f"F1-Score          : {f1 * 100:.2f}%"
)

print(
    "\nTraining/Validation loss graph displayed above."
)

print(
    "\nLSTM Customer Review Sentiment Analyzer completed!"
)
