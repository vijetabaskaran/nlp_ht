# ============================================================
# ATTENTION-BASED ENGLISH-TAMIL TRANSLATOR
# ============================================================

!pip install -q torch pandas numpy matplotlib requests

import re
import random
import requests
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. DOWNLOAD ENGLISH-TAMIL DATASET
# ============================================================

english_url = (
    "https://raw.githubusercontent.com/"
    "nlpcuom/English-Tamil-Parallel-Corpus/"
    "master/En-Ta%20Corpus/En-Ta%20English.txt"
)

tamil_url = (
    "https://raw.githubusercontent.com/"
    "nlpcuom/English-Tamil-Parallel-Corpus/"
    "master/En-Ta%20Corpus/En-Ta%20Tamil.txt"
)


def download_file(url, filename):

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Download failed: {response.status_code}"
        )

    with open(filename, "wb") as f:
        f.write(response.content)

    print(filename, "downloaded successfully")


download_file(
    english_url,
    "english.txt"
)

download_file(
    tamil_url,
    "tamil.txt"
)


# ============================================================
# 3. READ DATASET
# ============================================================

with open(
    "english.txt",
    "r",
    encoding="utf-8"
) as f:

    english_lines = f.read().splitlines()


with open(
    "tamil.txt",
    "r",
    encoding="utf-8"
) as f:

    tamil_lines = f.read().splitlines()


print("English lines:", len(english_lines))
print("Tamil lines:", len(tamil_lines))


# The original files contain header information.
# Remove the first 3 lines.

english_lines = english_lines[3:]
tamil_lines = tamil_lines[3:]


# Make sure both files have the same number of sentences.

num_pairs = min(
    len(english_lines),
    len(tamil_lines)
)


english_lines = english_lines[:num_pairs]
tamil_lines = tamil_lines[:num_pairs]


print("Sentence pairs:", num_pairs)


# ============================================================
# 4. LIMIT DATA FOR CPU TRAINING
# ============================================================

MAX_SAMPLES = 5000

english_lines = english_lines[:MAX_SAMPLES]
tamil_lines = tamil_lines[:MAX_SAMPLES]


# ============================================================
# 5. CLEANING
# ============================================================

def clean_english(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z?.!,']+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_tamil(text):

    text = str(text)

    text = re.sub(
        r"[^\u0B80-\u0BFF\s?.!,']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


english_lines = [
    clean_english(x)
    for x in english_lines
]

tamil_lines = [
    clean_tamil(x)
    for x in tamil_lines
]


# ============================================================
# 6. REMOVE EMPTY SENTENCES
# ============================================================

pairs = []

for english, tamil in zip(
    english_lines,
    tamil_lines
):

    if english and tamil:

        pairs.append(
            (english, tamil)
        )


print("Valid sentence pairs:", len(pairs))


# ============================================================
# 7. SHOW SAMPLE DATA
# ============================================================

print("\nSample sentences:\n")

for english, tamil in pairs[:5]:

    print("English:", english)
    print("Tamil:", tamil)
    print()


# ============================================================
# 8. TOKENIZATION
# ============================================================

def tokenize_english(text):

    return text.split()


def tokenize_tamil(text):

    return text.split()


english_tokens = [
    tokenize_english(x[0])
    for x in pairs
]

tamil_tokens = [
    tokenize_tamil(x[1])
    for x in pairs
]


# ============================================================
# 9. SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    START_TOKEN,
    END_TOKEN
]


# ============================================================
# 10. BUILD VOCABULARY
# ============================================================

def build_vocab(sentences):

    counter = Counter()

    for sentence in sentences:

        counter.update(sentence)


    vocab = {}

    for token in SPECIAL_TOKENS:

        vocab[token] = len(vocab)


    for word, count in counter.items():

        if word not in vocab:

            vocab[word] = len(vocab)


    return vocab


english_vocab = build_vocab(
    english_tokens
)

tamil_vocab = build_vocab(
    tamil_tokens
)


english_itos = {
    index: word
    for word, index in english_vocab.items()
}

tamil_itos = {
    index: word
    for word, index in tamil_vocab.items()
}


print("English vocabulary:", len(english_vocab))
print("Tamil vocabulary:", len(tamil_vocab))


# ============================================================
# 11. ENCODE SENTENCES
# ============================================================

def encode_sentence(
    tokens,
    vocab
):

    tokens = (
        [START_TOKEN]
        + tokens
        + [END_TOKEN]
    )

    return [
        vocab.get(
            token,
            vocab[UNK_TOKEN]
        )
        for token in tokens
    ]


encoded_pairs = []

for english, tamil in pairs:

    source = encode_sentence(
        tokenize_english(english),
        english_vocab
    )

    target = encode_sentence(
        tokenize_tamil(tamil),
        tamil_vocab
    )

    encoded_pairs.append(
        (source, target)
    )


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

random.seed(42)

random.shuffle(encoded_pairs)


split_index = int(
    len(encoded_pairs) * 0.9
)


train_pairs = encoded_pairs[
    :split_index
]

test_pairs = encoded_pairs[
    split_index:
]


print(
    "\nTraining pairs:",
    len(train_pairs)
)

print(
    "Testing pairs:",
    len(test_pairs)
)


# ============================================================
# 13. PYTORCH DATASET
# ============================================================

class TranslationDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

        source, target = self.pairs[index]

        return (
            torch.tensor(
                source,
                dtype=torch.long
            ),
            torch.tensor(
                target,
                dtype=torch.long
            )
        )


# ============================================================
# 14. PADDING
# ============================================================

def collate_fn(batch):

    sources = [
        item[0]
        for item in batch
    ]

    targets = [
        item[1]
        for item in batch
    ]


    sources = pad_sequence(
        sources,
        batch_first=True,
        padding_value=english_vocab[
            PAD_TOKEN
        ]
    )


    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=tamil_vocab[
            PAD_TOKEN
        ]
    )


    return sources, targets


train_dataset = TranslationDataset(
    train_pairs
)

test_dataset = TranslationDataset(
    test_pairs
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# 15. ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()


        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=english_vocab[
                PAD_TOKEN
            ]
        )


        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )


    def forward(self, source):

        embedded = self.embedding(
            source
        )


        outputs, (
            hidden,
            cell
        ) = self.lstm(
            embedded
        )


        return (
            outputs,
            hidden,
            cell
        )


# ============================================================
# 16. BAHDANAU ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        encoder_hidden_dim,
        decoder_hidden_dim
    ):

        super().__init__()


        self.attention = nn.Linear(
            encoder_hidden_dim
            + decoder_hidden_dim,
            decoder_hidden_dim
        )


        self.v = nn.Linear(
            decoder_hidden_dim,
            1,
            bias=False
        )


    def forward(
        self,
        decoder_hidden,
        encoder_outputs
    ):

        source_length = (
            encoder_outputs.shape[1]
        )


        decoder_hidden = (
            decoder_hidden[-1]
        )


        decoder_hidden = (
            decoder_hidden.unsqueeze(1)
        )


        decoder_hidden = (
            decoder_hidden.repeat(
                1,
                source_length,
                1
            )
        )


        energy = torch.tanh(
            self.attention(
                torch.cat(
                    (
                        decoder_hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )


        attention = self.v(
            energy
        ).squeeze(2)


        return torch.softmax(
            attention,
            dim=1
        )


# ============================================================
# 17. DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        attention
    ):

        super().__init__()


        self.output_dim = output_dim

        self.attention = attention


        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=tamil_vocab[
                PAD_TOKEN
            ]
        )


        self.lstm = nn.LSTM(
            embedding_dim
            + encoder_hidden_dim,
            decoder_hidden_dim,
            batch_first=True
        )


        self.fc_out = nn.Linear(
            decoder_hidden_dim
            + encoder_hidden_dim
            + embedding_dim,
            output_dim
        )


    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs
    ):

        input_token = (
            input_token.unsqueeze(1)
        )


        embedded = self.embedding(
            input_token
        )


        attention_weights = (
            self.attention(
                hidden,
                encoder_outputs
            )
        )


        attention_weights = (
            attention_weights.unsqueeze(1)
        )


        context = torch.bmm(
            attention_weights,
            encoder_outputs
        )


        lstm_input = torch.cat(
            (
                embedded,
                context
            ),
            dim=2
        )


        output, (
            hidden,
            cell
        ) = self.lstm(
            lstm_input,
            (hidden, cell)
        )


        prediction = self.fc_out(
            torch.cat(
                (
                    output,
                    context,
                    embedded
                ),
                dim=2
            )
        )


        return (
            prediction.squeeze(1),
            hidden,
            cell,
            attention_weights.squeeze(1)
        )


# ============================================================
# 18. SEQ2SEQ MODEL
# ============================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.device = device


    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.shape[0]

        target_length = target.shape[1]

        target_vocab_size = (
            self.decoder.output_dim
        )


        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=self.device
        )


        encoder_outputs, hidden, cell = (
            self.encoder(source)
        )


        input_token = target[:, 0]


        for t in range(
            1,
            target_length
        ):


            output, hidden, cell, _ = (
                self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )


            outputs[:, t] = output


            teacher_force = (
                random.random()
                < teacher_forcing_ratio
            )


            best_prediction = (
                output.argmax(1)
            )


            input_token = (
                target[:, t]
                if teacher_force
                else best_prediction
            )


        return outputs


# ============================================================
# 19. CREATE MODEL
# ============================================================

INPUT_DIM = len(
    english_vocab
)

OUTPUT_DIM = len(
    tamil_vocab
)


ENC_EMB_DIM = 128
DEC_EMB_DIM = 128

ENC_HID_DIM = 256
DEC_HID_DIM = 256


attention = Attention(
    ENC_HID_DIM,
    DEC_HID_DIM
)


encoder = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    ENC_HID_DIM
)


decoder = Decoder(
    OUTPUT_DIM,
    DEC_EMB_DIM,
    ENC_HID_DIM,
    DEC_HID_DIM,
    attention
)


model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)


print("\nModel created successfully")


# ============================================================
# 20. LOSS AND OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


criterion = nn.CrossEntropyLoss(
    ignore_index=tamil_vocab[
        PAD_TOKEN
    ]
)


# ============================================================
# 21. TRAINING FUNCTION
# ============================================================

def train(model, loader):

    model.train()

    total_loss = 0


    for source, target in loader:

        source = source.to(device)

        target = target.to(device)


        optimizer.zero_grad()


        output = model(
            source,
            target,
            teacher_forcing_ratio=0.5
        )


        output_dim = output.shape[-1]


        output = output[:, 1:].reshape(
            -1,
            output_dim
        )


        target = target[:, 1:].reshape(
            -1
        )


        loss = criterion(
            output,
            target
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )


        optimizer.step()


        total_loss += loss.item()


    return (
        total_loss
        / len(loader)
    )


# ============================================================
# 22. TRAIN MODEL
# ============================================================

EPOCHS = 5

loss_history = []


print("\nTraining started...\n")


for epoch in range(EPOCHS):

    loss = train(
        model,
        train_loader
    )


    loss_history.append(loss)


    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" - Loss: {loss:.4f}"
    )


print("\nTraining completed!")


# ============================================================
# 23. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    range(
        1,
        EPOCHS + 1
    ),
    loss_history,
    marker="o"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training Loss"
)

plt.title(
    "Training Loss"
)


plt.grid()

plt.show()


# ============================================================
# 24. TRANSLATION FUNCTION
# ============================================================

def translate_sentence(
    sentence,
    model,
    max_length=30
):

    model.eval()


    tokens = tokenize_english(
        clean_english(sentence)
    )


    source_indices = (
        encode_sentence(
            tokens,
            english_vocab
        )
    )


    source_tensor = torch.tensor(
        source_indices,
        dtype=torch.long
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        encoder_outputs, hidden, cell = (
            model.encoder(
                source_tensor
            )
        )


    input_token = torch.tensor(
        [
            tamil_vocab[
                START_TOKEN
            ]
        ],
        dtype=torch.long
    ).to(device)


    translated_tokens = []

    attention_values = []


    for _ in range(max_length):


        with torch.no_grad():

            output, hidden, cell, attention_weights = (
                model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )


        prediction = (
            output.argmax(1).item()
        )


        attention_values.append(
            attention_weights[
                0
            ].cpu().numpy()
        )


        if prediction == tamil_vocab[
            END_TOKEN
        ]:

            break


        if prediction == tamil_vocab[
            PAD_TOKEN
        ]:

            break


        translated_tokens.append(
            tamil_itos[
                prediction
            ]
        )


        input_token = torch.tensor(
            [prediction],
            dtype=torch.long
        ).to(device)


    return (
        translated_tokens,
        attention_values
    )


# ============================================================
# 25. ATTENTION VISUALIZATION
# ============================================================

def show_attention(
    sentence,
    translation,
    attention_values
):

    source_words = (
        [START_TOKEN]
        + tokenize_english(
            clean_english(sentence)
        )
        + [END_TOKEN]
    )


    if len(attention_values) == 0:

        print("No attention values available.")

        return


    attention_matrix = np.array(
        attention_values
    )


    attention_matrix = (
        attention_matrix[
            :len(translation),
            :len(source_words)
        ]
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.imshow(
        attention_matrix,
        aspect="auto"
    )


    plt.xticks(
        range(
            len(source_words)
        ),
        source_words,
        rotation=45
    )


    plt.yticks(
        range(
            len(translation)
        ),
        translation
    )


    plt.xlabel(
        "English Source Words"
    )


    plt.ylabel(
        "Generated Tamil Words"
    )


    plt.title(
        "Bahdanau Attention Visualization"
    )


    plt.colorbar(
        label="Attention Weight"
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# 26. TEST SENTENCE
# ============================================================

input_sentence = "How are you?"


translation, attention_values = (
    translate_sentence(
        input_sentence,
        model
    )
)


print("\n====================================")
print("ENGLISH:")
print(input_sentence)


print("\nGENERATED TAMIL:")

print(
    " ".join(translation)
)


print("====================================")


# ============================================================
# 27. SHOW ATTENTION
# ============================================================

show_attention(
    input_sentence,
    translation,
    attention_values
)


# ============================================================
# 28. TRY YOUR OWN SENTENCE
# ============================================================

while True:

    test_sentence = input(
        "\nEnter an English sentence "
        "(or type 'exit'): "
    )


    if test_sentence.lower() == "exit":

        print("Translator closed.")

        break


    translation, attention_values = (
        translate_sentence(
            test_sentence,
            model
        )
    )


    print("\nEnglish:")

    print(test_sentence)


    print("\nGenerated Tamil:")

    print(
        " ".join(translation)
    )


    show_attention(
        test_sentence,
        translation,
        attention_values
    )
