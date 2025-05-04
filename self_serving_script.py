#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Two-phase MLP tagger (POS / NER)
#  • Phase-A: train embeddings, MLP frozen
#  • Phase-B: freeze embeddings, train MLP
#  • Independent learning-rate & weight-decay for each param-group
# ------------------------------------------------------------------
from __future__ import annotations
from abc import ABC, abstractmethod
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────── configuration ──────────────────────────────
TASK: str = "ner"  # choose "pos" or "ner"
DATASET_PREFIX: str = ""  # "" for local paths, "/kaggle/input/nlp-tagger-data/" for kaggle

WORD_VECTOR_SIZE: int = 50
GLOBAL_RANDOM_SEED: int = 42

FILE_LOCATIONS = {
    "pos": {
        "training": f"{DATASET_PREFIX}data/pos/train",
        "development": f"{DATASET_PREFIX}data/pos/dev",
        "test": f"{DATASET_PREFIX}data/pos/test",
    },
    "ner": {
        "training": f"{DATASET_PREFIX}data/ner/train",
        "development": f"{DATASET_PREFIX}data/ner/dev",
        "test": f"{DATASET_PREFIX}data/ner/test",
    },
    "embeddings": {
        "vocabulary": f"{DATASET_PREFIX}data/embeddings/vocab.txt",
        "vectors": f"{DATASET_PREFIX}data/embeddings/wordVectors.txt",
    },
}

HYPER_PARAMETER_SETS = {
    "pos": {
        "embedding_learning_rate": 1e-2,
        "embedding_weight_decay": 0,
        "classifier_learning_rate": 1e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-4,
        "training_epochs": 50,
        "number_of_warm_epochs": 6,
        "classifier_hidden_units": 64,
        "dropout_probability": 0.25,
        "training_batch_size": int(8192 * 4),
    },
    "ner": {
        "embedding_learning_rate": 6e-1,
        "embedding_weight_decay": 0.0,
        # "classifier_learning_rate": 3e-4,
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-5,
        "training_epochs": 40,
        "number_of_warm_epochs": 0,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.5,
        "training_batch_size": int(8192),
    },
}
CONFIG = HYPER_PARAMETER_SETS[TASK]


# ────────────────────────────── utilities ───────────────────────────────────
def read_blocks(path_to_file: str) -> list[str]:
    """Return list of sentence blocks separated originally by blank lines."""
    return pathlib.Path(path_to_file).read_text(encoding="utf-8").rstrip().split("\n\n")


def load_text_embeddings(vocabulary_path: str, path_vectors: str) -> dict[str, np.ndarray]:
    """Load word vectors from plain-text files into a dictionary."""
    vocabulary_words = [line.strip() for line in open(vocabulary_path, encoding="utf-8")]
    vector_matrix = np.loadtxt(path_vectors, dtype=np.float32)
    return {word: vector for word, vector in zip(vocabulary_words, vector_matrix)}


# ─────────────────────────── data-reader classes ────────────────────────────
class BaseSequenceReader(ABC):
    """Handles padding, vocabularies, and sliding window generation."""

    def __init__(self, train_path: str, dev_path: str, test_path: str,
                 pretrained_vectors_word_to_vec: dict[str, np.ndarray] | None,
                 use_subwords_in_reader: bool = False,
                 padding_token: str = "<PAD>") -> None:

        self.use_subwords = use_subwords_in_reader
        self.padding_token = padding_token
        self.padding_tag = "PAD"
        self.unknown_token = "UUUNKKK"
        self.pretrained_vectors_word_to_vec = pretrained_vectors_word_to_vec

        self.sentences_training = self._read_file(train_path)
        self.sentences_development = self._read_file(dev_path)
        self.sentences_test = self._read_file(test_path, is_test=True)
        test_words = {word for sentence in self.sentences_test for word in sentence if sentence}
        training_and_dev_words = {word for sentence in self.sentences_training + self.sentences_development
                                  for word, _ in sentence}
        self.words_vocabulary = sorted(training_and_dev_words | test_words |
                                       set(pretrained_vectors_word_to_vec) |
                                       {padding_token, self.unknown_token}) if pretrained_vectors_word_to_vec else \
            sorted(training_and_dev_words | test_words | {padding_token, self.unknown_token})
        if use_subwords_in_reader:
            self.prefix_to_index, self.suffix_to_index = get_prefix_and_suffix_to_index_dicts(self.words_vocabulary)
        self.word_to_index = {word: idx for idx, word in enumerate(self.words_vocabulary)}
        all_tag_set = {tag for sentence in self.sentences_training + self.sentences_development
                       for _, tag in sentence}
        self.tag_to_index = {tag: idx for idx, tag in enumerate(all_tag_set)}
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}

        self.word_embedding_matrix = self._construct_embedding_matrix(len(self.words_vocabulary),
                                                                      embeddings_dim=WORD_VECTOR_SIZE,
                                                                      use_pretrained_embeddings=pretrained_vectors_word_to_vec is not None)

    # ----------------------------------------------------------------------
    def _construct_embedding_matrix(self, num_embeddings, embeddings_dim=WORD_VECTOR_SIZE,
                                    use_pretrained_embeddings=False) -> np.ndarray:
        matrix = np.random.normal(0, 0.1,
                                  (num_embeddings, embeddings_dim)).astype(np.float32)
        if not use_pretrained_embeddings:
            return matrix

        vector_for_unknown = self.pretrained_vectors_word_to_vec.get(self.unknown_token,
                                                                     np.zeros(WORD_VECTOR_SIZE))
        for word, index in self.word_to_index.items():
            matrix[index] = self.pretrained_vectors_word_to_vec.get(word.lower(), vector_for_unknown)
        return matrix

    @abstractmethod
    def _read_file(self, path_to_file: str, is_test: bool = False) -> list[list[tuple[str, str]]] | list[list[str]]:
        ...

    # ----------------------------------------------------------------------
    def encode_word(self, token: str) -> tuple[int] | tuple[int, int, int]:
        """Return (word‑id, prefix‑id, suffix‑id) for one token."""
        unk_index_words = self.word_to_index[self.unknown_token]
        word_index = self.word_to_index.get(token, unk_index_words)
        if not self.use_subwords:
            return (word_index,)
        unk_index_prefixes = self.prefix_to_index["<UNK_PREF>"]
        unk_index_suffixes = self.suffix_to_index["<UNK_SUF>"]
        return (
            self.word_to_index.get(token, unk_index_words),
            self.prefix_to_index.get(trigram_prefix(token), unk_index_prefixes),
            self.suffix_to_index.get(trigram_suffix(token), unk_index_suffixes),
        )

    def sliding_windows(self, split: str = "train", test_mode: bool = False):
        """Yield 5-token index windows.  In test-mode only indices are yielded.
               * train/dev  →  ((word_ids, pref_ids, suf_ids), target_tag_idx)
               * test       →  (word_ids, pref_ids, suf_ids)
        """
        sentences = (
            self.sentences_training if split == "train"
            else self.sentences_development if split == "dev"
            else self.sentences_test
        )
        for sentence in sentences:
            for center_position in range(2, len(sentence) - 2):
                center_token = sentence[center_position][0] if not test_mode else sentence[center_position]
                if center_token == self.padding_token:
                    continue
                window_tokens = [
                    sentence[i][0] if not test_mode else sentence[i]
                    for i in range(center_position - 2, center_position + 3)
                ]

                encoded = list(zip(*[self.encode_word(token) for token in window_tokens]))
                if test_mode:
                    yield [list(x) for x in encoded] if self.use_subwords else encoded[0]
                else:
                    tag_idx = self.tag_to_index[sentence[center_position][1]]
                    if self.use_subwords:
                        yield tuple(list(x) for x in encoded), tag_idx
                    else:
                        yield list(encoded[0]), tag_idx


# helper for both readers ----------------------------------------------------
def trigram_prefix(word):  # pad if word shorter than 3 chars
    return (word[:3] if len(word) >= 3 else word + "#" * (3 - len(word))).lower()


def trigram_suffix(word):
    return (word[-3:] if len(word) >= 3 else "#" * (3 - len(word)) + word).lower()


def get_prefix_and_suffix_to_index_dicts(corpus):
    all_prefixes = sorted({trigram_prefix(word) for word in corpus})
    all_suffixes = sorted({trigram_suffix(word) for word in corpus})
    return {"<PAD_PREF>": 0, "<UNK_PREF>": 1, **{prefix: i + 2 for i, prefix in enumerate(all_prefixes)}}, {
        "<PAD_SUF>": 0, "<UNK_SUF>": 1, **{suffix: i + 2 for i, suffix in enumerate(all_suffixes)}}


def convert_blocks_to_sentences(blocks: list[str], padding_token: str,
                                skip_docstart_lines: bool, is_test: bool = False) -> list[list[tuple[str, str]]] | list[
    list[str]]:
    if is_test:
        return [[line for line in block.splitlines() if line and not line.startswith("-DOCSTART-")] for block in blocks]
    sentences = []
    for block in blocks:
        tokens_with_tags: list[tuple[str, str]] = []
        for line in block.splitlines():
            if (skip_docstart_lines and line.startswith("-DOCSTART-")) or not line.strip():
                continue
            parts = line.split()
            token = parts[0]
            tag = parts[-1] if len(parts) > 1 else "O"
            tokens_with_tags.append((token, tag))
        sentences.append(
            [(padding_token, "PAD")] * 2 + tokens_with_tags + [(padding_token, "PAD")] * 2
        )
    return sentences


class PosReader(BaseSequenceReader):
    def _read_file(self, path_to_file: str, is_test: bool = False):
        return convert_blocks_to_sentences(
            read_blocks(path_to_file), self.padding_token, skip_docstart_lines=False,
            is_test=is_test
        )


class NerReader(BaseSequenceReader):
    def _read_file(self, path_to_file: str, is_test: bool = False):
        return convert_blocks_to_sentences(
            read_blocks(path_to_file), self.padding_token, skip_docstart_lines=True,
            is_test=is_test
        )


# ───────────────────────────── model class ──────────────────────────────────
class MlpTaggerSimple(nn.Module):
    def __init__(self,
                 hidden_units: int,
                 number_of_tags: int,
                 padding_word_index: int,
                 initial_embeddings: np.ndarray,
                 dropout_probability: float) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor(initial_embeddings),
            freeze=False,
            padding_idx=padding_word_index
        )
        self.classifier_pipeline = nn.Sequential(
            nn.Linear(WORD_VECTOR_SIZE * 5, hidden_units),
            nn.Tanh(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_units, number_of_tags),
        )

    def forward(self, five_token_indices: torch.Tensor) -> torch.Tensor:
        embedded_tokens = self.embedding_layer(five_token_indices)  # (batch, 5, 50)
        flattened_input = embedded_tokens.flatten(1)  # (batch, 250)
        return self.classifier_pipeline(flattened_input)  # (batch, num_tags)


class MlpTaggerSum(nn.Module):
    def __init__(self,
                 prefix_vocab_size: int,
                 suffix_vocab_size: int,
                 hidden_units: int,
                 number_of_tags: int,
                 padding_word_index: int,
                 initial_embeddings: np.ndarray,
                 dropout_probability: float) -> None:
        super().__init__()
        shared_dim = initial_embeddings.shape[1]
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(initial_embeddings),
            freeze=False,
            padding_idx=padding_word_index
        )
        self.prefix_embeddings = nn.Embedding(prefix_vocab_size, shared_dim, padding_idx=0)
        self.suffix_embeddings = nn.Embedding(suffix_vocab_size, shared_dim, padding_idx=0)
        self.classifier_pipeline = nn.Sequential(
            nn.Linear(WORD_VECTOR_SIZE * 5, hidden_units),
            nn.Tanh(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_units, number_of_tags),
        )

    def forward(self, packed_ids: torch.Tensor) -> torch.Tensor:
        words_ids = packed_ids[:, 0, :]
        prefixes_ids = packed_ids[:, 1, :]
        suffixes_ids = packed_ids[:, 2, :]
        words_embeddings = self.word_embeddings(words_ids)
        prefixes_embeddings = self.prefix_embeddings(prefixes_ids)
        suffixes_embeddings = self.suffix_embeddings(suffixes_ids)
        sum_embeddings = prefixes_embeddings + words_embeddings + suffixes_embeddings  # (batch, 5, 50)
        flattened_input = sum_embeddings.flatten(1)  # (batch, 250)
        return self.classifier_pipeline(flattened_input)  # (batch, num_tags)


# ───────────────────────── training utilities ───────────────────────────────
def calculate_masked_accuracy(predicted_indices: torch.Tensor,
                              gold_indices: torch.Tensor,
                              index_of_padding_tag: int,
                              index_of_outside_tag: int | None = None) -> tuple[int, int]:
    valid_mask = gold_indices != index_of_padding_tag
    if index_of_outside_tag is not None:
        valid_mask &= ~((gold_indices == index_of_outside_tag) &
                        (predicted_indices == index_of_outside_tag))
    correct_predictions = ((predicted_indices == gold_indices) & valid_mask).sum().item()
    total_considered = valid_mask.sum().item()
    return correct_predictions, total_considered


def execute_epoch(model: nn.Module,
                  data_loader: DataLoader,
                  loss_function: nn.CrossEntropyLoss,
                  optimiser: torch.optim.Optimizer | None,
                  padding_tag_index: int,
                  outside_tag_index: int | None,
                  is_training: bool) -> tuple[float, float]:
    model.train() if is_training else model.eval()
    cumulative_loss = correct_total = token_total = 0
    for batch_inputs, batch_targets in data_loader:
        batch_inputs, batch_targets = batch_inputs.to(COMPUTE_DEVICE), batch_targets.to(COMPUTE_DEVICE)
        if is_training:
            optimiser.zero_grad()  # type: ignore[arg-type]
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_targets)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()  # type: ignore[union-attr]
        correct, considered = calculate_masked_accuracy(
            logits.argmax(1), batch_targets, padding_tag_index, outside_tag_index
        )
        cumulative_loss += loss.item() * batch_inputs.size(0)
        correct_total += correct
        token_total += considered
    average_loss = cumulative_loss / token_total
    accuracy = correct_total / token_total
    return average_loss, accuracy


# ───────────────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(GLOBAL_RANDOM_SEED)
    np.random.seed(GLOBAL_RANDOM_SEED)
    COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare embeddings and reader
    pretrained_dictionary = load_text_embeddings(
        FILE_LOCATIONS["embeddings"]["vocabulary"],
        FILE_LOCATIONS["embeddings"]["vectors"]
    )
    ReaderClass = PosReader if TASK == "pos" else NerReader
    data_reader = ReaderClass(
        FILE_LOCATIONS[TASK]["training"],
        FILE_LOCATIONS[TASK]["development"],
        FILE_LOCATIONS[TASK]["test"],
        # pretrained_dictionary,
        None,
        use_subwords_in_reader=True
    )


    # 2. Create data loaders
    def build_data_loader(split_label: str, shuffle_flag: bool) -> DataLoader:
        windows_list, tag_indices = zip(*(data_reader.sliding_windows(split_label)))
        return DataLoader(
            TensorDataset(torch.tensor(windows_list), torch.tensor(tag_indices)),
            batch_size=CONFIG["training_batch_size"],
            shuffle=shuffle_flag,
            num_workers=4,
        )


    loader_train = build_data_loader("train", shuffle_flag=True)
    loader_dev = build_data_loader("dev", shuffle_flag=False)

    # 3. Build model instance
    model_instance = MlpTaggerSum(
        prefix_vocab_size=len(data_reader.prefix_to_index),
        suffix_vocab_size=len(data_reader.suffix_to_index),
        hidden_units=CONFIG["classifier_hidden_units"],
        number_of_tags=len(data_reader.tag_to_index),
        padding_word_index=data_reader.word_to_index[data_reader.padding_token],
        initial_embeddings=data_reader.word_embedding_matrix,
        dropout_probability=CONFIG["dropout_probability"]
    ).to(COMPUTE_DEVICE)

    embedding_parameter_group = [
        parameter for name, parameter in model_instance.named_parameters()
        if name.startswith("embedding_layer")
    ]
    classifier_parameter_group = [
        parameter for name, parameter in model_instance.named_parameters()
        if name.startswith("classifier_pipeline")
    ]

    loss_function_main = nn.CrossEntropyLoss(
        ignore_index=data_reader.tag_to_index[data_reader.padding_tag]
    )
    outside_index_for_ner = data_reader.tag_to_index.get("O") if TASK == "ner" else None

    # ─── Phase A: train embeddings only ───
    optimiser_phase_a = torch.optim.Adam([
        {"params": embedding_parameter_group,
         "lr": CONFIG["embedding_learning_rate"],
         "weight_decay": CONFIG["embedding_weight_decay"]},
        {"params": classifier_parameter_group,
         "lr": 0.0,
         "weight_decay": 0.0}
    ])
    print("\n=== Phase A : Embedding warm-up ===")
    for epoch_id in range(1, CONFIG['number_of_warm_epochs'] + 1):
        train_loss, train_accuracy = execute_epoch(
            model_instance, loader_train, loss_function_main, optimiser_phase_a,
            data_reader.tag_to_index[data_reader.padding_tag],
            outside_index_for_ner, is_training=True
        )
        dev_loss, dev_accuracy = execute_epoch(
            model_instance, loader_dev, loss_function_main, None,
            data_reader.tag_to_index[data_reader.padding_tag],
            outside_index_for_ner, is_training=False
        )
        print(f"Warm {epoch_id:02d} | "
              f"train {train_loss:.5f} / {train_accuracy * 100:5.2f}% | "
              f"dev {dev_loss:.5f} / {dev_accuracy * 100:5.2f}%")

    # ─── Phase B: freeze embeddings, train classifier ───
    # for parameter in embedding_parameter_group:
    #     parameter.requires_grad = False

    optimiser_phase_b = torch.optim.Adam([
        {"params": embedding_parameter_group,
         # "lr": 0.0, "weight_decay": 0.0},
         "lr": CONFIG["classifier_learning_rate"], "weight_decay": CONFIG["classifier_weight_decay"]},
        {"params": classifier_parameter_group,
         "lr": CONFIG["classifier_learning_rate"],
         "weight_decay": CONFIG["classifier_weight_decay"]}
    ])
    scheduler_learning_rate = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser_phase_b,
        mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_decay_factor"],
        min_lr=CONFIG["scheduler_min_lr"]
    )
    previous_learning_rate_value = optimiser_phase_b.param_groups[1]["lr"]

    print("\n=== Phase B : Classifier fine-tune ===")
    for epoch_id in range(1, CONFIG["training_epochs"] + 1):
        train_loss, train_accuracy = execute_epoch(
            model_instance, loader_train, loss_function_main, optimiser_phase_b,
            data_reader.tag_to_index[data_reader.padding_tag],
            outside_index_for_ner, is_training=True
        )
        dev_loss, dev_accuracy = execute_epoch(
            model_instance, loader_dev, loss_function_main, None,
            data_reader.tag_to_index[data_reader.padding_tag],
            outside_index_for_ner, is_training=False
        )

        scheduler_learning_rate.step(dev_loss)
        current_learning_rate_value = optimiser_phase_b.param_groups[1]["lr"]
        if current_learning_rate_value != previous_learning_rate_value:
            scaling_factor = current_learning_rate_value / previous_learning_rate_value
            optimiser_phase_b.param_groups[1]["weight_decay"] *= scaling_factor
            previous_learning_rate_value = current_learning_rate_value
            print(f"Learning-rate decayed → {current_learning_rate_value:.1e}; "
                  f"weight-decay scaled × {scaling_factor:.3f}")

        print(f"Fine {epoch_id:02d} | "
              f"train {train_loss:.5f} / {train_accuracy * 100:5.2f}% | "
              f"dev {dev_loss:.5f} / {dev_accuracy * 100:5.2f}% | "
              f"lr {current_learning_rate_value:.1e}")

    # ─── Inference on test set ───
    test_windows_list = list(data_reader.sliding_windows(
        FILE_LOCATIONS[TASK]["test"], test_mode=True)
    )
    test_tensor_input = torch.tensor(test_windows_list)
    logits_test = model_instance(test_tensor_input.to(COMPUTE_DEVICE))
    predicted_tag_indices = logits_test.argmax(1).cpu().tolist()

    output_filename = f"test.{TASK}"
    with open(output_filename, "w", encoding="utf-8") as file_handle:
        for index in predicted_tag_indices:
            file_handle.write(data_reader.index_to_tag[index] + "\n")
    print(f"\nPredictions stored in file: {output_filename}")
