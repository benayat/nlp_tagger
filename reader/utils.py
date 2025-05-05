import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# from reader.base_reader import BaseReader


def read_blocks(path_to_file: str) -> list[str]:
    """Return list of sentence blocks separated originally by blank lines."""
    return pathlib.Path(path_to_file).read_text(encoding="utf-8").rstrip().split("\n\n")


def load_text_embeddings(vocabulary_path: str, path_vectors: str) -> dict[str, np.ndarray]:
    """Load word vectors from plain-text files into a dictionary."""
    vocabulary_words = [line.strip() for line in open(vocabulary_path, encoding="utf-8")]
    vector_matrix = np.loadtxt(path_vectors, dtype=np.float32)
    return {word: vector for word, vector in zip(vocabulary_words, vector_matrix)}


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


