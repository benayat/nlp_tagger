from abc import ABC, abstractmethod

import numpy as np

from constants.constants import WORD_VECTOR_SIZE
from reader.utils import trigram_prefix, trigram_suffix, get_prefix_and_suffix_to_index_dicts


class BaseReader(ABC):
    """Handles padding, vocabularies, and sliding window generation."""

    def __init__(self, train_path: str, dev_path: str, test_path: str,
                 pretrained_vectors_word_to_vec: dict[str, np.ndarray] | None,
                 word_embeddings_dim: int = WORD_VECTOR_SIZE,
                 use_subwords_in_reader: bool = False,
                 use_charcnn: bool = False,
                 char_max_len: int = 15,
                 padding_token: str = "<PAD>") -> None:
        self.use_subwords = use_subwords_in_reader
        self.use_charcnn = use_charcnn
        self.padding_token = padding_token
        self.padding_tag = "PAD"
        self.unknown_token = "UUUNKKK"
        self.char_max_len = char_max_len
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

        if self.use_charcnn:
            self.char_to_idx = self._build_char_to_index_vocab()
        self.word_to_index = {word: idx for idx, word in enumerate(self.words_vocabulary)}
        all_tag_set = {tag for sentence in self.sentences_training + self.sentences_development
                       for _, tag in sentence}
        self.tag_to_index = {tag: idx for idx, tag in enumerate(all_tag_set)}
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}

        self.word_embedding_matrix = self._construct_embedding_matrix(len(self.words_vocabulary),
                                                                      embeddings_dim=word_embeddings_dim,
                                                                      use_pretrained_embeddings=pretrained_vectors_word_to_vec is not None)

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
    def _build_char_to_index_vocab(self):
        chars = sorted({char for word in self.words_vocabulary for char in word})
        # print("Characters:", chars)  # Debugging
        char_to_idx = {"#": 0, "<UNK_CHAR>": 1,
                       **{char: i + 2 for i, char in enumerate(chars) if char not in {"#", "<UNK_CHAR>"}}}
        # print("Char to Index:", char_to_idx)  # Debugging
        return char_to_idx

    def encode_word(self, token: str) -> tuple[int] | tuple[int, int, int] | tuple:
        """Return (word‑id, prefix‑id, suffix‑id) for one token."""
        unk_index_words = self.word_to_index[self.unknown_token]
        word_index = self.word_to_index.get(token, unk_index_words)
        if not self.use_subwords and not self.use_charcnn:
            return (word_index,)
        elif not self.use_charcnn:
            return (
                word_index,
                self.prefix_to_index.get(trigram_prefix(token), self.prefix_to_index["<UNK_PREF>"]),
                self.suffix_to_index.get(trigram_suffix(token), self.suffix_to_index["<UNK_SUF>"]),
            )
        else:
            if token == self.padding_token:
                return word_index, *([0] * self.char_max_len)

            chars = [self.char_to_idx.get(char, 1)  # 1 = UNK_CHAR
                     for char in token[:self.char_max_len]]
            chars += [0] * (self.char_max_len - len(chars))
            return word_index, *chars

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
                    yield encoded[0] if len(encoded) == 1 else tuple(list(component) for component in encoded)
                else:
                    tag_idx = self.tag_to_index[sentence[center_position][1]]
                    yield (list(encoded[0]) if len(encoded) == 1 else tuple(
                        list(component) for component in encoded)), tag_idx
