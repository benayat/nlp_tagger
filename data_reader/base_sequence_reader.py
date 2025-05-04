from abc import ABC, abstractmethod

import numpy as np

class BaseSequenceReader(ABC):
    def __init__(self, train_file_path, dev_file_path, pad_token="<PAD>", unk_token="<UNK>", word_to_vec=None):
        self.word_to_vec = word_to_vec
        self.pad_token = pad_token
        self.unk_token = unk_token if word_to_vec is None else "UUUNKKK"
        self.pad_tag = "PAD"
        self.sentences_train = self._read_and_pad_sentences(train_file_path)
        self.sentences_dev = self._read_and_pad_sentences(dev_file_path)
        self.vocab = self._build_vocab(self.sentences_train+self.sentences_dev)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.tag_vocab = self._build_tag_vocab(self.sentences_train+self.sentences_dev)
        self.embeddings_matrix = self._build_embeddings_matrix(word_to_vec, self.word_to_index)
        self.tag_to_index = {tag: i for i, tag in enumerate(self.tag_vocab)}
        self.index_to_tag = {index: tag for tag, index in self.tag_to_index.items()}

    def get_tag_from_index(self, index):
        return self.index_to_tag.get(index, None)  # defaults to None

    @abstractmethod
    def _read_and_pad_sentences(self, file_path, is_test=False):
        pass

    def _build_vocab(self, sentences):
        unique_words = {word for sent in sentences for word, _ in sent}
        unique_words.update([self.pad_token, self.unk_token])
        return sorted(unique_words)

    @staticmethod
    def _build_tag_vocab(sentences):
        unique_tags = {tag for sent in sentences for _, tag in sent}
        return sorted(unique_tags)

    def _build_embeddings_matrix(self, word_to_vec, word_to_index, embedding_dim=50):
        embedding_matrix = np.random.normal(scale=0.1, size=(len(word_to_vec), embedding_dim)).astype(np.float32)
        for word, index in word_to_index.items():
            key = word.lower()
            if key in word_to_vec:
                embedding_matrix[index] = word_to_vec[key]
            elif self.unk_token in word_to_vec:
                embedding_matrix[index] = word_to_vec[self.unk_token]
        return embedding_matrix


    def generate_windows(self, source="train", is_test=False):
        unk_idx = self.word_to_index[self.unk_token]
        if source == "train":
            sentences = self.sentences_train
        elif source == "dev":
            sentences = self.sentences_dev
        else:
            sentences = self._read_and_pad_sentences(source, is_test)
        if is_test:
            for sentence in sentences:
                for i in range(2, len(sentence) - 2):
                    center_word = sentence[i]
                    if center_word == self.pad_token:
                        continue
                    window_words = [sentence[j] for j in range(i - 2, i + 3)]
                    word_indices = [self.word_to_index.get(w, unk_idx) for w in window_words]
                    yield word_indices
        else:
            for sentence in sentences:
                for i in range(2, len(sentence) - 2):
                    center_word = sentence[i][0]
                    if center_word == self.pad_token:
                        continue
                    window_words = [sentence[j][0] for j in range(i - 2, i + 3)]
                    word_indices = [self.word_to_index.get(w, unk_idx) for w in window_words]
                    tag_index = self.tag_to_index[sentence[i][1]]
                    yield word_indices, tag_index

