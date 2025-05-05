import torch
from torch import nn

from constants.constants import WORD_VECTOR_SIZE
from reader.base_reader import BaseReader


class MlpTaggerSubwords(nn.Module):
    def __init__(self, reader: BaseReader,
                 hidden_units: int,
                 dropout_probability: float) -> None:
        super().__init__()
        prefix_vocab_size = len(reader.prefix_to_index)
        suffix_vocab_size = len(reader.suffix_to_index)
        number_of_tags = len(reader.tag_to_index)
        padding_word_index = reader.word_to_index[reader.padding_token]
        initial_embeddings = reader.word_embedding_matrix
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(initial_embeddings),
            freeze=False,
            padding_idx=padding_word_index
        )
        self.prefix_embeddings = nn.Embedding(prefix_vocab_size, WORD_VECTOR_SIZE, padding_idx=0)
        self.suffix_embeddings = nn.Embedding(suffix_vocab_size, WORD_VECTOR_SIZE, padding_idx=0)
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
