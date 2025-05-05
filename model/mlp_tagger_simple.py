import torch
from torch import nn

from constants.constants import WORD_VECTOR_SIZE
from reader.base_reader import BaseReader


class MlpTaggerSimple(nn.Module):
    def __init__(self, reader: BaseReader,
                 hidden_units: int,
                 dropout_probability: float) -> None:
        super().__init__()
        number_of_tags = len(reader.tag_to_index)
        padding_word_index = reader.word_to_index[reader.padding_token]
        initial_embeddings = reader.word_embedding_matrix
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
