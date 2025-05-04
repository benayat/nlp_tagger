import torch
import torch.nn as nn


class MLPTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dropout_prob=0.2,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,  # Set to True if you don't want to fine-tune the embeddings
                padding_idx=pad_idx
            ) if pretrained_embeddings is not None else nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, 5, 50)
        flat = embedded.reshape(x.size(0), -1)  # (batch_size, 250)
        return self.mlp(flat)  # logits: (batch_size, num_classes)
