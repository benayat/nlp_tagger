import torch
import torch.nn as nn

class MLPPosTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)         # (batch_size, 5, 50)
        flat = embedded.view(x.size(0), -1)  # (batch_size, 250)
        return self.mlp(flat)                # logits: (batch_size, num_classes)