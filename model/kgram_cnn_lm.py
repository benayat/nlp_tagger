import torch
from torch import nn


class KGramCnnLm(nn.Module):
    def __init__(self, vocab_size: int, embeddings_dim: int,
                 n_filters: int = 128, kernel: int = 3,
                 p_drop: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embeddings_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(
            embeddings_dim, n_filters, kernel_size=kernel, padding=int(kernel / 2)
        )
        self.conv2 = nn.Conv1d(
            n_filters, n_filters, kernel_size=kernel, padding=int(kernel / 2)
        )
        self.projection = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(n_filters, vocab_size)
        )

    def forward(self, context):  # ctx: (B, k)
        e = self.emb(context).transpose(1, 2)  # (B, D, k)
        h1 = torch.relu(self.conv1(e))  # (B, F, k)
        h2 = torch.relu(self.conv2(h1) + h1)  # Residual connection (B, F, k)
        h = h2.max(dim=2).values  # (B, F)
        return self.projection(h)  # logits
