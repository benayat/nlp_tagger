from torch import nn

class KGramMlpLm(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int, k: int,
                 hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.emb   = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.ff    = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(k * emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vocab_sz)
        )

    def forward(self, ctx):           # ctx: (B, k)
        e = self.emb(ctx).flatten(1)  # (B, k*D)
        return self.ff(e)             # logits  (B, |V|)