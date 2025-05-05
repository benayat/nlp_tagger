import torch
from pathlib import Path

class CharLmReader:
    """
    Creates k‑character contexts and their next‑character target.
      • <PAD>=0  <UNK>=1  <BOS>=2  <EOS>=3
    """
    PAD, UNK, BOS, EOS = range(4)

    def __init__(self, path: str | Path, k: int) -> None:
        self.k = k

        text = Path(path).read_text(encoding="utf‑8")
        self.vocab = self._build_vocab(text)
        self.itos  = {i: ch for ch, i in self.vocab.items()}

        # encode corpus once (BOS … EOS)
        self.encoded = (
            [self.BOS] +
            [self.vocab.get(ch, self.UNK) for ch in text] +
            [self.EOS]
        )

    # ──────────────────────────────────────────────────────────────
    def _build_vocab(self, txt: str):
        chars = sorted(set(txt))
        offset = 4
        return {
            "<PAD>": self.PAD,
            "<UNK>": self.UNK,
            "<BOS>": self.BOS,
            "<EOS>": self.EOS,
            **{c: i + offset for i, c in enumerate(chars)}
        }

    # iterable → yields (context_tensor, target_tensor)
    def sliding_windows(self):
        seq = self.encoded
        contexts = [seq[i: i + self.k] for i in range(len(seq) - self.k)]
        targets = [seq[i + self.k] for i in range(len(seq) - self.k)]
        return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)