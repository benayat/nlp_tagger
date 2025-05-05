import torch
from torch import nn

class MlpTaggerCharCnn(nn.Module):
    """
    input shape = (batch, components, 5)

   • components = 1 + char_max_len
      row‑0  : word indices
      rows≥1 : character indices of every position in the word, left‑padded with 0 (PAD_CHAR)

    (If you later add prefix/suffix rows, put them **before** the char rows
     and the model will still work – it looks only at row‑0 and the *last* rows.)
    """

    def __init__(self,
                 reader,                       # to fetch sizes straight from the reader
                 hidden_units: int,
                 char_embedding_dim: int = 50,
                 dropout_probability: float=0.5,
                 char_cnn_filters: int=30,
                 char_cnn_window_size: int=3
                 ):
        super().__init__()
        word_embeddings_dim   = reader.word_embedding_matrix.shape[1]
        num_tags       = len(reader.tag_to_index)
        pad_word_idx   = reader.word_to_index[reader.padding_token]

        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(reader.word_embedding_matrix),
            freeze=False,
            padding_idx=pad_word_idx
        )

        self.char_embeddings = nn.Embedding(
            len(reader.char_to_idx)+1, char_embedding_dim, padding_idx=0
        )
        self.char_dropout = nn.Dropout(dropout_probability)

        # padding=1 keeps output‑length == input‑length when kernel_size=3
        self.char_cnn = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_cnn_filters,
            kernel_size=char_cnn_window_size,
            padding=1               # SAME‑padding ⇒ len_out == len_in
        )

        combined_dim = word_embeddings_dim + char_cnn_filters
        self.classifier_pipeline = nn.Sequential(
            nn.Linear(combined_dim * 5, hidden_units),
            nn.Tanh(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_units, num_tags),
        )

    # ─────────────────────────── forward ────────────────────────────────
    def forward(self, packed_ids: torch.Tensor) -> torch.Tensor:
        """
        packed_ids shape: (batch, components, 5)
        row‑0 ........................ word indices
        rows 1…N ..................... char indices (N = char_max_len)
        """
        words_ids = packed_ids[:, 0, :]                    # (B,5)

        # char_rows shape: (B, char_len, 5) → transpose to (B,5,char_len)
        char_rows = packed_ids[:, 1:, :]                   # exclude the word row
        char_inputs = char_rows.transpose(1, 2)            # (B,5,char_len)
        # -------- embeddings & CNN -------------------------------------
        word_embeds = self.word_embeddings(words_ids)      # (B,5,D_w)

        B, T, L = char_inputs.shape                        # B=batch, T=5, L=char_len
        chars_flat = char_inputs.reshape(B * T, L)         # (B·5, L)
        # print("word max:", words_ids.max().item(), "vs", self.word_embeddings.num_embeddings - 1)
        # print("char max:", chars_flat.max().item(), "vs", self.char_embeddings.num_embeddings - 1)
        char_embeds = self.char_embeddings(chars_flat)     # (B·5, L, D_c)
        char_embeds = self.char_dropout(char_embeds).permute(0, 2, 1)  # → (B·5, D_c, L)

        cnn_out = self.char_cnn(char_embeds)               # (B·5, F, L)
        pooled  = torch.nn.functional.max_pool1d(cnn_out, kernel_size=L).squeeze(2)  # (B·5, F)
        char_features = pooled.view(B, T, -1)                 # (B,5,F)

        # -------- concatenate and classify -----------------------------
        combined = torch.cat([word_embeds, char_features], dim=2)  # (B,5,D_w+F)
        return self.classifier_pipeline(combined.flatten(1))
