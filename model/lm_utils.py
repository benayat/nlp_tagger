import torch


def sample(model, vocab, k: int, n_chars: int, prefix: str = "",
           temperature: float = 1.0, device: str = "cuda"):
    """Return `prefix` + newly‑sampled characters."""
    string_to_index = vocab
    index_to_string = {i: char for char, i in vocab.items()}

    # seed context
    context = [string_to_index.get(char, string_to_index["<UNK>"]) for char in prefix][-k:]
    if len(context) < k:
        context = [string_to_index["<BOS>"]] * (k - len(context)) + context
    generated = list(prefix)

    model.eval()
    with torch.no_grad():
        for _ in range(n_chars):
            input = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input)[:, -len(vocab):] / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_i = torch.multinomial(probs, 1).item()
            if next_i == string_to_index["<EOS>"]:
                break
            next_char = index_to_string.get(next_i, "?")
            generated.append(next_char)
            context = context[1:] + [next_i]   # slide window
    model.train()
    return "".join(generated)