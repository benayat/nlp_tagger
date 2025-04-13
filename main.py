# python
import torch
import torch.nn as nn
import torch.optim as optim

from data_reader.ner_reader import NERReader
from data_reader.pos_reader import POSReader
from torch.utils.data import DataLoader, TensorDataset
from data_reader.util import load_dataset_from_reader
from model.mlp_pos_tagger import MLPPosTagger

# Reproducibility
torch.manual_seed(42)

reader = NERReader("data/ner/train")

X_train, y_train = load_dataset_from_reader(reader, "train")
X_val, y_val = load_dataset_from_reader(reader, "data/ner/dev")

# Create data loaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Model setup
vocab_size = len(reader.vocab)
pad_idx = reader.word_to_index[reader.pad_token]
output_dim = len(reader.tag_vocab)

model = MLPPosTagger(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_dim=100,
    output_dim=output_dim,
    pad_idx=pad_idx,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-4)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += X_batch.size(0)
    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    model.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            val_samples += X_batch.size(0)
    val_loss /= val_samples
    val_acc = val_correct / val_samples

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

print("Training complete.")