import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model.kgram_cnn_lm import KGramCnnLm
from model.lm_utils import sample
from reader.lm_reader import CharLmReader

K = 5
BATCH_SIZE = 2048
EMBEDDINGS_DIM = 100
DEVICE = "cuda"
NUM_EPOCHS = 60
LR = 3e-1
WEIGHT_DECAY = 3e-3
SCHEDULER_DECAY_FACTOR = 0.1
SCHEDULER_PATIENCE = 1
reader = CharLmReader("data/lm-data/eng-data/input.txt", k=K)
contexts, targets = reader.sliding_windows()
train_contexts, val_contexts, train_targets, val_targets = train_test_split(
    contexts, targets, test_size=0.1, random_state=42
)
train_dataset = TensorDataset(train_contexts, train_targets)
val_dataset = TensorDataset(val_contexts, val_targets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = KGramCnnLm(len(reader.vocab), embeddings_dim=EMBEDDINGS_DIM, n_filters=256).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=SCHEDULER_PATIENCE,
    factor=SCHEDULER_DECAY_FACTOR,
    min_lr=1e-5
)
loss_fn = nn.CrossEntropyLoss()
previous_learning_rate_value = scheduler.get_last_lr()[0]
for epoch in range(NUM_EPOCHS):
    total_loss, n = 0.0, 0
    for context_train, target_train in train_loader:
        context_train, target_train = context_train.to(DEVICE),target_train.to(DEVICE)
        logits = model(context_train)
        loss = loss_fn(logits, target_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * context_train.size(0)
        n += context_train.size(0)
    train_loss = total_loss / n

    # Validation
    model.eval()
    validation_loss, validation_n = 0.0, 0
    with torch.no_grad():
        for context_val, target_val in val_loader:
            context_val, target_val = context_val.to(DEVICE), target_val.to(DEVICE)
            loss = loss_fn(model(context_val), target_val)
            validation_loss += loss.item() * context_val.size(0)
            validation_n += context_val.size(0)
    validation_loss /= validation_n
    perplexity = torch.exp(torch.tensor(validation_loss))
    if (epoch + 1) % 5 == 0:  # sample every 5 epochs
        print(sample(model, reader.vocab, k=K, n_chars=200, prefix="Best "))
    scheduler.step(validation_loss)
    current_learning_rate_value = scheduler.get_last_lr()[0]
    if current_learning_rate_value != previous_learning_rate_value:
        optimizer.param_groups[0]["weight_decay"] *= SCHEDULER_DECAY_FACTOR
        previous_learning_rate_value = current_learning_rate_value
        print(f"Learning-rate decayed → {current_learning_rate_value:.1e}; "
              f"weight-decay scaled × {SCHEDULER_DECAY_FACTOR:.3f}")
    print(f"epoch {epoch + 1}: train-loss {train_loss:.5f} | val loss {validation_loss:.4f} | perplexity {perplexity:.4f} | learning-rate {current_learning_rate_value:.1e}")

