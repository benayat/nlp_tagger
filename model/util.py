import torch.optim as optim
import torch
import torch.nn as nn

def create_optimizer_and_scheduler(model, initial_lr, weight_decay, scheduler_patience, scheduler_factor, min_lr):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience,
                                                           factor=scheduler_factor, min_lr=min_lr)
    return optimizer, scheduler


def batch_accuracy(predictions, gold, pad_idx, o_idx=None):
    mask = gold != pad_idx
    if o_idx is not None:  # NER: drop trivial O‑hits
        mask &= ~((gold == o_idx) & (predictions == o_idx))
    correct = ((predictions == gold) & mask).sum().item()
    return correct, mask.sum().item()


def train_one_epoch(model, loader, criterion, optimizer,
                    pad_idx, o_idx, device):
    model.train()
    total_loss = total_correct = total_tokens = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct, tokens = batch_accuracy(logits.argmax(1), y, pad_idx, o_idx)
        total_correct += correct
        total_tokens += tokens
    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx, o_idx, device):
    model.eval()
    total_loss = total_correct = total_tokens = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct, tokens = batch_accuracy(logits.argmax(1), y, pad_idx, o_idx)
        total_correct += correct
        total_tokens += tokens
    return total_loss / total_tokens, total_correct / total_tokens

@torch.no_grad()
def predict(model, X, device):
    model.eval()
    X = X.to(device)
    logits = model(X)
    predictions = logits.argmax(1)
    return predictions