import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reader.base_reader import BaseReader
from reader.ner_reader import NerReader
import matplotlib.pyplot as plt


def calculate_masked_accuracy(predicted_indices: torch.Tensor,
                              gold_indices: torch.Tensor,
                              index_of_padding_tag: int,
                              index_of_outside_tag: int | None = None) -> tuple[int, int]:
    valid_mask = gold_indices != index_of_padding_tag
    if index_of_outside_tag is not None:
        valid_mask &= ~((gold_indices == index_of_outside_tag) &
                        (predicted_indices == index_of_outside_tag))
    correct_predictions = ((predicted_indices == gold_indices) & valid_mask).sum().item()
    total_considered = valid_mask.sum().item()
    return correct_predictions, total_considered


def execute_epoch(model: nn.Module,
                  data_loader: DataLoader,
                  loss_function: nn.CrossEntropyLoss,
                  optimiser: torch.optim.Optimizer | None,
                  reader: BaseReader,
                  is_training: bool,
                  device='cuda') -> tuple[float, float]:
    padding_tag_index = reader.tag_to_index[reader.padding_tag]
    outside_tag_index = reader.tag_to_index.get("O") if isinstance(reader, NerReader) else None
    model.train() if is_training else model.eval()
    cumulative_loss = correct_total = token_total = 0
    for batch_inputs, batch_targets in data_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        if is_training:
            optimiser.zero_grad()
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_targets)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        correct, considered = calculate_masked_accuracy(
            logits.argmax(1), batch_targets, padding_tag_index, outside_tag_index
        )
        cumulative_loss += loss.item() * batch_inputs.size(0)
        correct_total += correct
        token_total += considered
    average_loss = cumulative_loss / token_total
    accuracy = correct_total / token_total
    return average_loss, accuracy


def build_data_loader(data_reader: BaseReader, batch_size: int, split_label: str, shuffle_flag: bool) -> DataLoader:
    windows_list, tag_indices = zip(*(data_reader.sliding_windows(split_label)))
    return DataLoader(
        TensorDataset(torch.tensor(windows_list), torch.tensor(tag_indices)),
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=4,
    )


def visualize_model_results(num_epochs, dev_accuracies, dev_losses, task):
    epochs_range = range(1, num_epochs + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs_range, dev_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Dev Accuracy per Epoch - {task.upper()}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epochs_range, dev_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Dev Loss per Epoch - {task.upper()}")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_model(model: nn.Module, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
