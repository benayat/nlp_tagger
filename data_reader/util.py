import torch


def load_dataset_from_reader(reader, source):
    features = []
    labels = []
    for window, tag_index in reader.generate_windows(source):
        features.append(torch.tensor(window))
        labels.append(tag_index)
    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y