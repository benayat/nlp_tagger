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

def load_test_dataset_from_reader(reader, source):
    return torch.stack([torch.tensor(window) for window in reader.generate_windows(source, is_test=True)])

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip().split("\n\n")

def process_sentences(blocks, pad_token, task):
    sentences = []
    for block in blocks:
        tokens = []
        for line in block.splitlines():
            if task == 'ner' and (line.startswith("-DOCSTART-") or not line.strip()):
                continue
            word, tag = line.strip().split()
            tokens.append((word, tag))
        padded = [(pad_token, "PAD")] * 2 + tokens + [(pad_token, "PAD")] * 2
        sentences.append(padded)
    return sentences

def process_sentences_for_test(blocks, pad_token):
    sentences = []
    for block in blocks:
        tokens = []
        for line in block.splitlines():
            if line.startswith("-DOCSTART-") or not line.strip():
                continue
            word = line.strip()
            tokens.append(word)
        padded = [pad_token * 2] + tokens + [pad_token * 2]
        sentences.append(padded)
    return sentences