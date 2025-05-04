import numpy as np
import torch


def load_dataset_from_reader(reader, source):
    features = []
    labels = []
    for window, tag_index in reader.generate_windows(source):
        features.append(torch.tensor(window, dtype=torch.long))
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

def create_word_to_vec_vocab_dict_from_embeddings_files(words_file_path, embeddings_file_path):
    with open(words_file_path, "r") as f:
        words = [line.strip() for line in f]
    vecs = np.loadtxt(embeddings_file_path)
    return {word: vec for word, vec in zip(words, vecs)}

def calculate_similarity(vecs, word_vec):
    return np.dot(vecs, word_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(word_vec))

def most_similar(word_to_vec, word, k):
    vecs = list(word_to_vec.values())
    words = list(word_to_vec.keys())
    word_vec = word_to_vec[word]
    similarities = np.dot(vecs, word_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(word_vec))
    top_k_indices = np.argsort(similarities)[-k-1:-1][::-1]
    return [(words[i], similarities[i]) for i in top_k_indices]
