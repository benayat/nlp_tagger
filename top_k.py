import numpy as np

with open("data/embeddings/vocab.txt", "r") as f:
    words = [line.strip() for line in f]
vecs = np.loadtxt("data/embeddings/wordVectors.txt")
word_to_vec = {word: vec for word, vec in zip(words, vecs)}

def most_similar(word, k):
    word_vec = word_to_vec[word]
    similarities = np.dot(vecs, word_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(word_vec))
    top_k_indices = np.argsort(similarities)[-k-1:-1][::-1]
    return [(words[i], similarities[i]) for i in top_k_indices]

# words = dog, england, john, explode, office.
words_list = ["dog", "england", "john", "explode", "office"]
k = 5
print(f"Most similar words to each word, in a word: distance format':")
for word in words_list:
    print(f"Current word is {word}")
    similar_words = most_similar(word, k)
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.4f}")
    print()
