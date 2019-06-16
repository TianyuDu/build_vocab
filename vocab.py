import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from typing import Dict
import json
from tqdm import tqdm


def load_vocab(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="main", header=0)
    column_kept = ["Word", "UK Phonetics", "US Phonetics",
            "Paraphrase (w/ POS)", "Paraphrase (English)"]
    return df[column_kept]


def load_word2vec(embedding_path: str) -> dict:
    embeddings_index = {}
    print("Loading word2vec embedding file...")
    with open(embedding_path) as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float)
            embeddings_index[word] = coefs
    print("Embedding data loaded.")
    return embeddings_index

def similarity(word0: str, word1: str, embedding):
    def res(x): return embedding[x].reshape(1, len(embedding[x]))
    word0 = res(word0)
    word1 = res(word1)
    # return cos_distance(word0, word1)
    return cosine_similarity(word0, word1)
    # return - np.linalg.norm(embeddings_index[word0] - embeddings_index[word1])


def cos_distance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def find_match(word: str, embedding: dict, vocab_bank: pd.DataFrame):
    scores = [
        similarity(word, y, embedding) for y in vocab_bank["word"]
    ]
    argmax = np.argmax(scores)
    return vocab_bank["word"][argmax]

def build_best_match_map(
        vocab_path: str,
        embedding_path: str
        ) -> pd.DataFrame:
    print("Building similarity map...")
    embedding = load_word2vec(embedding_path)
    vocab_bank = load_vocab(vocab_path)
    
    total = len(vocab_bank)
    goodness = [w in emb_idx.keys() for w in vocab_bank["word"]]
    vocab_bank = vocab_bank[goodness]
    print(f"{len(vocab_bank)/total*100:0.2f}% of vocab bank are found in the embedding file.")

    match_map = {}
    for word in tqdm(vocab_bank["word"]):
        best_match = find_match(word, embedding, vocab_bank)
    return match_map


# c = build_similarity_map(vocab_path="./database/gre3000.xlsx",
#                          embedding_path="/Users/tianyudu/Downloads/gre/glove.6B.50d.txt")

# path = "./vocab.xlsx"
# vocab_bank_all = load_vocab(path)
# goodness = [w in embeddings_index.keys() for w in vocab_bank_all["word"]]
# vocab_bank = vocab_bank_all[goodness]

def quiz(start: str, vocab_bank: pd.DataFrame, embedding: dict):
    print("Action: [m/Enter]Meaning, [n]Next, [q]Quit")
    current = find_match(start)
    action = ""
    count = 0
    while True:
        print(f"\t\033[91m {current}\033[00m")
        action = input(">>> ")
        while action.lower() not in ["", "m", "n", "q"]:
            action = input(">>> ")
        if action in ["", "m"]:
            print(vocab_bank[vocab_bank["word"] == current]["meaning"].values[0])
            current = find_match(current, embedding)
        elif action == "n":
            current = find_match(current, embedding)
        elif action == "q":
            break
        count += 1
    print(f"Vocab built: {count}")
