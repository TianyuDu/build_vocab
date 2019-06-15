import pandas as pd
import argparse
import os

import vocab


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "path", help="directory to detect", type=str)
#     parser.add_argument(
#         "")
#     args = parser.parse_args()
#     print(
#         f"Path: {parser.parse_args().path} exists: {os.path.exists(parser.parse_args().path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--initialize", type=bool, help="initialize dataset")
    parser.add_argument("-e", "--embedding", type=str, help="embedding file directory")
    parser.add_argument("--vocab", type=str, help="What vocab bank to build.")
    args = parser.parse_args()
    
    if args.initialize:
        assert args.embedding is not None, "Iniitalize is called, (--embedding) path is required."
        assert os.path.exists(args.embedding), f"Embedding path not found: {args.embedding}"
        print(
            f"Initialize with word2vec embedding: {args.embedding}")
        sim_map = vocab.build_similarity_map()
    if args.vocab.lower() == "gre":
        start = input("Choose an arbitrary vocab to start.")
        vocab = vocab.load_vocab("./database/gre3000.xlsx")
        emb_idx = vocab.load_word2vec(args.embedding)
        quiz(start, vocab, emb_idx)
    
