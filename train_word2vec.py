import argparse
from collections import Counter

import numpy as np

from src.core.preprocessing import encode_text
from src.core.dataset import build_skipgram_pairs
from src.core.skip_gram_negative_sampling import Word2VecSGNS
from src.core.training import train


def build_unigram_probs(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    probs = counts.astype(np.float64) ** power
    probs /= probs.sum()
    return probs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--subsample-t", type=float, default=1e-5)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)

    args = parser.parse_args()

    with open("data/text8", "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    word_counts = Counter(words)

    unique_words = [w for w, c in word_counts.items() if c > 5]
    word_counts = Counter({w: c for w, c in word_counts.items() if w in unique_words})

    word2id = {word: idx for idx, word in enumerate(unique_words)}
    id2word = {idx: word for word, idx in word2id.items()}

    id_counts = Counter({word2id[w]: c for w, c in word_counts.items()})
    total_tokens = sum(word_counts.values())

    tokens = encode_text(text=text, word2id=word2id)

    skipgram_pairs = build_skipgram_pairs(
        tokens,
        args.context_window,
        id_counts,
        total_tokens,
        tolerance=args.subsample_t,
    )

    word_counts_array = np.zeros(len(unique_words), dtype=np.float64)
    for i in range(len(word_counts_array)):
        current_word = id2word[i]
        word_counts_array[i] = word_counts[current_word]

    probs = build_unigram_probs(word_counts_array)
    rng = np.random.default_rng(42)

    model = Word2VecSGNS(
        vocab_size=len(unique_words),
        dim=args.embedding_dim,
        word2id=word2id,
        id2word=id2word,
        optimizer=args.optimizer,
        lr=args.learning_rate,
    )

    train(
        model=model,
        data=skipgram_pairs,
        num_epochs=args.epochs,
        k=args.negative_samples,
        batch_size=512,
        probs=probs,
        rng=rng,
        print_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()