from typing import List, Tuple
import numpy as np
from typing import Counter

def subsample(tokens: List[int], id_counts: Counter, total_tokens:int, t:int=1e-3):
    result = []
    for w in tokens:
        f = id_counts[w] / total_tokens
        p_keep = min(1.0, np.sqrt(t / f))
        if np.random.rand() < p_keep:
            result.append(w)
    return result


def build_skipgram_pairs(
    tokens: List[int], context_window: int, id_counts: Counter, total_counts:int
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    tokens = subsample(tokens, id_counts, total_counts)
    n = len(tokens)

    for i in range(n):
        center = tokens[i]

        start = max(0, i - context_window)
        end = min(n, i + context_window + 1)

        for j in range(start, end):
            if j == i:
                continue

            context = tokens[j]
            pairs.append((center, context))

    return pairs
