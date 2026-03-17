from tqdm import tqdm
import numpy as np
import random
from src.core.model_exp import Word2VecSGNS
from typing import List, Tuple

def dataloader(data: List[Tuple[int, int]], batch_size:int, shuffle: bool=True):
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = [data[i] for i in batch_indices]

        central_ids = [x[0] for x in batch]
        pos_context_ids = [x[1] for x in batch]

        yield central_ids, pos_context_ids


def sample_negatives_for_batch(
    central_ids: np.ndarray,
    pos_ids: np.ndarray,
    k: int,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    B = len(central_ids)
    vocab_size = len(probs)

    neg_ids = np.zeros((B, k), dtype=np.int32)

    for i in range(B):
        forbidden = {int(central_ids[i]), int(pos_ids[i])}
        negs: list[int] = []

        while len(negs) < k:
            draw = rng.choice(
                vocab_size, size=(k - len(negs)) * 3, replace=True, p=probs
            )
            for w in draw:
                w = int(w)
                if w in forbidden:
                    continue
                negs.append(w)
                if len(negs) == k:
                    break

        neg_ids[i] = negs

    return neg_ids


def train(
    model: Word2VecSGNS,
    data: List[Tuple[int, int]],
    num_epochs: int,
    k: int = 5,
    batch_size: int = 32,
    probs=None,
    rng=None,
    print_every: int = 100
):

    words_processed = 0

    for epoch in tqdm(range(num_epochs)):
        print("Epoch:", epoch + 1)
        print()

        total_loss = 0
        batches = 0

        for central_ids, pos_ids in dataloader(data, batch_size=batch_size):
            neg_ids = sample_negatives_for_batch(
                central_ids, pos_ids, k=k, probs=probs, rng=rng
            )
            model.zero_grad()
            last_loss = model.forward(central_ids, pos_ids, neg_ids)
            model.backward()
            model.step()

            total_loss += last_loss
            batches += 1
            words_processed += batch_size

            if batches % print_every == 0:
                epoch_part = batches * batch_size / len(data)
                print(f"Epoch part {epoch_part:.2f}: loss={total_loss / batches:.4f}")

                filename = f"outputs/model/epoch_{epoch + 1}_part_{epoch_part:.2f}_loss_{total_loss / batches:.4f}.npz"
                model.save(filename)

        print(f"epoch {epoch + 1}: loss={total_loss / batches:.4f}")
        filename = f"outputs/model/epoch_{epoch + 1}.npz"
        model.save(filename)
