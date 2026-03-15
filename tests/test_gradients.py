import numpy as np
from src.core.skip_gram_negative_sampling import Word2VecSGNS

def test_gradients():
    eps = 1e-5
    rtol = 1e-5

    vocab = 100
    dim = 5
    K = 3 #negative_samples
    B = 4 #batch_size

    model = Word2VecSGNS(vocab, dim, word2id={}, id2word={})

    center = np.random.randint(0, vocab, size=B)
    pos = np.random.randint(0, vocab, size=B)
    neg = np.random.randint(0, vocab, size=(B, K))

    model.zero_grad()
    model.forward(center, pos, neg)
    model.backward()

    # check V matrix
    for i in range(model.V.shape[0]):
        for j in range(model.V.shape[1]):
            old = model.V[i, j]

            model.V[i, j] = old + eps
            lp = model.forward(center, pos, neg)

            model.V[i, j] = old - eps
            lm = model.forward(center, pos, neg)

            model.V[i, j] = old

            gnum = (lp - lm) / (2 * eps)
            gan = model.dV[i, j]

            rel = abs(gnum - gan) / max(1e-8, abs(gnum), abs(gan))

            assert rel <= rtol

    # check U
    for i in range(model.U.shape[0]):
        for j in range(model.U.shape[1]):
            old = model.U[i, j]

            model.U[i, j] = old + eps
            lp = model.forward(center, pos, neg)

            model.U[i, j] = old - eps
            lm = model.forward(center, pos, neg)

            model.U[i, j] = old

            gnum = (lp - lm) / (2 * eps)
            gan = model.dU[i, j]

            rel = abs(gnum - gan) / max(1e-8, abs(gnum), abs(gan))

            assert rel <= rtol

