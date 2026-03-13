import numpy as np
from SkipGram_NegativeSampling import *

def grad_check(model, eps=1e-5, rtol=1e-5, vocab=20):
    K = 3
    B = 4

    center = np.random.randint(0, vocab, size=B)
    pos = np.random.randint(0, vocab, size=B)
    neg = np.random.randint(0, vocab, size=(B, K))

    print("Center id", center)
    print("Pos id", pos)
    print("Neg id", neg)

    model.zero_grad()
    model.forward(center, pos, neg)
    model.backward()

    #check V matrix
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

            if rel > rtol:
                print("NOT PASSED")
                return

    #check U
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

            if rel > rtol:
                print("NOT PASSED")
                return

    return "PASSED"


