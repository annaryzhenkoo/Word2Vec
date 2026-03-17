import numpy as np
from typing import Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class Word2VecSGNS:
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        word2id: Dict[str, int],
        id2word: Dict[int, str],
        lr: float = 0.003,
        seed: int = 42,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, size=(vocab_size, dim)).astype(
            np.float64
        )  # central embeddings
        self.U = rng.normal(0.0, 0.01, size=(vocab_size, dim)).astype(
            np.float64
        )  # context embeddings

        self.dV = np.zeros_like(self.V)
        self.dU = np.zeros_like(self.U)

        self.lr = lr
        self.optimizer = optimizer.lower()

        self.word2id = word2id
        self.id2word = id2word

        self.cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None

        # Adam state
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mU = np.zeros_like(self.U)
        self.vU = np.zeros_like(self.U)

    def zero_grad(self):
        self.dV.fill(0.0)
        self.dU.fill(0.0)

    def forward(
        self, center_ids: np.ndarray, pos_ids: np.ndarray, neg_ids: np.ndarray
    ) -> float:
        center_ids = np.asarray(center_ids, dtype=np.int32)
        pos_ids = np.asarray(pos_ids, dtype=np.int32)
        neg_ids = np.asarray(neg_ids, dtype=np.int32)

        v = self.V[center_ids]      # (B, D)
        u_pos = self.U[pos_ids]     # (B, D)
        u_neg = self.U[neg_ids]     # (B, K, D)

        pos_scores = np.sum(v * u_pos, axis=1)                 # (B,)
        neg_scores = np.sum(u_neg * v[:, None, :], axis=2)     # (B, K)

        # SGNS loss: -log σ(pos) - sum log σ(-neg)
        loss_pos = -np.log(sigmoid(pos_scores) + 1e-12)              # (B,)
        loss_neg = -np.log(sigmoid(-neg_scores) + 1e-12).sum(axis=1) # (B,)
        loss = (loss_pos + loss_neg).mean()

        self.cache = (
            center_ids,
            pos_ids,
            neg_ids,
            v,
            u_pos,
            u_neg,
            pos_scores,
            neg_scores,
        )
        return float(loss)

    def backward(self):
        if self.cache is None:
            raise RuntimeError("Call forward() before backward().")

        center_ids, pos_ids, neg_ids, v, u_pos, u_neg, pos_scores, neg_scores = (
            self.cache
        )
        B = center_ids.shape[0]

        # dL/dpos_scores = (σ(pos) - 1) / B
        g_pos = (sigmoid(pos_scores) - 1.0) / B  # (B,)

        # dL/dneg_scores = σ(neg) / B
        g_neg = sigmoid(neg_scores) / B  # (B, K)

        # dv = g_pos * u_pos + sum_k g_neg_k * u_neg_k
        dv = g_pos[:, None] * u_pos + np.sum(
            g_neg[:, :, None] * u_neg, axis=1
        )  # (B, D)

        # du_pos = g_pos * v
        du_pos = g_pos[:, None] * v  # (B, D)

        # du_neg = g_neg * v
        du_neg = g_neg[:, :, None] * v[:, None, :]  # (B, K, D)

        # scatter-add into dV and dU
        np.add.at(self.dV, center_ids, dv)
        np.add.at(self.dU, pos_ids, du_pos)

        neg_flat = neg_ids.reshape(-1)                    # (B*K,)
        du_neg_flat = du_neg.reshape(-1, du_neg.shape[-1])  # (B*K, D)
        np.add.at(self.dU, neg_flat, du_neg_flat)

    def step(self):
        if self.optimizer == "sgd":
            self.V -= self.lr * self.dV
            self.U -= self.lr * self.dU

        elif self.optimizer == "adam":
            self.t += 1

            self.mV = self.beta1 * self.mV + (1.0 - self.beta1) * self.dV
            self.vV = self.beta2 * self.vV + (1.0 - self.beta2) * (self.dV ** 2)

            mV_hat = self.mV / (1.0 - self.beta1 ** self.t)
            vV_hat = self.vV / (1.0 - self.beta2 ** self.t)

            self.V -= self.lr * mV_hat / (np.sqrt(vV_hat) + self.eps)

            self.mU = self.beta1 * self.mU + (1.0 - self.beta1) * self.dU
            self.vU = self.beta2 * self.vU + (1.0 - self.beta2) * (self.dU ** 2)

            mU_hat = self.mU / (1.0 - self.beta1 ** self.t)
            vU_hat = self.vU / (1.0 - self.beta2 ** self.t)

            self.U -= self.lr * mU_hat / (np.sqrt(vU_hat) + self.eps)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def save(self, path: str):
        np.savez(
            path,
            V=self.V,
            U=self.U,
            dV=self.dV,
            dU=self.dU,
            lr=self.lr,
            optimizer=self.optimizer,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            t=self.t,
            mV=self.mV,
            vV=self.vV,
            mU=self.mU,
            vU=self.vU,
            word2id=self.word2id,
            id2word=self.id2word,
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)

        self.V = data["V"]
        self.U = data["U"]
        self.dV = np.zeros_like(self.V)
        self.dU = np.zeros_like(self.U)

        self.lr = float(data["lr"])
        self.optimizer = str(data["optimizer"])
        self.beta1 = float(data["beta1"])
        self.beta2 = float(data["beta2"])
        self.eps = float(data["eps"])
        self.t = int(data["t"])

        self.mV = data["mV"]
        self.vV = data["vV"]
        self.mU = data["mU"]
        self.vU = data["vU"]

        self.word2id = data["word2id"].item()
        self.id2word = data["id2word"].item()

        self.cache = None