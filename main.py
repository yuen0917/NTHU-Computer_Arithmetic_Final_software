import numpy as np

# --------- 基本運算函式們 ---------


def conv2d(x, w, b, padding=1, stride=1):
    """
    x: (N, C_in, H, W)
    w: (C_out, C_in, KH, KW)
    b: (C_out,)
    回傳: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # zero padding
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for co in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    w_start = w_out * stride
                    patch = x_padded[
                        n, :, h_start : h_start + KH, w_start : w_start + KW
                    ]
                    out[n, co, h_out, w_out] = np.sum(patch * w[co]) + b[co]
    return out


def maxpool2x2(x):
    """
    x: (N, C, H, W)
    2x2, stride=2
    回傳: (N, C, H/2, W/2)
    """
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    H_out, W_out = H // 2, W // 2

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    patch = x[
                        n, c, 2 * h_out : 2 * h_out + 2, 2 * w_out : 2 * w_out + 2
                    ]
                    out[n, c, h_out, w_out] = np.max(patch)
    return out


def relu(x):
    return np.maximum(0.0, x)


def selu(x, lam=1.0507, alpha=1.6733):
    neg_mask = x <= 0
    out = np.empty_like(x, dtype=np.float32)
    out[~neg_mask] = lam * x[~neg_mask]
    out[neg_mask] = lam * alpha * np.expm1(x[neg_mask])
    return out


def gelu(x):
    # tanh 近似式
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


def global_avg_pool(x):
    """
    x: (N, C, H, W)
    回傳: (N, C, 1, 1)
    """
    return np.mean(x, axis=(2, 3), keepdims=True)


def flatten(x):
    """
    x: (N, C, 1, 1) -> (N, C)
    或一般 (N, C, H, W) 也可以攤平
    """
    return x.reshape(x.shape[0], -1)


def linear(x, W, b):
    """
    x: (N, in_features)
    W: (out_features, in_features)
    b: (out_features,)
    回傳: (N, out_features)
    """
    return x @ W.T + b


def softmax(x):
    """
    x: (N, num_classes)
    回傳: (N, num_classes)
    """
    x_shift = x - np.max(x, axis=1, keepdims=True)  # 避免 overflow
    exp = np.exp(x_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)


# --------- 照作業架構的整體 CNN ---------


class GlobalAvgPoolCNN:
    def __init__(self, rng=None, weights_path=None):
        """
        Args:
            rng: Random number generator for random initialization
            weights_path: Path to .npz file containing trained weights
        """
        if weights_path is not None:
            # Load trained weights
            weights = np.load(weights_path)
            self.W1 = weights["W1"].astype(np.float32)
            self.b1 = weights["b1"].astype(np.float32)
            self.W2 = weights["W2"].astype(np.float32)
            self.b2 = weights["b2"].astype(np.float32)
            self.W3 = weights["W3"].astype(np.float32)
            self.b3 = weights["b3"].astype(np.float32)
            self.W4 = weights["W4"].astype(np.float32)
            self.b4 = weights["b4"].astype(np.float32)
        else:
            # Random initialization
            if rng is None:
                rng = np.random.default_rng(0)

            # 1. Conv2d(1→8, kernel=3, padding=1)
            self.W1 = rng.normal(0, 0.1, size=(8, 1, 3, 3)).astype(np.float32)
            self.b1 = np.zeros(8, dtype=np.float32)

            # 2. Conv2d(8→16, kernel=3, padding=1)
            self.W2 = rng.normal(0, 0.1, size=(16, 8, 3, 3)).astype(np.float32)
            self.b2 = np.zeros(16, dtype=np.float32)

            # 4. Conv2d(16→32, kernel=3, padding=1)
            self.W3 = rng.normal(0, 0.1, size=(32, 16, 3, 3)).astype(np.float32)
            self.b3 = np.zeros(32, dtype=np.float32)

            # 7. FC(32→10)
            self.W4 = rng.normal(0, 0.1, size=(10, 32)).astype(np.float32)
            self.b4 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        """
        x: (N, 1, 28, 28) 的 MNIST 灰階圖片
        回傳: (N, 10) 的 softmax 機率
        """

        # 1. Conv(1→8) + ReLU  -> (N, 8, 28, 28)
        x = conv2d(x, self.W1, self.b1, padding=1)
        x = relu(x)

        # 2. Conv(8→16) + SELU -> (N, 16, 28, 28)
        x = conv2d(x, self.W2, self.b2, padding=1)
        x = selu(x)

        # 3. MaxPool 2×2       -> (N, 16, 14, 14)
        x = maxpool2x2(x)

        # 4. Conv(16→32) + GELU -> (N, 32, 14, 14)
        x = conv2d(x, self.W3, self.b3, padding=1)
        x = gelu(x)

        # 5. Global Average Pooling -> (N, 32, 1, 1)
        x = global_avg_pool(x)

        # 6. Flatten -> (N, 32)
        x = flatten(x)

        # 7. FC(32→10) + Softmax -> (N, 10)
        logits = linear(x, self.W4, self.b4)
        probs = softmax(logits)
        return probs


# --------- 小測試：確認維度沒問題 ---------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    model = GlobalAvgPoolCNN(rng)

    # 假設一個 batch 裡有 4 張 MNIST 圖片
    x_dummy = rng.random((4, 1, 28, 28), dtype=np.float32)
    out = model.forward(x_dummy)

    print("input shape :", x_dummy.shape)  # (4, 1, 28, 28)
    print("output shape:", out.shape)  # (4, 10)
    print("每筆 softmax 和 =", out[0].sum())
