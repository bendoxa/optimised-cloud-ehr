"""NumPy forward pass of the trained CNN-BiLSTM-Attention model.

Inference only, so the prototype runs without TensorFlow (about 150 MB of
memory instead of about 700 MB) and fits free hosting tiers. The weights are
exported from the trained Keras model by code/export_prototype_model.py and
give the same probabilities as Keras to within floating-point rounding.
"""
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _conv_same(x, k, b):
    """Conv1D, padding='same', stride 1. x: (T, Fin); k: (K, Fin, Fout)."""
    K = k.shape[0]
    left = (K - 1) // 2
    xp = np.pad(x, ((left, K - 1 - left), (0, 0)))
    return sum(xp[i:i + len(x)] @ k[i] for i in range(K)) + b


def _bn(x, g, b, mean, var, eps=1e-3):
    return (x - mean) / np.sqrt(var + eps) * g + b


def _lstm(xs, W, U, b):
    """Keras LSTM (gate order i, f, c, o) over the given steps; returns one output per step."""
    u = U.shape[0]
    h, c = np.zeros(u), np.zeros(u)
    out = []
    for x in xs:
        z = x @ W + h @ U + b
        i, f, g, o = _sig(z[:u]), _sig(z[u:2 * u]), np.tanh(z[2 * u:3 * u]), _sig(z[3 * u:])
        c = f * c + i * g
        h = o * np.tanh(c)
        out.append(h)
    return out


class Model:
    def __init__(self, path):
        self.w = dict(np.load(path))

    def predict(self, X, M):
        """X: (1, T, F) encounters, M: (1, T) mask. Returns (probabilities (4,), attention (T,))."""
        w = self.w
        x, m = np.asarray(X, float)[0], np.asarray(M, float)[0]
        z = np.maximum(_bn(_conv_same(x, w["c1_k"], w["c1_b"]), w["bn1_g"], w["bn1_b"], w["bn1_m"], w["bn1_v"]), 0)
        z = np.maximum(_bn(_conv_same(z, w["c2_k"], w["c2_b"]), w["bn2_g"], w["bn2_b"], w["bn2_m"], w["bn2_v"]), 0)
        T, u = len(z), w["f_U"].shape[0]
        idx = [t for t in range(T) if m[t] > 0]  # masked steps are skipped, as Keras does
        h = np.zeros((T, 2 * u))
        for t, ht in zip(idx, _lstm(z[idx], w["f_W"], w["f_U"], w["f_b"])):
            h[t, :u] = ht
        for t, ht in zip(idx[::-1], _lstm(z[idx[::-1]], w["b_W"], w["b_U"], w["b_b"])):
            h[t, u:] = ht
        e = np.tanh(h @ w["a_W"] + w["a_b"]) @ w["a_v"]
        e = np.where(m > 0, e, -1e9)
        a = np.exp(e - e.max()); a = a / a.sum() * m
        a = a / (a.sum() + 1e-9)
        s = (h * a[:, None]).sum(0)
        d = np.maximum(s @ w["d1_k"] + w["d1_b"], 0)
        return _sig(d @ w["d2_k"] + w["d2_b"]), a
