"""
Toy non-LLM example trainer using only NumPy.

This keeps the example lightweight in environments where torch is not yet
installed. It prints `val_mse` so research_loop.py can score the run.
"""

import numpy as np


def make_dataset(n_samples: int, n_features: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    weights = np.linspace(0.2, 1.4, n_features, dtype=np.float32)
    y = (
        0.7 * np.sin(x[:, 0])
        + 0.5 * x[:, 1] * x[:, 2]
        + (x * weights).sum(axis=1) * 0.05
        + rng.standard_normal(n_samples, dtype=np.float32) * 0.05
    )
    return x, y.reshape(-1, 1)


def build_features(x: np.ndarray) -> np.ndarray:
    """Use polynomial and interaction features for a non-linear baseline."""
    feats = [x, x**2, (x[:, :1] * x[:, 1:2]), np.sin(x[:, :1])]
    return np.concatenate(feats, axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=x_aug.dtype)
    eye[-1, -1] = 0.0
    w = np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y)
    return w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def main() -> None:
    x_train_raw, y_train = make_dataset(20_000, seed=7)
    x_val_raw, y_val = make_dataset(5_000, seed=11)
    x_train = build_features(x_train_raw)
    x_val = build_features(x_val_raw)

    w = fit_ridge(x_train, y_train)
    train_mse = float(((predict(x_train, w) - y_train) ** 2).mean())
    val_mse = float(((predict(x_val, w) - y_val) ** 2).mean())

    print(f"train_mse:        {train_mse:.6f}")
    print("---")
    print(f"val_mse:          {val_mse:.6f}")


if __name__ == "__main__":
    main()
