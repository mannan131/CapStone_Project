"""Neural network for churn: PyTorch MLP when available, sklearn MLP fallback otherwise."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


if _TORCH_OK:  # pragma: no cover - requires torch installed

    class ChurnMLP(nn.Module):  # type: ignore[no-redef]  # pragma: no cover
        def __init__(
            self,
            input_dim: int,
            hidden_layers: tuple[int, ...] = (64, 32, 16),
            dropout: float = 0.3,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden_layers:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

else:
    ChurnMLP = None  # type: ignore


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible NN. Uses PyTorch if installed, else sklearn MLPClassifier.

    NOTE: ClassifierMixin must come first so sklearn resolves it as a classifier
    (otherwise is_classifier() is False and roc_auc scoring breaks).
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] | list[int] = (64, 32, 16),
        hidden_dim: int = 64,  # legacy single-width alias; ignored if hidden_layers set
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_layers = tuple(hidden_layers) if hidden_layers else (hidden_dim,)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

    def _torch_fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - needs torch
        assert ChurnMLP is not None
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        yb = y.astype(np.float32).reshape(-1, 1)
        patience = int(self.early_stopping_patience or 0)
        if patience > 0 and len(X) >= 20:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, yb, test_size=0.15, random_state=self.random_state, stratify=y
            )
        else:
            X_tr, y_tr, X_va, y_va = X, yb, None, None
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.model_ = ChurnMLP(X.shape[1], self.hidden_layers, self.dropout)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        best_state, best_loss, bad = None, float("inf"), 0
        for _ in range(self.epochs):
            self.model_.train()
            for xb, ybb in train_loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), ybb)
                loss.backward()
                opt.step()
            if X_va is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = float(
                        loss_fn(self.model_(torch.from_numpy(X_va)), torch.from_numpy(y_va))
                    )
                if val_loss < best_loss - 1e-4:
                    best_loss, bad = val_loss, 0
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if _TORCH_OK:
            self._torch_fit(X, y)
        else:
            self._sklearn_fit(X, y)
        return self

    def _sklearn_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """sklearn fallback with manual patience-based early stopping.

        (sklearn's built-in early stopping stalls on noisy validation scores,
        so patience is implemented explicitly over warm-started chunks.)
        """
        from sklearn.metrics import log_loss
        from sklearn.neural_network import MLPClassifier

        patience = int(self.early_stopping_patience or 0)
        budget = max(200, self.epochs * 10)
        X_tr, y_tr, X_va, y_va = X, y, None, None
        if patience > 0 and len(X) >= 30 and len(np.unique(y)) == 2:
            try:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X, y, test_size=0.15, random_state=self.random_state, stratify=y
                )
            except ValueError:
                pass
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=10,
            warm_start=True,
            random_state=self.random_state,
        )
        best_loss, bad, best_state = float("inf"), 0, None
        n_chunks = max(1, budget // 10)
        for _ in range(n_chunks):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X_tr, y_tr)
            if X_va is None:
                continue
            val_loss = float(log_loss(y_va, mlp.predict_proba(X_va), labels=self.classes_))
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = ([c.copy() for c in mlp.coefs_], [b.copy() for b in mlp.intercepts_])
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            mlp.coefs_, mlp.intercepts_ = best_state
        self.model_ = mlp

    def predict_proba(self, X):
        assert getattr(self, "model_", None) is not None, "Not fitted"
        if (
            _TORCH_OK and ChurnMLP is not None and isinstance(self.model_, ChurnMLP)
        ):  # pragma: no cover
            self.model_.eval()  # pragma: no cover
            with torch.no_grad():  # pragma: no cover
                logits = (  # pragma: no cover
                    self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().ravel()
                )
            p1 = 1 / (1 + np.exp(-logits))  # pragma: no cover
            return np.vstack([1 - p1, p1]).T  # pragma: no cover
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        proba = self.predict_proba(X)
        # positive class is the second column (classes_[1])
        return (proba[:, 1] >= 0.5).astype(int)
