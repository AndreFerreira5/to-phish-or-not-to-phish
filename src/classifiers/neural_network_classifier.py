import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence, Optional
import numpy as np

__all__ = ["NeuralNetworkClassifier"]

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class _MLP(nn.Module):
    """A tiny fully-connected net: (input) → [Linear+ReLU]* → 1-logit"""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(1)   # (B) instead of (B,1)


class NeuralNetworkClassifier:
    """Lightweight binary NN classifier (PyTorch) with sklearn-like API."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (64, 32),
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        patience: Optional[int] = 5,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self._net: Optional[_MLP] = None
        self._fitted = False

    # ------------------------------------------------------
    # Core API
    # ------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_t = _to_tensor(X, self.device)
        y_t = _to_tensor(y.reshape(-1, 1), self.device)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._net = _MLP(X.shape[1], self.hidden_layers).to(self.device)
        optimiser = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf"); no_improve = 0
        best_state = self._net.state_dict()

        for epoch in range(self.epochs):
            self._net.train(); running = 0.0
            for xb, yb in dl:
                optimiser.zero_grad()
                loss = criterion(self._net(xb), yb.squeeze())
                loss.backward(); optimiser.step()
                running += loss.item() * xb.size(0)
            epoch_loss = running / len(ds)
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs} – loss {epoch_loss:.4f}")
            if self.patience is not None:
                if epoch_loss + 1e-4 < best_loss:
                    best_loss = epoch_loss; no_improve = 0; best_state = self._net.state_dict()
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if self.verbose:
                            print("Early stopping at epoch", epoch+1)
                        break
        self._net.load_state_dict(best_state)
        self._net.eval()
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        with torch.no_grad():
            logits = self._net(_to_tensor(X, self.device))
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------
    # sklearn-style param helpers (so GridSearchCV works)
    # ------------------------------------------------------
    def get_params(self, deep: bool = True):
        return {
            "hidden_layers": self.hidden_layers,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "device": str(self.device),
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
