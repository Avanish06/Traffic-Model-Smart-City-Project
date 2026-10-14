"""
Baseline models: LSTM and ARIMA for traffic speed forecasting.

LSTM: multi-layer LSTM treating each sensor independently or jointly.
ARIMA: per-node fitting (parallelised via joblib) for short horizons.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    """
    Simple multi-layer LSTM that treats each sensor independently.
    Input is flattened across nodes: (B, T, N*C) → encode → decode to (B, num_pred, N, C).

    For large N (207 or 325 nodes) we reduce dim with a linear projection first.
    """

    def __init__(
        self,
        num_nodes:   int,
        in_channels: int = 1,
        hidden_dim:  int = 256,
        num_layers:  int = 2,
        num_pred:    int = 12,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.num_nodes  = num_nodes
        self.in_ch      = in_channels
        self.num_pred   = num_pred
        input_size      = num_nodes * in_channels

        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden_dim, num_nodes * in_channels * num_pred)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) → (B, num_pred, N, C)
        """
        B, T, N, C = x.shape
        x_flat = x.reshape(B, T, N * C)          # (B, T, N*C)

        _, (h, _) = self.encoder(x_flat)         # h: (num_layers, B, hidden)
        last_h    = h[-1]                        # (B, hidden)

        out = self.decoder(last_h)               # (B, num_pred * N * C)
        out = out.reshape(B, self.num_pred, N, C)
        return out


def build_lstm(num_nodes, num_pred=12, device="cpu") -> LSTMBaseline:
    return LSTMBaseline(
        num_nodes  = num_nodes,
        in_channels= 1,
        hidden_dim = 256,
        num_layers = 2,
        num_pred   = num_pred,
        dropout    = 0.2,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA Baseline
# ─────────────────────────────────────────────────────────────────────────────
class ARIMABaseline:
    """
    ARIMA(p, d, q) fitted independently per sensor node.
    Uses statsmodels.

    This is intentionally non-differentiable; treated as a one-time fit
    on the training tail and evaluated on the test windows.
    """

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order  = order
        self.models = {}   # node_idx → fitted model

    def fit(self, train_series: np.ndarray) -> None:
        """
        train_series: (T, N) raw speed (pre-split, not normalised).
        Fits one ARIMA per node. Progress printed every 10 nodes.
        """
        from statsmodels.tsa.arima.model import ARIMA
        N = train_series.shape[1]
        print(f"  Fitting ARIMA{self.order} on {N} nodes …")
        for n in range(N):
            if n % 50 == 0:
                print(f"    node {n}/{N}")
            series = train_series[:, n]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    mdl = ARIMA(series, order=self.order).fit()
                    self.models[n] = mdl
                except Exception:
                    self.models[n] = None

    def predict(self, history: np.ndarray, num_pred: int) -> np.ndarray:
        """
        history:  (T, N) most recent window of raw speed.
        returns:  (num_pred, N) forecast.
        """
        from statsmodels.tsa.arima.model import ARIMA
        N   = history.shape[1]
        out = np.zeros((num_pred, N), dtype=np.float32)
        for n in range(N):
            mdl = self.models.get(n)
            if mdl is None:
                out[:, n] = history[-1, n]
                continue
            try:
                fc = mdl.forecast(steps=num_pred)
                out[:, n] = fc
            except Exception:
                out[:, n] = history[-1, n]
        return out

    def batch_predict(
        self,
        x_windows: np.ndarray,
        num_pred: int,
        scaler: tuple[float, float],
    ) -> np.ndarray:
        """
        x_windows: (B, T, N, 1) normalised test windows.
        scaler:    (mean, std) used during preprocessing.
        Returns:   (B, num_pred, N, 1) predictions in normalised space.
        """
        mean, std = scaler
        # Denormalise
        x_raw = x_windows[..., 0] * std + mean     # (B, T, N)
        preds  = []
        for i in range(x_raw.shape[0]):
            p = self.predict(x_raw[i], num_pred)    # (num_pred, N)
            preds.append(p)
        preds = np.stack(preds, axis=0)             # (B, num_pred, N)
        # Renormalise
        preds_norm = (preds - mean) / (std + 1e-8)
        return preds_norm[..., np.newaxis]          # (B, num_pred, N, 1)
