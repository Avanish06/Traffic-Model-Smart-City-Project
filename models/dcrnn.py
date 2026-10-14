"""
DCRNN — Diffusion Convolutional Recurrent Neural Network
Li et al., ICLR 2018  (https://arxiv.org/abs/1707.01926)

Key ideas:
  - Diffusion convolution replaces standard graph convolution
    using forward AND backward random-walk matrices
  - Used inside a GRU encoder–decoder with scheduled sampling
  - Multi-step forecasting via seq2seq
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Convolution (one direction, one step)
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionConv(nn.Module):
    """
    Computes:  Z = sum_{k=0}^{K} P^k X W_k
    where P is the random-walk normalised adjacency (one direction).
    """

    def __init__(self, c_in: int, c_out: int, K: int):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(c_in * (K + 1), c_out)

    def forward(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, c_in)
        P: (N, N)
        """
        out = [x]
        xk  = x
        for _ in range(self.K):
            xk = torch.einsum("nm,bmc->bnc", P, xk)
            out.append(xk)
        h = torch.cat(out, dim=-1)      # (B, N, c_in*(K+1))
        return self.linear(h)           # (B, N, c_out)


# ─────────────────────────────────────────────────────────────────────────────
# DCGRUCell — GRU with diffusion convolution replacing matrix multiplies
# ─────────────────────────────────────────────────────────────────────────────
class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, K: int, num_supports: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Each gate: r, u, c — operates on (input, hidden) → (hidden)
        c_in = input_dim + hidden_dim

        # We build one DiffusionConv per support per gate
        # For simplicity combine both supports into one by concatenation
        self.dc_ru = nn.ModuleList(
            [DiffusionConv(c_in, hidden_dim, K) for _ in range(num_supports)]
        )
        self.dc_c  = nn.ModuleList(
            [DiffusionConv(c_in, hidden_dim, K) for _ in range(num_supports)]
        )
        self.fc_ru_r = nn.Linear(hidden_dim * num_supports, hidden_dim)
        self.fc_ru_u = nn.Linear(hidden_dim * num_supports, hidden_dim)
        self.fc_c    = nn.Linear(hidden_dim * num_supports, hidden_dim)

    def forward(
        self,
        x:        torch.Tensor,
        h:        torch.Tensor,
        supports: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        x:  (B, N, input_dim)
        h:  (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)  # (B, N, input+hidden)

        # Gates r, u
        ru_outs = [dc(xh, P) for dc, P in zip(self.dc_ru, supports)]
        ru_cat  = torch.cat(ru_outs, dim=-1)
        r = torch.sigmoid(self.fc_ru_r(ru_cat))
        u = torch.sigmoid(self.fc_ru_u(ru_cat))

        # Candidate
        xrh  = torch.cat([x, r * h], dim=-1)
        c_outs = [dc(xrh, P) for dc, P in zip(self.dc_c, supports)]
        c_cat  = torch.cat(c_outs, dim=-1)
        c      = torch.tanh(self.fc_c(c_cat))

        h_new = u * h + (1 - u) * c
        return h_new


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer DCGRU encoder / decoder
# ─────────────────────────────────────────────────────────────────────────────
class DCGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, K, num_supports=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            c_in = input_dim if i == 0 else hidden_dim
            self.layers.append(DCGRUCell(c_in, hidden_dim, K, num_supports))

    def forward(self, x_seq, supports):
        """
        x_seq:  (B, T, N, input_dim)
        returns: list of hidden states, one per layer, each (B, N, hidden_dim)
        """
        B, T, N, _ = x_seq.shape
        h_list = [torch.zeros(B, N, self.layers[0].hidden_dim, device=x_seq.device)
                  for _ in self.layers]

        for t in range(T):
            x_t = x_seq[:, t]   # (B, N, input_dim)
            for l, cell in enumerate(self.layers):
                h_list[l] = cell(x_t, h_list[l], supports)
                x_t = h_list[l]

        return h_list


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, K, num_supports=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            c_in = input_dim if i == 0 else hidden_dim
            self.layers.append(DCGRUCell(c_in, hidden_dim, K, num_supports))
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, go_symbol, h_list, supports, num_pred, teacher_forcing_x=None,
                teacher_forcing_ratio=0.0):
        """
        go_symbol: (B, N, input_dim) — usually zeros
        h_list:    encoder hidden states
        returns:   (B, num_pred, N, input_dim)
        """
        outputs = []
        x_t     = go_symbol

        for t in range(num_pred):
            for l, cell in enumerate(self.layers):
                h_list[l] = cell(x_t, h_list[l], supports)
                x_t = h_list[l]
            pred = self.out_proj(x_t)       # (B, N, input_dim)
            outputs.append(pred.unsqueeze(1))

            # Scheduled sampling
            if teacher_forcing_x is not None and torch.rand(1).item() < teacher_forcing_ratio:
                x_t = teacher_forcing_x[:, t]
            else:
                x_t = pred

        return torch.cat(outputs, dim=1)    # (B, num_pred, N, input_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Full DCRNN Model
# ─────────────────────────────────────────────────────────────────────────────
class DCRNN(nn.Module):
    """
    Diffusion Convolutional RNN (Li et al., ICLR 2018).

    Args:
        num_nodes:    N  (number of sensors)
        in_channels:  input feature dim
        hidden_dim:   hidden state size
        num_layers:   number of stacked DCGRU layers
        K:            diffusion steps
        num_pred:     prediction horizon steps
        adj_mx:       (N, N) adjacency matrix as numpy array
    """

    def __init__(
        self,
        num_nodes:   int,
        in_channels: int   = 1,
        hidden_dim:  int   = 64,
        num_layers:  int   = 2,
        K:           int   = 3,
        num_pred:    int   = 12,
        adj_mx:      np.ndarray | None = None,
    ):
        super().__init__()
        self.num_pred  = num_pred
        self.in_ch     = in_channels

        if adj_mx is None:
            adj_mx = np.eye(num_nodes, dtype=np.float32)

        self.register_buffer("fwd", self._rw(adj_mx))
        self.register_buffer("bwd", self._rw(adj_mx.T))

        self.encoder = DCGRUEncoder(in_channels, hidden_dim, num_layers, K)
        self.decoder = DCGRUDecoder(in_channels, hidden_dim, num_layers, K)

    @staticmethod
    def _rw(adj: np.ndarray) -> torch.Tensor:
        """Row-normalised (random-walk) adjacency."""
        d   = adj.sum(axis=1, keepdims=True).clip(min=1)
        rw  = (adj / d).astype(np.float32)
        return torch.tensor(rw)

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 0.0
                ) -> torch.Tensor:
        """
        x:      (B, T, N, C)  historical window
        returns (B, num_pred, N, C)
        """
        supports = [self.fwd, self.bwd]
        h_list   = self.encoder(x, supports)

        B, _, N, C = x.shape
        go = torch.zeros(B, N, C, device=x.device)

        out = self.decoder(
            go, h_list, supports, self.num_pred,
            teacher_forcing_x     = x,            # for scheduled sampling during train
            teacher_forcing_ratio = teacher_forcing_ratio,
        )
        return out   # (B, num_pred, N, C)


def build_dcrnn(num_nodes, adj_mx=None, num_pred=12, device="cpu") -> DCRNN:
    return DCRNN(
        num_nodes  = num_nodes,
        in_channels= 1,
        hidden_dim = 64,
        num_layers = 2,
        K          = 3,
        num_pred   = num_pred,
        adj_mx     = adj_mx,
    ).to(device)
