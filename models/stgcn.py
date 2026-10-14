"""
ST-GCN — Spatio-Temporal Graph Convolutional Network
Yu et al., IJCAI 2018  (https://arxiv.org/abs/1709.04875)

Key ideas:
  - Chebyshev spectral graph convolution (ChebConv) on the sensor graph
  - Gated temporal convolution (GLU) applied after each graph conv
  - Multiple ST blocks stacked in a sandwich: TCN → GCN → TCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev Spectral Graph Convolution
# ─────────────────────────────────────────────────────────────────────────────
class ChebConv(nn.Module):
    """
    Chebyshev polynomial graph convolution up to order K.
    Z = sum_{k=0}^{K-1} T_k(L~) X W_k
    where L~ = 2L/lambda_max - I (rescaled Laplacian).
    """

    def __init__(self, c_in: int, c_out: int, K: int):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, c_in, c_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, N, c_in)
        L_tilde: (N, N) rescaled Laplacian
        returns: (B, N, c_out)
        """
        Tx_prev  = x                                # T_0 = I
        Tx_cur   = torch.einsum("nm,bmc->bnc", L_tilde, x)  # T_1 = L~
        out = torch.einsum("bnc,cq->bnq", Tx_prev, self.weight[0])
        if self.K > 1:
            out += torch.einsum("bnc,cq->bnq", Tx_cur, self.weight[1])

        for k in range(2, self.K):
            Tx_next = 2 * torch.einsum("nm,bmc->bnc", L_tilde, Tx_cur) - Tx_prev
            out    += torch.einsum("bnc,cq->bnq", Tx_next, self.weight[k])
            Tx_prev, Tx_cur = Tx_cur, Tx_next

        return out


import math


# ─────────────────────────────────────────────────────────────────────────────
# Gated Temporal Convolution (GLU)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalConv(nn.Module):
    """
    Gated 1-D causal convolution:
      output = tanh(W_f * x) o sigmoid(W_g * x)
    Causal padding is applied so output T == input T.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size - 1   # causal left-pad to preserve length
        self.conv_f = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), padding=(0, 0))
        self.conv_g = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N, T)  -> (B, c_out, N, T)  [length preserved via causal pad]"""
        # Left-pad the time axis for causal convolution
        x = F.pad(x, (self.pad, 0))
        return torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))


# ─────────────────────────────────────────────────────────────────────────────
# ST Block: TCN → GCN → TCN with residual
# ─────────────────────────────────────────────────────────────────────────────
class STBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, K: int, kernel_size: int = 3):
        super().__init__()
        self.tcn1  = TemporalConv(c_in,  c_out, kernel_size)
        self.gcn   = ChebConv(c_out, c_out, K)
        self.tcn2  = TemporalConv(c_out, c_out, kernel_size)
        self.res   = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()
        self.bn    = nn.LayerNorm([c_out])

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, c_in,  N, T)
        L_tilde: (N, N)
        returns: (B, c_out, N, T)  [T preserved]
        """
        residual = x

        # First temporal conv (T preserved via causal pad)
        h = self.tcn1(x)              # (B, c_out, N, T)

        # Graph conv — must operate node-by-node across batch+time
        B, C, N, T2 = h.shape
        h_t = h.permute(0, 3, 2, 1).reshape(B * T2, N, C)  # (B*T, N, C)
        h_t = self.gcn(h_t, L_tilde)                        # (B*T, N, C)
        h   = h_t.reshape(B, T2, N, C).permute(0, 3, 2, 1) # (B, C, N, T)
        h   = torch.relu(h)

        # Second temporal conv (T preserved)
        h = self.tcn2(h)              # (B, c_out, N, T)

        # Residual (no cropping needed — T is constant)
        h = h + self.res(residual)

        # LayerNorm on feature dim
        h = self.bn(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Output layer
# ─────────────────────────────────────────────────────────────────────────────
class OutputLayer(nn.Module):
    def __init__(self, c_in: int, num_pred: int, kernel_size: int = 1):
        super().__init__()
        self.tcn = TemporalConv(c_in, c_in, kernel_size)
        self.fc  = nn.Conv2d(c_in, num_pred, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N, T_remaining) → (B, num_pred, N)"""
        h   = self.tcn(x)             # shrink T further
        h   = self.fc(h)              # (B, num_pred, N, T_small)
        return h.mean(dim=-1)         # global average pool over remaining T


# ─────────────────────────────────────────────────────────────────────────────
# Full ST-GCN
# ─────────────────────────────────────────────────────────────────────────────
class STGCN(nn.Module):
    """
    Spatio-Temporal GCN (Yu et al., 2018).

    Architecture:
      Input → [ST-Block] × n_blocks → OutputLayer → (B, num_pred, N)
    """

    def __init__(
        self,
        num_nodes:   int,
        in_channels: int   = 1,
        num_pred:    int   = 12,
        num_his:     int   = 12,
        adj_mx:      np.ndarray | None = None,
        channels:    list  = None,    # channel progression per ST block
        K:           int   = 3,       # Chebyshev order
        kernel_size: int   = 3,       # temporal kernel
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64]

        self.num_pred = num_pred

        # Compute rescaled graph Laplacian L~ = 2L/λ_max - I
        L_tilde = self._compute_laplacian(adj_mx, num_nodes)
        self.register_buffer("L_tilde", L_tilde)

        # Build ST blocks
        block_ch = [in_channels] + channels
        self.st_blocks = nn.ModuleList()
        for i in range(len(channels)):
            self.st_blocks.append(
                STBlock(block_ch[i], block_ch[i + 1], K, kernel_size)
            )

        self.out = OutputLayer(channels[-1], num_pred, kernel_size=1)

    @staticmethod
    def _compute_laplacian(adj_mx: np.ndarray | None, num_nodes: int) -> torch.Tensor:
        if adj_mx is None:
            adj_mx = np.eye(num_nodes, dtype=np.float32)
        # Symmetric normalised Laplacian L = I - D^{-1/2} A D^{-1/2}
        A   = adj_mx.copy()
        np.fill_diagonal(A, 0)
        d   = A.sum(axis=1)
        d   = np.where(d == 0, 1, d)
        D_inv_sqrt = np.diag(d ** -0.5)
        L   = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

        # Rescale: L~ = 2L / lambda_max - I
        eigvals = np.linalg.eigvalsh(L)
        lam_max = eigvals.max() if eigvals.max() > 0 else 2.0
        L_tilde = (2 / lam_max) * L - np.eye(num_nodes)
        return torch.tensor(L_tilde, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, T, N, C)
        returns (B, num_pred, N, 1)
        """
        # (B, T, N, C) → (B, C, N, T)
        x = x.permute(0, 3, 2, 1)

        for block in self.st_blocks:
            x = block(x, self.L_tilde)

        out = self.out(x)           # (B, num_pred, N)
        return out.unsqueeze(-1)    # (B, num_pred, N, 1)


def build_stgcn(num_nodes, adj_mx=None, num_pred=12, num_his=12,
                device="cpu") -> STGCN:
    return STGCN(
        num_nodes   = num_nodes,
        in_channels = 1,
        num_pred    = num_pred,
        num_his     = num_his,
        adj_mx      = adj_mx,
        channels    = [64, 64, 64],
        K           = 3,
        kernel_size = 3,
    ).to(device)
