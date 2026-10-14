"""
Graph WaveNet for Deep Spatial-Temporal Graph Modeling
Wu et al., IJCAI 2019  (https://arxiv.org/abs/1906.00121)

Key ideas:
  - Self-adaptive adjacency matrix learned from data
  - Gated dilated causal convolutions (WaveNet-style TCN)
  - Graph convolution stacked with TCN inside residual blocks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Graph Convolution with support for multiple adjacency matrices
# ─────────────────────────────────────────────────────────────────────────────
class GraphConv(nn.Module):
    """
    Chebyshev-free graph convolution:
      Z = sum_k (A_k X) W_k
    where each A_k is a (possibly adaptive) adjacency matrix.
    """

    def __init__(self, c_in: int, c_out: int, num_supports: int, order: int = 2):
        super().__init__()
        # +1 because we prepend identity support
        self.c_in   = c_in
        self.c_out  = c_out
        self.order  = order
        c_in_total  = c_in * (order * num_supports + 1)
        self.linear = nn.Linear(c_in_total, c_out)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """
        x:        (B, c_in, N, T)
        supports: list of (N, N) tensors
        returns:  (B, c_out, N, T)
        """
        # Reshape to (B*T, N, c_in) for sparse matmul convenience
        B, C, N, T = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, N, C)   # (B*T, N, C)

        out = [x_flat]   # include x itself (identity diffusion)
        for sup in supports:
            x1 = torch.einsum("nm,bmc->bnc", sup, x_flat)
            out.append(x1)
            for _ in range(self.order - 1):
                x1 = torch.einsum("nm,bmc->bnc", sup, x1)
                out.append(x1)

        h = torch.cat(out, dim=-1)                              # (B*T, N, features)
        h = self.linear(h)                                      # (B*T, N, c_out)
        h = h.reshape(B, T, N, self.c_out).permute(0, 3, 2, 1) # (B, c_out, N, T)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Single WaveNet block
# ─────────────────────────────────────────────────────────────────────────────
class WaveNetBlock(nn.Module):
    def __init__(
        self,
        residual_ch: int,
        dilation_ch: int,
        skip_ch: int,
        kernel_size: int,
        dilation: int,
        num_supports: int,
        gcn_order: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = dropout

        # Dilated causal conv — gated activation (tanh ⊙ sigmoid)
        self.filter_conv = nn.Conv2d(residual_ch, dilation_ch,
                                     kernel_size=(1, kernel_size),
                                     dilation=(1, dilation))
        self.gate_conv   = nn.Conv2d(residual_ch, dilation_ch,
                                     kernel_size=(1, kernel_size),
                                     dilation=(1, dilation))

        self.gcn = GraphConv(dilation_ch, dilation_ch, num_supports, gcn_order)

        self.residual_conv = nn.Conv2d(dilation_ch, residual_ch, kernel_size=(1, 1))
        self.skip_conv     = nn.Conv2d(dilation_ch, skip_ch,     kernel_size=(1, 1))
        self.bn            = nn.BatchNorm2d(residual_ch)

    def forward(
        self, x: torch.Tensor, supports: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        f = self.filter_conv(x)
        g = self.gate_conv(x)
        x = torch.tanh(f) * torch.sigmoid(g)   # gated activation

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph convolution
        x = self.gcn(x, supports)

        skip = self.skip_conv(x)
        x    = self.residual_conv(x)

        # Crop residual to match temporal length after causal conv
        x = x + residual[..., -x.size(-1):]
        x = self.bn(x)
        return x, skip


# ─────────────────────────────────────────────────────────────────────────────
# Graph WaveNet
# ─────────────────────────────────────────────────────────────────────────────
class GraphWaveNet(nn.Module):
    """
    Graph WaveNet (Wu et al., 2019).

    Args:
        num_nodes:       number of sensors / graph nodes
        in_channels:     input feature dim  (default 1 for speed)
        out_channels:    output feature dim (default 1)
        num_pred:        number of future steps to predict
        num_his:         number of historical steps used as input
        adj_mx:          pre-computed adjacency matrix  (N, N) numpy array
        residual_ch:     channels inside residual stream
        dilation_ch:     channels inside dilation stream
        skip_ch:         channels in skip connections
        num_layers:      number of WaveNet blocks
        kernel_size:     temporal kernel size
        dropout:         dropout probability
        adaptive:        whether to learn an adaptive adjacency
        gcn_order:       diffusion steps for graph conv
    """

    def __init__(
        self,
        num_nodes:    int,
        in_channels:  int = 1,
        out_channels: int = 1,
        num_pred:     int = 12,
        num_his:      int = 12,
        adj_mx:       np.ndarray | None = None,
        residual_ch:  int = 32,
        dilation_ch:  int = 32,
        skip_ch:      int = 256,
        end_ch:       int = 512,
        num_layers:   int = 8,
        kernel_size:  int = 2,
        dropout:      float = 0.3,
        adaptive:     bool = True,
        gcn_order:    int = 2,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.num_pred    = num_pred
        self.adaptive    = adaptive

        # ── Adjacency supports ──────────────────────────────────────────────
        self.register_buffer(
            "adj_fwd",
            self._normalise(adj_mx) if adj_mx is not None else torch.eye(num_nodes),
        )
        self.register_buffer(
            "adj_bwd",
            self._normalise(adj_mx.T) if adj_mx is not None else torch.eye(num_nodes),
        )
        num_supports = 2  # forward + backward diffusion

        if adaptive:
            self.node_emb1 = nn.Embedding(num_nodes, 10)
            self.node_emb2 = nn.Embedding(num_nodes, 10)
            num_supports += 1

        # ── Input projection ────────────────────────────────────────────────
        self.start_conv = nn.Conv2d(in_channels, residual_ch, kernel_size=(1, 1))

        # ── Dilation schedule: [1,2,4,8,1,2,4,8,…] ─────────────────────────
        dilations = [2 ** (i % 8) for i in range(num_layers)]

        self.blocks = nn.ModuleList()
        receptive_field = 1
        for dilation in dilations:
            self.blocks.append(
                WaveNetBlock(
                    residual_ch   = residual_ch,
                    dilation_ch   = dilation_ch,
                    skip_ch       = skip_ch,
                    kernel_size   = kernel_size,
                    dilation      = dilation,
                    num_supports  = num_supports,
                    gcn_order     = gcn_order,
                    dropout       = dropout,
                )
            )
            receptive_field += (kernel_size - 1) * dilation

        self.receptive_field = receptive_field

        # ── Output head ─────────────────────────────────────────────────────
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(skip_ch, end_ch,      kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(end_ch,  num_pred * out_channels, kernel_size=(1, 1)),
        )

    @staticmethod
    def _normalise(adj: np.ndarray) -> torch.Tensor:
        """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
        adj = adj.copy()
        np.fill_diagonal(adj, 0.0)
        d   = adj.sum(axis=1, keepdims=True).clip(min=1)
        adj = adj / d
        return torch.tensor(adj, dtype=torch.float32)

    def _get_supports(self) -> list[torch.Tensor]:
        supports = [self.adj_fwd, self.adj_bwd]
        if self.adaptive:
            e1   = self.node_emb1.weight
            e2   = self.node_emb2.weight
            adp  = F.softmax(F.relu(e1 @ e2.T), dim=1)
            supports.append(adp)
        return supports

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, T, N, C)  — historical window
        returns (B, num_pred, N, C)
        """
        # Rearrange to (B, C, N, T)
        x = x.permute(0, 3, 2, 1)     # (B, C, N, T)

        # Pad so temporal dim >= receptive field
        if x.size(-1) < self.receptive_field:
            pad = self.receptive_field - x.size(-1)
            x   = F.pad(x, (pad, 0))

        x        = self.start_conv(x)  # (B, residual_ch, N, T)
        supports = self._get_supports()

        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x, supports)
            skip_sum = skip_sum + skip[..., -1:]   # keep last time step only

        # skip_sum: (B, skip_ch, N, 1)
        # out: (B, num_pred*C_out, N, 1)
        out = self.out(skip_sum)
        B, _, N, _ = out.shape
        out = out.reshape(B, self.num_pred, -1, N)  # (B, num_pred, C_out, N)
        out = out.permute(0, 1, 3, 2)               # (B, num_pred, N, C_out)
        return out


def build_graph_wavenet(num_nodes, adj_mx=None, num_pred=12, num_his=12,
                        device="cpu") -> GraphWaveNet:
    model = GraphWaveNet(
        num_nodes   = num_nodes,
        in_channels = 1,
        out_channels= 1,
        num_pred    = num_pred,
        num_his     = num_his,
        adj_mx      = adj_mx,
        residual_ch = 32,
        dilation_ch = 32,
        skip_ch     = 256,
        end_ch      = 512,
        num_layers  = 8,
        dropout     = 0.3,
        adaptive    = True,
    )
    return model.to(device)
