"""
quick_demo.py — Smoke-test all models on SYNTHETIC data (no real dataset needed).

Generates a random graph + random speed time-series, trains each model for
5 epochs, and prints metrics. Use this to verify the environment is correct
before downloading real data.

Run:
    python quick_demo.py
"""
from __future__ import annotations
import numpy as np
import torch
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.graph_wavenet import build_graph_wavenet
from models.dcrnn         import build_dcrnn
from models.stgcn         import build_stgcn
from models.baselines     import build_lstm
from utils.trainer        import masked_mae, compute_metrics_np

torch.manual_seed(42)
np.random.seed(42)

# ── Synthetic config ─────────────────────────────────────────────────────────
N        = 20       # tiny: 20 nodes (sensors)
T_in     = 12       # 12 historical steps
T_out    = 12       # predict 12 future steps
B        = 16       # batch size
EPOCHS   = 5
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Sparse random adjacency
adj = (np.random.rand(N, N) > 0.7).astype(np.float32)
np.fill_diagonal(adj, 0)

print(f"\n{'='*55}")
print(f"  Quick Demo — Synthetic  nodes={N}  device={DEVICE}")
print(f"{'='*55}\n")


def make_batch():
    x = torch.randn(B, T_in,  N, 1, device=DEVICE)
    y = torch.randn(B, T_out, N, 1, device=DEVICE)
    return x, y


models = {
    "GraphWaveNet": build_graph_wavenet(N, adj, T_out, T_in, DEVICE),
    "DCRNN":        build_dcrnn(N, adj, T_out, DEVICE),
    "STGCN":        build_stgcn(N, adj, T_out, T_in, DEVICE),
    "LSTM":         build_lstm(N, T_out, DEVICE),
}

for name, model in models.items():
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = sum(p.numel() for p in model.parameters())
    print(f"  >> {name:<18}  params={params:>8,}")
    for ep in range(1, EPOCHS + 1):
        model.train()
        x, y  = make_batch()
        opt.zero_grad()
        pred  = model(x)
        # Align shapes if needed
        if pred.shape != y.shape:
            pred = pred.expand_as(y)
        loss  = masked_mae(pred, y)
        loss.backward()
        opt.step()
        if ep == EPOCHS:
            print(f"    Epoch {ep}: loss={loss.item():.4f}  "
                  f"output_shape={tuple(pred.shape)}  OK")
    print()

print("All models ran successfully on synthetic data!")
print("  Next: python data/download_data.py  ->  python data/preprocess.py  ->  python main_train.py")
