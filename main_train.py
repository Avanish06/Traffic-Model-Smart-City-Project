"""
main_train.py — Master training & evaluation script.

Usage:
    python main_train.py --dataset metr-la --epochs 100
    python main_train.py --dataset pems-bay --epochs 100
    python main_train.py --dataset metr-la --models graphwavenet dcrnn --epochs 50
    python main_train.py --dataset metr-la --arima        # also run ARIMA baseline
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import torch

# ── Resolve paths ────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
UTILS  = os.path.join(ROOT, "utils")
OUT    = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

from models.graph_wavenet import build_graph_wavenet
from models.dcrnn         import build_dcrnn
from models.stgcn         import build_stgcn
from models.baselines     import build_lstm, ARIMABaseline
from utils.trainer        import make_loaders, train_model, evaluate, compute_metrics_np
from utils.visualise      import (
    plot_loss_curves, plot_horizon_comparison,
    plot_node_error_heatmap, plot_prediction_timeseries, print_results_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
DATASET_INFO = {
    "metr-la":  {"adj": "adj.npy",     "nodes": 207},
    "pems-bay": {"adj": "adj.npy",     "nodes": 325},
}

MODEL_BUILDERS = {
    "graphwavenet": build_graph_wavenet,
    "dcrnn":        build_dcrnn,
    "stgcn":        build_stgcn,
    "lstm":         lambda num_nodes, adj_mx, num_pred, device: build_lstm(num_nodes, num_pred, device),
}

DISPLAY_NAMES = {
    "graphwavenet": "GraphWaveNet",
    "dcrnn":        "DCRNN",
    "stgcn":        "STGCN",
    "lstm":         "LSTM",
}


def load_dataset(dataset: str):
    """Load adj matrix, scaler, and return processed_dir."""
    info = DATASET_INFO[dataset]
    proc = os.path.join(DATA, dataset, "processed")
    if not os.path.isdir(proc):
        raise FileNotFoundError(
            f"Processed data not found at {proc}.\n"
            f"Run:  python data/download_data.py\n"
            f"Then: python data/preprocess.py"
        )
    adj    = np.load(os.path.join(proc, "adj.npy"))
    scaler = tuple(np.load(os.path.join(proc, "scaler.npy")).tolist())
    return proc, adj, scaler, info["nodes"]


def run_nn_model(
    name:       str,
    dataset:    str,
    proc_dir:   str,
    adj:        np.ndarray,
    num_nodes:  int,
    device:     str,
    epochs:     int,
    batch_size: int,
    num_pred:   int = 12,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Train one NN model; return (metrics, preds, targets, history)."""
    train_loader, val_loader, test_loader = make_loaders(
        proc_dir, batch_size=batch_size
    )
    display = DISPLAY_NAMES[name]
    save_pt  = os.path.join(OUT, f"{dataset}_{name}_best.pt")

    builder = MODEL_BUILDERS[name]
    try:
        model = builder(num_nodes=num_nodes, adj_mx=adj,
                        num_pred=num_pred, device=device)
    except TypeError:
        # LSTM builder doesn't accept adj_mx
        model = builder(num_nodes=num_nodes, adj_mx=adj,
                        num_pred=num_pred, device=device)

    history = train_model(
        model, train_loader, val_loader,
        device    = device,
        epochs    = epochs,
        lr        = 1e-3,
        patience  = 15,
        use_amp   = device.startswith("cuda"),
        save_path = save_pt,
        model_name= display,
    )

    _, preds, targets = evaluate(model, test_loader, device)
    return preds, targets, history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ST-GNN Traffic Forecasting")
    parser.add_argument("--dataset",    default="metr-la",
                        choices=["metr-la", "pems-bay"])
    parser.add_argument("--models",     nargs="+",
                        default=["graphwavenet", "dcrnn", "stgcn", "lstm"],
                        choices=["graphwavenet", "dcrnn", "stgcn", "lstm"])
    parser.add_argument("--arima",      action="store_true",
                        help="Also run ARIMA baseline (slow, ~30 min)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--num_pred",   type=int,   default=12)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n  Device: {device}")
    if device.startswith("cuda"):
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dataset = args.dataset
    proc_dir, adj, scaler, num_nodes = load_dataset(dataset)
    print(f"  Dataset: {dataset}  nodes={num_nodes}  scaler={scaler}")

    # Horizon steps for 5-min data: 3→15min, 6→30min, 12→60min
    horizon_steps = [s for s in [3, 6, 12] if s <= args.num_pred]

    all_results  = {}
    all_histories= {}
    all_preds_map= {}
    targets_ref  = None

    # ── Neural models ────────────────────────────────────────────────────────
    for model_name in args.models:
        print(f"\n>> {DISPLAY_NAMES[model_name]}")
        preds, targets, history = run_nn_model(
            name       = model_name,
            dataset    = dataset,
            proc_dir   = proc_dir,
            adj        = adj,
            num_nodes  = num_nodes,
            device     = device,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            num_pred   = args.num_pred,
        )
        metrics = compute_metrics_np(preds, targets, scaler, horizon_steps)
        all_results[DISPLAY_NAMES[model_name]]   = metrics
        all_histories[DISPLAY_NAMES[model_name]] = history
        all_preds_map[DISPLAY_NAMES[model_name]] = preds
        targets_ref = targets

    # -- ARIMA baseline -------------------------------------------------------
    if args.arima:
        print("\n>> ARIMA")
        raw_train = np.load(os.path.join(proc_dir, "train.npz"))["x"]
        x_test    = np.load(os.path.join(proc_dir, "test.npz"))["x"]
        y_test    = np.load(os.path.join(proc_dir, "test.npz"))["y"]

        mean, std = scaler
        train_raw = raw_train[:, :, :, 0] * std + mean   # (B, T, N) denorm

        arima = ARIMABaseline(order=(2, 1, 2))
        arima.fit(train_raw[-500:].reshape(-1, num_nodes))

        arima_preds = arima.batch_predict(x_test, args.num_pred, scaler)
        arima_metrics = compute_metrics_np(arima_preds, y_test, scaler, horizon_steps)
        all_results["ARIMA"]   = arima_metrics
        all_preds_map["ARIMA"] = arima_preds
        if targets_ref is None:
            targets_ref = y_test

    # -- Print table ----------------------------------------------------------
    print_results_table(all_results, dataset)

    # -- Save JSON ------------------------------------------------------------
    json_path = os.path.join(OUT, f"{dataset}_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved -> {json_path}")

    # -- Plots ----------------------------------------------------------------
    print("\n  Generating plots ...")
    plot_dir = os.path.join(OUT, "plots", dataset)
    os.makedirs(plot_dir, exist_ok=True)

    if all_histories:
        plot_loss_curves(all_histories,
                         os.path.join(plot_dir, "loss_curves.png"))

    # Horizon comparison
    nn_results = {k: v for k, v in all_results.items()
                  if k in [DISPLAY_NAMES[m] for m in args.models] + (["ARIMA"] if args.arima else [])}
    plot_horizon_comparison(nn_results, dataset, plot_dir)

    # Per-node heatmap (best GNN vs LSTM)
    node_maes = {}
    for name, preds in all_preds_map.items():
        mean_v, std_v = scaler
        p = preds[:, 5, :, 0] * std_v + mean_v    # 30-min horizon
        t = targets_ref[:, 5, :, 0] * std_v + mean_v
        node_maes[name] = np.abs(p - t).mean(axis=0)   # (N,)

    plot_node_error_heatmap(node_maes, None, dataset, plot_dir)

    # Time-series for node 0
    if targets_ref is not None:
        plot_prediction_timeseries(
            all_preds_map, targets_ref, scaler, dataset, plot_dir,
            node_idx=0, horizon=12
        )

    print(f"\n[DONE] Outputs in: {OUT}")


if __name__ == "__main__":
    main()
