"""
Visualisation utilities:
  - Plot training loss curves
  - Per-horizon MAE/RMSE/MAPE bar charts comparing all models
  - Spatial heatmap of per-node MAE on the sensor network
  - Time-series prediction plot for selected nodes
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "GraphWaveNet": "#2196F3",
    "DCRNN":        "#4CAF50",
    "STGCN":        "#FF9800",
    "LSTM":         "#E91E63",
    "ARIMA":        "#9E9E9E",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training loss curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(histories: dict[str, dict], save_path: str):
    """
    histories: {model_name: {"train_loss": [...], "val_loss": [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        color = MODEL_COLORS.get(name, None)
        axes[0].plot(h["train_loss"], label=name, color=color)
        axes[1].plot(h["val_loss"],   label=name, color=color)

    for ax, title in zip(axes, ["Train Loss (MAE)", "Validation Loss (MAE)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Masked MAE (normalised)")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-horizon metric comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_horizon_comparison(
    results:  dict[str, dict],  # {model: {horizon_min: {MAE, RMSE, MAPE}}}
    dataset:  str,
    save_dir: str,
):
    horizons = [15, 30, 60]
    metrics  = ["MAE", "RMSE", "MAPE"]
    models   = list(results.keys())
    x        = np.arange(len(horizons))
    width    = 0.8 / len(models)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, mdl in enumerate(models):
            vals   = [results[mdl].get(h, {}).get(metric, np.nan) for h in horizons]
            offset = (i - len(models) / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=mdl, color=MODEL_COLORS.get(mdl, None),
                            alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}-min" for h in horizons])
        ax.set_xlabel("Prediction Horizon")
        unit = "%" if metric == "MAPE" else "mph"
        ax.set_ylabel(f"{metric} ({unit})")
        ax.set_title(f"{metric} Comparison — {dataset}")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{dataset.lower()}_{metric.lower()}_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial heatmap of per-node MAE
# ─────────────────────────────────────────────────────────────────────────────
def plot_node_error_heatmap(
    per_node_mae: dict[str, np.ndarray],  # {model_name: (N,) array}
    sensor_coords: np.ndarray | None,     # (N, 2) lon/lat; None → scatter by index
    dataset: str,
    save_dir: str,
):
    """
    If sensor_coords provided: scatter plot coloured by error.
    Otherwise: bar chart sorted by error for the best GNN vs LSTM.
    """
    n_models = len(per_node_mae)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (name, node_mae) in zip(axes, per_node_mae.items()):
        if sensor_coords is not None:
            sc = ax.scatter(
                sensor_coords[:, 0], sensor_coords[:, 1],
                c=node_mae, cmap="RdYlGn_r", s=30, alpha=0.85,
                norm=mcolors.Normalize(vmin=node_mae.min(), vmax=node_mae.max()),
            )
            plt.colorbar(sc, ax=ax, label="MAE (mph)")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        else:
            idx = np.argsort(node_mae)
            ax.bar(range(len(node_mae)), node_mae[idx],
                   color=cm.RdYlGn_r(np.linspace(0, 1, len(node_mae))))
            ax.set_xlabel("Sensor (sorted by MAE)")
            ax.set_ylabel("Per-node MAE (mph)")

        ax.set_title(f"{name} — {dataset}\nSpatial Error Map")

    plt.suptitle(f"Per-Node Prediction Error Heatmap — {dataset}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset.lower()}_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-series prediction vs ground truth
# ─────────────────────────────────────────────────────────────────────────────
def plot_prediction_timeseries(
    all_preds:  dict[str, np.ndarray],  # {model: (B, T_out, N, 1)}
    targets:    np.ndarray,             # (B, T_out, N, 1)
    scaler:     tuple[float, float],
    dataset:    str,
    save_dir:   str,
    node_idx:   int = 0,
    horizon:    int = 12,               # which step (1-indexed)
    num_samples: int = 288,             # one day of 5-min samples
):
    """
    Plot the 60-min ahead speed prediction vs ground truth over one day.
    """
    mean, std = scaler
    def denorm(arr):
        return arr * std + mean

    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(min(num_samples, targets.shape[0]))

    target_series = denorm(targets[:len(t), horizon - 1, node_idx, 0])
    ax.plot(t, target_series, color="black", linewidth=1.5,
            label="Ground Truth", zorder=10)

    for name, preds in all_preds.items():
        pred_series = denorm(preds[:len(t), horizon - 1, node_idx, 0])
        ax.plot(t, pred_series,
                color=MODEL_COLORS.get(name, None),
                linewidth=1.2, alpha=0.8, label=name)

    ax.set_xlabel("Time Step (5-min intervals)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"{dataset} — Sensor {node_idx} — {horizon*5}-min Ahead Prediction")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset.lower()}_node{node_idx}_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table (printed + saved as CSV)
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(results: dict[str, dict], dataset: str):
    horizons = [15, 30, 60]
    header = f"\n{'='*72}\n  Results: {dataset}\n{'='*72}"
    print(header)
    row = f"{'Model':<18}" + "".join(
        f"{'@'+str(h)+'min':>12}" for h in horizons for _ in ["MAE","RMSE","MAPE"]
    )
    sub = f"{'':18}" + "".join(
        f"{'MAE':>12}{'RMSE':>12}{'MAPE%':>12}" for _ in horizons
    )
    print(row)
    print(sub)
    print("-" * 72)
    for model, res in results.items():
        row_data = f"{model:<18}"
        for h in horizons:
            m = res.get(h, {})
            row_data += f"{m.get('MAE','N/A'):>12}{m.get('RMSE','N/A'):>12}{m.get('MAPE','N/A'):>12}"
        print(row_data)
    print("=" * 72)
