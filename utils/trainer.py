"""
Training and evaluation utilities.
Handles: dataset loading, DataLoaders, train loop, metrics (MAE, RMSE, MAPE),
         mixed-precision training on CUDA, and result saving.
"""
from __future__ import annotations
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TrafficDataset(Dataset):
    """Wraps pre-processed .npz files."""

    def __init__(self, npz_path: str):
        data     = np.load(npz_path)
        self.x   = torch.tensor(data["x"], dtype=torch.float32)  # (S, T_in, N, C)
        self.y   = torch.tensor(data["y"], dtype=torch.float32)  # (S, T_out, N, C)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loaders(processed_dir: str, batch_size: int = 16, num_workers: int = 0):
    train_ds = TrafficDataset(os.path.join(processed_dir, "train.npz"))
    val_ds   = TrafficDataset(os.path.join(processed_dir, "val.npz"))
    test_ds  = TrafficDataset(os.path.join(processed_dir, "test.npz"))

    kw = dict(batch_size=batch_size, num_workers=num_workers,
               pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def masked_mae(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0):
    mask  = (target != null_val).float()
    loss  = torch.abs(pred - target) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0):
    mask  = (target != null_val).float()
    loss  = (pred - target) ** 2 * mask
    mse   = loss.sum() / mask.sum().clamp(min=1)
    return torch.sqrt(mse)


def masked_mape(pred: torch.Tensor, target: torch.Tensor,
                null_val: float = 0.0, eps: float = 1e-5):
    mask  = (target.abs() > eps).float()
    loss  = torch.abs((pred - target) / target.abs().clamp(min=eps)) * mask
    return 100.0 * loss.sum() / mask.sum().clamp(min=1)


def compute_metrics_np(
    pred: np.ndarray,
    target: np.ndarray,
    scaler: tuple[float, float],
    horizon_steps: list[int] = (3, 6, 12),   # 15, 30, 60 min at 5-min resolution
) -> dict:
    """
    Denormalise predictions and targets, then compute per-horizon metrics.
    pred, target: (B, T_out, N, 1) normalised.
    Returns dict of {horizon_min: {MAE, RMSE, MAPE}}.
    """
    mean, std = scaler
    pred_raw   = pred[..., 0]   * std + mean    # (B, T_out, N)
    target_raw = target[..., 0] * std + mean

    results = {}
    horizon_map = {3: 15, 6: 30, 12: 60}
    for step in horizon_steps:
        p = pred_raw[:, step - 1, :]    # (B, N)
        t = target_raw[:, step - 1, :]
        mask = t > 0
        mae  = float(np.abs(p - t)[mask].mean())
        rmse = float(np.sqrt(((p - t) ** 2)[mask].mean()))
        mape = float((np.abs((p - t) / t.clip(min=1e-5)))[mask].mean() * 100)
        results[horizon_map.get(step, step * 5)] = {
            "MAE":  round(mae,  4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: str,
    scaler,
    epoch: int = 1,
    epochs: int = 1,
    teacher_forcing_ratio: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}/{epochs} [Train]", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                if hasattr(model, "forward") and "teacher_forcing_ratio" in \
                        model.forward.__code__.co_varnames:
                    pred = model(x, teacher_forcing_ratio=teacher_forcing_ratio)
                else:
                    pred = model(x)
                loss = masked_mae(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            pred = model(x)
            loss = masked_mae(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

        total_loss += loss.item() * x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: str,
    desc: str = "Evaluating",
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    pbar = tqdm(loader, desc=f"  {desc}", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = masked_mae(pred, y)
        total_loss  += loss.item() * x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return total_loss / len(loader.dataset), preds, targets


def train_model(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       str,
    epochs:       int  = 100,
    lr:           float = 1e-3,
    patience:     int   = 15,
    use_amp:      bool  = True,
    save_path:    str   = "best_model.pt",
    model_name:   str   = "Model",
) -> dict:
    """
    Full training loop with early stopping, LR scheduling, AMP.
    Returns history dict {train_loss, val_loss} per epoch.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5, min_lr=1e-5
    )
    amp_scaler = torch.amp.GradScaler("cuda") if (use_amp and torch.cuda.is_available() and device.startswith("cuda")) else None

    best_val   = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}

    print(f"\n{'='*60}", flush=True)
    print(f"  Training: {model_name}   device={device}", flush=True)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(1, epochs + 1):
        # Scheduled sampling decay (for DCRNN)
        tf_ratio = max(0.0, 0.5 * (1 - epoch / epochs))

        t0       = time.time()
        train_l  = train_one_epoch(model, train_loader, optimiser, device,
                                   amp_scaler, epoch=epoch, epochs=epochs,
                                   teacher_forcing_ratio=tf_ratio)
        val_l, *_ = evaluate(model, val_loader, device, desc="Validating")
        scheduler.step(val_l)
        elapsed  = time.time() - t0

        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={train_l:.4f}  val={val_l:.4f}  "
              f"lr={optimiser.param_groups[0]['lr']:.2e}  "
              f"t={elapsed:.1f}s", flush=True)

        if val_l < best_val:
            best_val   = val_l
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    # Reload best
    model.load_state_dict(torch.load(save_path, map_location=device))
    return history
