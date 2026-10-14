"""
Preprocess METR-LA and PEMS-BAY datasets.
Outputs: train/val/test .npz files with (samples, timesteps, nodes, features).
"""
import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_adj(adj_path: str):
    """Load adjacency matrix from pickle."""
    with open(adj_path, "rb") as f:
        adj_mx = pickle.load(f, encoding="latin1")
    # DCRNN format: (sensor_ids, sensor_id_to_ind, adj_matrix)
    if isinstance(adj_mx, tuple):
        return adj_mx[-1].astype(np.float32)
    return np.array(adj_mx, dtype=np.float32)


def seq2instance(data: np.ndarray, num_his: int, num_pred: int):
    """Slide a window over time axis to create (X, Y) pairs."""
    num_step = data.shape[0]
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros((num_sample, num_his, data.shape[1], data.shape[2]), dtype=np.float32)
    y = np.zeros((num_sample, num_pred, data.shape[1], data.shape[2]), dtype=np.float32)
    for i in range(num_sample):
        x[i] = data[i : i + num_his]
        y[i] = data[i + num_his : i + num_his + num_pred]
    return x, y


def preprocess_dataset(
    h5_path: str,
    adj_path: str,
    output_dir: str,
    num_his: int = 12,      # 1-hour history  (5-min intervals × 12 = 60 min)
    num_pred: int = 12,     # predict 15/30/60 min ahead = steps 3, 6, 12
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    missing_value: float = 0.0,
):
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load raw speed data (time × nodes) -----
    df = pd.read_hdf(h5_path)
    data = df.values.astype(np.float32)          # (T, N)
    data = data[:, :, np.newaxis]               # (T, N, 1)

    # Replace missing (0-speed) with NaN → forward-fill
    data[data == missing_value] = np.nan
    df_tmp = pd.DataFrame(data[:, :, 0])
    df_tmp.fillna(method="ffill", inplace=True)
    df_tmp.fillna(method="bfill", inplace=True)
    data[:, :, 0] = df_tmp.values

    # ----- Z-score normalise -----
    mean = data.mean()
    std  = data.std()
    data_norm = (data - mean) / (std + 1e-8)

    # ----- Train / Val / Test split -----
    T = data_norm.shape[0]
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))

    train = data_norm[:t1]
    val   = data_norm[t1:t2]
    test  = data_norm[t2:]

    # ----- Sliding window -----
    x_train, y_train = seq2instance(train, num_his, num_pred)
    x_val,   y_val   = seq2instance(val,   num_his, num_pred)
    x_test,  y_test  = seq2instance(test,  num_his, num_pred)

    # ----- Load adjacency -----
    adj = load_adj(adj_path)

    # ----- Save -----
    np.savez_compressed(
        os.path.join(output_dir, "train.npz"), x=x_train, y=y_train
    )
    np.savez_compressed(
        os.path.join(output_dir, "val.npz"),   x=x_val,   y=y_val
    )
    np.savez_compressed(
        os.path.join(output_dir, "test.npz"),  x=x_test,  y=y_test
    )
    np.save(os.path.join(output_dir, "adj.npy"), adj)
    np.save(os.path.join(output_dir, "scaler.npy"),
            np.array([mean, std], dtype=np.float32))

    print(f"  Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")
    print(f"  Adj  : {adj.shape}")
    print(f"  Mean={mean:.4f}  Std={std:.4f}")
    return mean, std


if __name__ == "__main__":
    print("=== Preprocessing METR-LA ===")
    preprocess_dataset(
        h5_path  = os.path.join(DATA_DIR, "metr-la", "metr_la.h5"),
        adj_path = os.path.join(DATA_DIR, "metr-la", "adj_mx.pkl"),
        output_dir = os.path.join(DATA_DIR, "metr-la", "processed"),
    )

    print("=== Preprocessing PEMS-BAY ===")
    preprocess_dataset(
        h5_path  = os.path.join(DATA_DIR, "pems-bay", "pems_bay.h5"),
        adj_path = os.path.join(DATA_DIR, "pems-bay", "adj_mx_bay.pkl"),
        output_dir = os.path.join(DATA_DIR, "pems-bay", "processed"),
    )
    print("Done.")
