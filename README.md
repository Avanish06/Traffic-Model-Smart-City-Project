# Traffic Congestion Prediction — ST-GNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avanish06/Traffic-Model-Smart-City-Project/blob/main/Traffic_Prediction_Colab.ipynb)

Real-time traffic speed forecasting using Spatio-Temporal Graph Neural Networks.  
Models road networks as graphs (sensors = nodes, roads = edges) and predicts traffic speed **15 / 30 / 60 minutes ahead** using three GNN architectures compared against LSTM and ARIMA baselines.

---

## Project Structure

```
traffic-stgnn/
├── data/
│   ├── download_data.py      # Download METR-LA & PEMS-BAY
│   └── preprocess.py         # Normalise, sliding-window, train/val/test split
├── models/
│   ├── graph_wavenet.py      # Graph WaveNet (Wu et al., 2019)
│   ├── dcrnn.py              # DCRNN (Li et al., 2018)
│   ├── stgcn.py              # ST-GCN (Yu et al., 2018)
│   └── baselines.py          # LSTM + ARIMA
├── utils/
│   ├── trainer.py            # Train loop, metrics (MAE/RMSE/MAPE)
│   └── visualise.py          # All plots: loss curves, heatmaps, bar charts
├── outputs/                  # Saved models & plots (auto-created)
├── main_train.py             # Master training script
├── quick_demo.py             # Smoke-test on synthetic data
└── requirements.txt
```

---

## Quick Start

### 1. Set up environment

```powershell
# Create and activate conda env
conda create -n stgnn python=3.11 -y
conda activate stgnn

# Install PyTorch with CUDA 12.1 (for RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify installation

```powershell
python quick_demo.py
```

Expected output:
```
  ▶ GraphWaveNet       params=  309,537
  ▶ DCRNN              params=  220,161
  ▶ STGCN              params=   98,112
  ▶ LSTM               params=  318,465
✓ All models ran successfully on synthetic data!
```

### 3. Download datasets

```powershell
python data/download_data.py
python data/preprocess.py
```

> **Manual download** (if auto-download fails):
> - [DCRNN official repo](https://github.com/liyaguang/DCRNN) → `data/metr-la/metr_la.h5` + `adj_mx.pkl`
> - PEMS-BAY → `data/pems-bay/pems_bay.h5` + `adj_mx_bay.pkl`

### 4. Train all models

```powershell
# METR-LA  (all 3 GNNs + LSTM)
python main_train.py --dataset metr-la --epochs 100

# With ARIMA baseline too (adds ~30 min CPU time)
python main_train.py --dataset metr-la --epochs 100 --arima

# PEMS-BAY
python main_train.py --dataset pems-bay --epochs 100
```

### 5. Train a single model

```powershell
python main_train.py --dataset metr-la --models graphwavenet --epochs 100
python main_train.py --dataset metr-la --models dcrnn --epochs 100
python main_train.py --dataset metr-la --models stgcn --epochs 100
python main_train.py --dataset metr-la --models lstm --epochs 100
```

---

## Datasets

| Dataset  | Sensors | Location     | Duration          | Interval |
|----------|---------|--------------|-------------------|----------|
| METR-LA  | 207     | Los Angeles  | Mar–Jun 2012      | 5 min    |
| PEMS-BAY | 325     | Bay Area, CA | Jan–Jun 2017      | 5 min    |

**Preprocessing pipeline**:
1. Load HDF5 speed data (time × nodes)
2. Forward/backward fill missing values (sensor outages)
3. Z-score normalise (mean/std saved for denormalisation)
4. Train(70%) / Val(10%) / Test(20%) split
5. Sliding window: 12 historical steps → 12 prediction steps

---

## Models

### Graph WaveNet (Wu et al., IJCAI 2019)
- **Self-adaptive adjacency**: learns graph structure from data
- **Dilated causal TCN**: exponentially growing receptive field [1,2,4,8,…]
- **Gated activation**: tanh ⊙ sigmoid for feature selection

### DCRNN (Li et al., ICLR 2018)
- **Diffusion convolution**: bidirectional random walk on graph
- **GRU encoder–decoder**: seq2seq with scheduled sampling
- **Multi-step prediction**: all horizons in one forward pass

### ST-GCN (Yu et al., IJCAI 2018)
- **Chebyshev spectral GCN**: polynomial approximation order K=3
- **Gated temporal conv (GLU)**: tanh ⊙ sigmoid across time axis
- **ST Blocks**: TCN → GCN → TCN sandwiches

### Baselines
- **LSTM**: 2-layer, 256 hidden, trained jointly over all nodes
- **ARIMA(2,1,2)**: fitted per-node, forecasts from last window

---

## Metrics

All metrics computed on **denormalised** speed (mph):

| Metric | Formula |
|--------|---------|
| MAE    | mean\|ŷ - y\| (masked where y=0) |
| RMSE   | √mean[(ŷ - y)²] (masked) |
| MAPE   | 100 × mean\|ŷ-y\|/\|y\| (%) |

Evaluated at **3 horizons**: 15-min (step 3), 30-min (step 6), 60-min (step 12).

---

## Expected Results (METR-LA)

Based on published results (approximate):

| Model        | MAE@15 | RMSE@15 | MAE@60 | RMSE@60 |
|-------------|--------|---------|--------|---------|
| GraphWaveNet | 2.69   | 5.15    | 3.53   | 7.37    |
| DCRNN        | 2.77   | 5.38    | 3.60   | 7.59    |
| ST-GCN       | 3.10   | 5.96    | 4.05   | 8.50    |
| LSTM         | 3.44   | 6.30    | 5.21   | 9.57    |
| ARIMA        | 3.99   | 8.21    | 5.73   | 11.60   |

---

## Outputs

After training, `outputs/` contains:
- `metr-la_graphwavenet_best.pt` — best model checkpoint
- `metr-la_results.json` — all metrics as JSON
- `plots/metr-la/loss_curves.png` — training dynamics
- `plots/metr-la/metr-la_mae_comparison.png` — per-horizon bar chart
- `plots/metr-la/metr-la_heatmap.png` — spatial error heatmap
- `plots/metr-la/metr-la_node0_timeseries.png` — prediction vs ground truth

---

## Hardware Requirements

| Component | Minimum   | Recommended |
|-----------|-----------|-------------|
| GPU VRAM  | 4 GB      | 8 GB+       |
| RAM       | 8 GB      | 16 GB       |
| Storage   | 2 GB      | 5 GB        |
| CUDA      | 11.8+     | 12.1+       |

RTX 3050 (4GB) works with `--batch_size 16` if OOM at 32.

---

## References

1. Wu et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019.
2. Li et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018.
3. Yu et al. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting." IJCAI 2018.
4. METR-LA dataset: Jagadish et al. (2014); curated by Li et al. (2018).
5. PEMS-BAY dataset: Caltrans Performance Measurement System.
