#!/usr/bin/env python
"""
02_train.py

- Baseline modellek (mean, current_delay, Linear Regression) tanítása és mentése
- GraphSAGE GNN tanítása és mentése

A bemenet:  /app/data/processed_dataset.csv   (01_data_processing.py hozza létre)
Az eredmények: /app/output/baselines/* , /app/output/models/*
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/app/output")
BASELINE_DIR = OUTPUT_DIR / "baselines"
MODELS_DIR = OUTPUT_DIR / "models"

FEATURES = ["delay_seconds_calc", "hour", "weekday"]
TARGET = "y_end_delay_calc"
STOP_COL = "last_stop_id"
K_HOP = 2  # k-hop subgraph GNN-hez


# ------------------------------
# GNN modell
# ------------------------------
class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch, center_pos):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # batch központi node embedding kiválasztása
        # center_pos: [num_graphs] ahol mindegyik grafon belül a center node indexe
        # a batch-ben a node-ok össze vannak fűzve, ezért offset kell
        num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 0
        # node szám grafonként
        counts = torch.bincount(batch, minlength=num_graphs)
        offsets = torch.cumsum(counts, dim=0) - counts  # prefix sum
        center_global = offsets + center_pos.view(-1)

        x_center = x[center_global]  # [num_graphs, hidden]
        out = self.lin(x_center).squeeze(-1)
        return out


# ------------------------------
# Row-level GNN Dataset
# ------------------------------
class RowGNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        stop_id_to_idx: dict,
        khop_cache: dict,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        features: list,
        target: str,
        stop_col: str,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.stop_id_to_idx = stop_id_to_idx
        self.khop_cache = khop_cache
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.features = features
        self.target = target
        self.stop_col = stop_col
        self.in_dim = len(features) + 1  # +1 a center_flag-nek

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        stop_id = str(row[self.stop_col])
        center_idx = self.stop_id_to_idx.get(stop_id, None)
        if center_idx is None:
            x = torch.zeros((1, self.in_dim), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            y_scaled = self.y_scaler.transform([[row[self.target]]])[0, 0]

            d = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([y_scaled], dtype=torch.float32),
                center_pos = torch.tensor([0], dtype=torch.long)
            )
 
            return d


        nodes, edge_index_sub, center_pos = self.khop_cache[center_idx]

        # center node feature scaling
        x_raw = row[self.features].values.astype(np.float32).reshape(1, -1)
        x_scaled = self.x_scaler.transform(x_raw)[0]

        # subgraph node features: mindenhol 0, center kapja a feature-t és center_flag=1
        x_sub = np.zeros((nodes.shape[0], self.in_dim), dtype=np.float32)
        x_sub[center_pos, : len(self.features)] = x_scaled
        x_sub[center_pos, -1] = 1.0  # center flag

        y_scaled = self.y_scaler.transform([[row[self.target]]])[0, 0]

        data = Data(
            x=torch.from_numpy(x_sub),
            edge_index=edge_index_sub,
            y=torch.tensor([y_scaled], dtype=torch.float32),
            center_pos = torch.tensor([center_pos], dtype=torch.long)
        )
    
        return data


# ------------------------------
# GNN segédfüggvények
# ------------------------------
def build_stop_graph(stop_times_path: Path):
    """
    stop_times.txt alapján felépítjük a megálló-gráfot:
    - node: stop_id
    - edge: egymást követő megállók egy trip-ben (oda-vissza)
    """
    # stop_id-t eleve stringként olvassuk be
    st = pd.read_csv(stop_times_path, dtype={"stop_id": str})
    print("[02_train] Loaded stop_times.txt, rows:", len(st))
    st = st[["trip_id", "stop_id", "stop_sequence"]]
    st = st.sort_values(["trip_id", "stop_sequence"])

    # egyedi stop_id-k és mapping indexre (string -> int index)
    unique_stops = pd.Index(st["stop_id"].unique())
    stop_id_to_idx = {s: i for i, s in enumerate(unique_stops)}
    num_nodes = len(unique_stops)

    # élek: egymást követő megállók
    edges_src = []
    edges_dst = []
    for _, group in st.groupby("trip_id"):
        stops = group["stop_id"].astype(str).values
        for i in range(len(stops) - 1):
            u = stop_id_to_idx[stops[i]]
            v = stop_id_to_idx[stops[i + 1]]
            edges_src.append(u)
            edges_dst.append(v)
            edges_src.append(v)
            edges_dst.append(u)  # bidirectional

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return stop_id_to_idx, edge_index, num_nodes



def build_khop_cache(edge_index: torch.Tensor, num_nodes: int, k: int, used_node_indices=None):
    """
    Csak a ténylegesen használt node-okra építjük a k-hop cache-t.

    used_node_indices: iterable azokról a node indexekről, amelyeket tényleg használunk
    """
    cache = {}
    if used_node_indices is None:
        iterator = range(num_nodes)
    else:
        iterator = used_node_indices

    for center in iterator:
        nodes, edge_index_sub, _, _ = k_hop_subgraph(
            center, k, edge_index, relabel_nodes=True
        )
        center_pos = (nodes == center).nonzero(as_tuple=True)[0]
        center_pos = int(center_pos[0]) if len(center_pos) > 0 else 0
        cache[center] = (nodes, edge_index_sub, center_pos)
    return cache



# ------------------------------
# fő függvény
# ------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Feldolgozott dataset beolvasása
    data_path = DATA_DIR / "processed_dataset.csv"
    df = pd.read_csv(data_path, parse_dates=["vehicle_timestamp"])
    df = df.sort_values("vehicle_timestamp")
    df = df.dropna(subset=FEATURES + [TARGET] + [STOP_COL]).copy()
    df[STOP_COL] = df[STOP_COL].astype(str)
    # 2) Időalapú split
    MAX_ITER_ROWS = 100000
    if MAX_ITER_ROWS != -1:
        df = df.head(MAX_ITER_ROWS)
        print(f"[02_train] Limiting to first {MAX_ITER_ROWS} rows for faster iteration.")
    else:
        print(f"[02_train] Using all {len(df)} rows.")

    train_end = df["vehicle_timestamp"].quantile(0.70)
    val_end = df["vehicle_timestamp"].quantile(0.85)

    train_df = df[df["vehicle_timestamp"] < train_end].copy()
    val_df = df[(df["vehicle_timestamp"] >= train_end) & (df["vehicle_timestamp"] < val_end)].copy()
    test_df = df[df["vehicle_timestamp"] >= val_end].copy()

    print("[02_train] rows:", len(df))
    print("[02_train] train/val/test:", len(train_df), len(val_df), len(test_df))

    # -------------------------------------------------
    # Baseline-ek (ugyanaz, mint eddig)
    # -------------------------------------------------
    y_test = test_df[TARGET].values.astype(np.float32)

    # Mean baseline
    mean_target = train_df[TARGET].mean()
    pred_mean = np.full_like(y_test, fill_value=mean_target, dtype=np.float32)
    mae_mean = mean_absolute_error(y_test, pred_mean)
    rmse_mean = mean_squared_error(y_test, pred_mean, squared=False)

    # Current delay baseline
    pred_current = test_df["delay_seconds_calc"].values.astype(np.float32)
    mae_cur = mean_absolute_error(y_test, pred_current)
    rmse_cur = mean_squared_error(y_test, pred_current, squared=False)

    # Linear Regression baseline
    X_train_lr = train_df[FEATURES].values
    y_train_lr = train_df[TARGET].values.astype(np.float32)
    X_val_lr = val_df[FEATURES].values
    y_val_lr = val_df[TARGET].values.astype(np.float32)
    X_test_lr = test_df[FEATURES].values

    lr_scaler = StandardScaler()
    X_train_lr_s = lr_scaler.fit_transform(X_train_lr)
    X_val_lr_s = lr_scaler.transform(X_val_lr)
    X_test_lr_s = lr_scaler.transform(X_test_lr)

    lr = LinearRegression()
    lr.fit(X_train_lr_s, y_train_lr)

    pred_val_lr = lr.predict(X_val_lr_s)
    pred_test_lr = lr.predict(X_test_lr_s)

    val_mae_lr = mean_absolute_error(y_val_lr, pred_val_lr)
    val_rmse_lr = mean_squared_error(y_val_lr, pred_val_lr, squared=False)
    test_mae_lr = mean_absolute_error(y_test, pred_test_lr)
    test_rmse_lr = mean_squared_error(y_test, pred_test_lr, squared=False)

    print(f"[02_train] Baseline(mean)        MAE={mae_mean:.2f}  RMSE={rmse_mean:.2f}")
    print(f"[02_train] Baseline(current)     MAE={mae_cur:.2f}   RMSE={rmse_cur:.2f}")
    print(f"[02_train] LinearRegression VAL  MAE={val_mae_lr:.2f}   RMSE={val_rmse_lr:.2f}")
    print(f"[02_train] LinearRegression TEST MAE={test_mae_lr:.2f}  RMSE={test_rmse_lr:.2f}")

    # Baseline eredmények
    baseline_results = {
        "meta": {
            "features": FEATURES,
            "target": TARGET,
            "timestamp_col": "vehicle_timestamp",
        },
        "baseline_mean": {
            "model_type": "statistical",
            "mae": float(mae_mean),
            "rmse": float(rmse_mean),
        },
        "baseline_current_delay": {
            "model_type": "heuristic",
            "mae": float(mae_cur),
            "rmse": float(rmse_cur),
        },
        "linear_regression": {
            "model_type": "linear",
            "mae": float(test_mae_lr),
            "rmse": float(test_rmse_lr),
        },
    }

    # Mentés JSON + CSV
    json_path = BASELINE_DIR / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    rows = []
    for name, res in baseline_results.items():
        if name == "meta":
            continue
        rows.append(
            {
                "model": name,
                "model_type": res["model_type"],
                "MAE_seconds": res["mae"],
                "RMSE_seconds": res["rmse"],
            }
        )
    baseline_df = pd.DataFrame(rows)
    csv_path = BASELINE_DIR / "baseline_results.csv"
    baseline_df.to_csv(csv_path, index=False)

    joblib.dump(lr, BASELINE_DIR / "linear_regression.joblib")
    joblib.dump(lr_scaler, BASELINE_DIR / "linear_regression_scaler.joblib")

    print(f"[02_train] Saved baselines to: {BASELINE_DIR}")

    # -------------------------------------------------
    # GNN: GraphSAGE K=2 k-hop subgraph
    # -------------------------------------------------
    print("[02_train] Building stop graph for GNN...")
    stop_times_path = DATA_DIR / "stop_times.txt"
    stop_id_to_idx, edge_index, num_nodes = build_stop_graph(stop_times_path)
    all_used_stop_ids = pd.concat([
        train_df[STOP_COL],
        val_df[STOP_COL],
        test_df[STOP_COL],
    ]).astype(str).unique()

    used_node_indices = [
        stop_id_to_idx[s] for s in all_used_stop_ids if s in stop_id_to_idx
    ]

    print(f"[02_train] Number of used stops for GNN: {len(used_node_indices)} / {num_nodes}")

    khop_cache = build_khop_cache(edge_index, num_nodes, K_HOP, used_node_indices)

    # GNN scaler-ek (külön a baseline-tól)
    print("[02_train] Fitting GNN scalers...")
    x_scaler_gnn = StandardScaler()
    x_scaler_gnn.fit(train_df[FEATURES].values)
    y_scaler_gnn = StandardScaler()
    y_scaler_gnn.fit(train_df[[TARGET]].values)

    joblib.dump(x_scaler_gnn, BASELINE_DIR / "gnn_x_scaler.joblib")
    joblib.dump(y_scaler_gnn, BASELINE_DIR / "gnn_y_scaler.joblib")

    # PyG Dataset-ek
    gnn_train_ds = RowGNNDataset(
        train_df, stop_id_to_idx, khop_cache, x_scaler_gnn, y_scaler_gnn,
        FEATURES, TARGET, STOP_COL
    )
    gnn_val_ds = RowGNNDataset(
        val_df, stop_id_to_idx, khop_cache, x_scaler_gnn, y_scaler_gnn,
        FEATURES, TARGET, STOP_COL
    )
    gnn_test_ds = RowGNNDataset(
        test_df, stop_id_to_idx, khop_cache, x_scaler_gnn, y_scaler_gnn,
        FEATURES, TARGET, STOP_COL
    )

    train_loader = DataLoader(gnn_train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(gnn_val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(gnn_test_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[02_train] Using device: {device}")
    model = SimpleSAGE(in_channels=len(FEATURES) + 1, hidden_channels=64, num_layers=2)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_left = patience
    max_epochs = 4

    print("[02_train] Training GNN...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.batch, batch.center_pos)
            loss = loss_fn(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # validáció
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.batch, batch.center_pos)
                loss = loss_fn(preds, batch.y.view(-1))
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"[02_train][GNN] Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[02_train][GNN] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # modell mentése
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gnn_path = MODELS_DIR / "gnn_sage_k2.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": len(FEATURES) + 1,
            "hidden_channels": 64,
            "num_layers": 2,
            "k_hop": K_HOP,
        },
        gnn_path,
    )
    print(f"[02_train] Saved GNN model to {gnn_path}")


if __name__ == "__main__":
    print("[02_train] Starting training process...")
    #supress warnings
    import warnings
    warnings.filterwarnings("ignore")
    main()

