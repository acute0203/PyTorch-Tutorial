# -*- coding: utf-8 -*-
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf

# ============================ #
# Config
# ============================ #
symbol      = "AAPL"
start       = "2020-01-01"
end         = "2023-01-01"

seq_len     = 30
batch_size  = 64
epochs      = 120
hidden_size = 64
num_layers  = 2
dropout     = 0.2
lr          = 1e-3
clip_norm   = 1.0
seed        = 42
PLOT        = True   # 想看圖就 True

np.random.seed(seed)
torch.manual_seed(seed)

# ============================ #
# Device
# ============================ #
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and platform.system() == "Darwin":
        print("Using Apple MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_device()

# ============================ #
# Utils
# ============================ #
def create_xy(arr, seq_len):
    X, Y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        Y.append(arr[i+seq_len])
    return np.array(X), np.array(Y)

def directional_accuracy(series):
    return np.sign(series[1:] - series[:-1])

def evaluate_price_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    d_true = directional_accuracy(y_true)
    d_pred = directional_accuracy(y_pred)
    n = min(len(d_true), len(d_pred))
    da = np.mean((d_true[:n] == d_pred[:n]).astype(np.float64))
    return mse, mae, da

# ============================ #
# Data
# ============================ #
data = yf.download(symbol, start=start, end=end, progress=False)
close = data["Close"].values.astype(np.float64).reshape(-1, 1)
if len(close) < seq_len + 10:
    raise ValueError("Data too short.")

# ΔP 目標：比 log-return 訊號更強，重建時直接相加
delta = close[1:] - close[:-1]  # ΔP_t = P_t - P_{t-1}

# 以 ΔP 長度切分
train_ratio  = 0.8
n_delta      = len(delta)
train_n_del  = int(n_delta * train_ratio)  # 訓練 ΔP 筆數

# ============================ #
# Model: vanilla RNN
# ============================ #
class StackedRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, X_val, Y_val, epochs, lr, clip_norm, device):
    model.to(device)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    tr_losses, va_losses = [], []
    best_state = None
    best_val   = float("inf")

    for ep in range(1, epochs+1):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()
            run += loss.item() * xb.size(0)
        tr_losses.append(run / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            pv = model(X_val.to(device))
            val = crit(pv, Y_val.to(device)).item()
            va_losses.append(val)

        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            print(f"[{ep:03d}/{epochs}] train={tr_losses[-1]:.6f} | val={val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return tr_losses, va_losses

# ============================ #
# A) RNN-Price：直接預測價格（對照）
# ============================ #
# 嚴謹：MinMaxScaler 僅以訓練價格 fit
price_train_end_index = train_n_del  # 對齊 ΔP 的切法（ΔP[0]對應P[1]）
scaler_price = MinMaxScaler()
scaler_price.fit(close[:price_train_end_index+1])

price_scaled = scaler_price.transform(close)

X_price_train, Y_price_train = create_xy(price_scaled[:price_train_end_index+1], seq_len)
X_price_test,  Y_price_test  = create_xy(price_scaled[price_train_end_index - seq_len + 1:], seq_len)

X_price_train = torch.tensor(X_price_train, dtype=torch.float32)
Y_price_train = torch.tensor(Y_price_train, dtype=torch.float32)
X_price_test  = torch.tensor(X_price_test,  dtype=torch.float32)
Y_price_test  = torch.tensor(Y_price_test,  dtype=torch.float32)

train_loader_price = DataLoader(TensorDataset(X_price_train, Y_price_train),
                                batch_size=batch_size, shuffle=True, drop_last=False)

model_price = StackedRNN(input_size=1, hidden_size=hidden_size,
                         num_layers=num_layers, dropout=dropout, output_size=1)

print("\n=== Train: RNN-Price (direct price) ===")
_ , _ = train_model(model_price, train_loader_price, X_price_test, Y_price_test,
                    epochs, lr, clip_norm, device)

model_price.eval()
with torch.no_grad():
    pred_price_scaled = model_price(X_price_test.to(device)).cpu().numpy()
true_price_scaled = Y_price_test.cpu().numpy()

pred_price = scaler_price.inverse_transform(pred_price_scaled)
true_price = scaler_price.inverse_transform(true_price_scaled)

# Naive baseline（價格）：y_hat = 視窗最後一價
naive_price_scaled = X_price_test[:, -1, :].numpy()
naive_price = scaler_price.inverse_transform(naive_price_scaled)

# ============================ #
# B) RNN-Delta：預測 ΔP → 重建價格
# ============================ #
# 用 StandardScaler 使 ΔP 標準化（只以訓練期 fit）
scaler_dp = StandardScaler()
scaler_dp.fit(delta[:train_n_del])
delta_scaled = scaler_dp.transform(delta)

X_dp_train, Y_dp_train = create_xy(delta_scaled[:train_n_del], seq_len)
X_dp_test,  Y_dp_test  = create_xy(delta_scaled[train_n_del - seq_len:], seq_len)

X_dp_train = torch.tensor(X_dp_train, dtype=torch.float32)
Y_dp_train = torch.tensor(Y_dp_train, dtype=torch.float32)
X_dp_test  = torch.tensor(X_dp_test,  dtype=torch.float32)
Y_dp_test  = torch.tensor(Y_dp_test,  dtype=torch.float32)

train_loader_dp = DataLoader(TensorDataset(X_dp_train, Y_dp_train),
                             batch_size=batch_size, shuffle=True, drop_last=False)

model_dp = StackedRNN(input_size=1, hidden_size=hidden_size,
                      num_layers=num_layers, dropout=dropout, output_size=1)

print("\n=== Train: RNN-Delta (predict ΔP) ===")
_ , _ = train_model(model_dp, train_loader_dp, X_dp_test, Y_dp_test,
                    epochs, lr, clip_norm, device)

# 推論：ΔP（標準化）→ 還原 ΔP → 由 P0 累加重建價格
model_dp.eval()
with torch.no_grad():
    pred_dp_scaled = model_dp(X_dp_test.to(device)).cpu().numpy()
pred_dp = scaler_dp.inverse_transform(pred_dp_scaled)

P0 = close[train_n_del].item()  # 測試第一步前的實際價格
recon = [P0]
for dp in pred_dp.reshape(-1):
    recon.append(recon[-1] + dp)
pred_price_from_dp = np.array(recon[1:]).reshape(-1, 1)

true_price_from_dp = close[train_n_del+1 : train_n_del+1 + len(pred_price_from_dp)]

# Naive（ΔP=0 等價持有不動）
naive_price_from_dp = np.full_like(pred_price_from_dp, fill_value=P0)

# ============================ #
# 評估（價格尺度）& 僅輸出價格
# ============================ #
mse_p,  mae_p,  da_p  = evaluate_price_metrics(true_price,          pred_price)
mse_np, mae_np, da_np = evaluate_price_metrics(true_price,          naive_price)
mse_d,  mae_d,  da_d  = evaluate_price_metrics(true_price_from_dp,  pred_price_from_dp)
mse_nd, mae_nd, da_nd = evaluate_price_metrics(true_price_from_dp,  naive_price_from_dp)

print("\n===== TEST METRICS (Price-scale) =====")
print(f"[RNN-Price]   MSE={mse_p:.6f}  MAE={mae_p:.6f}  DirAcc={da_p:.3f}")
print(f"[Naive-Price] MSE={mse_np:.6f} MAE={mae_np:.6f} DirAcc={da_np:.3f}")
print(f"[RNN-Delta]   MSE={mse_d:.6f}  MAE={mae_d:.6f}  DirAcc={da_d:.3f}")
print(f"[Naive-ΔP=0]  MSE={mse_nd:.6f} MAE={mae_nd:.6f} DirAcc={da_nd:.3f}")

# 對齊到相同長度與日期索引，輸出「價格」表格
k = min(len(true_price_from_dp), len(pred_price), len(naive_price))
dates_aligned        = data.index[train_n_del+1 : train_n_del+1 + k]
true_price_aligned   = true_price_from_dp[:k].reshape(-1)
rnn_delta_price      = pred_price_from_dp[:k].reshape(-1)
rnn_direct_price     = pred_price[:k].reshape(-1)
naive_direct_price   = naive_price[:k].reshape(-1)

price_df = pd.DataFrame({
    "TruePrice":          true_price_aligned,
    "RNN_Delta_Price":    rnn_delta_price,
    "RNN_Direct_Price":   rnn_direct_price,
    "Naive_Direct_Price": naive_direct_price,
}, index=dates_aligned)

print("\n=== Predicted Prices (head) ===")
print(price_df.head(10).to_string())

# 視覺化
if PLOT:
    plt.figure(figsize=(13,6))
    plt.plot(price_df.index, price_df["TruePrice"],        label="True Price", linewidth=1.5)
    plt.plot(price_df.index, price_df["RNN_Delta_Price"],  label="RNN-Delta → Price", linewidth=1.2)
    plt.plot(price_df.index, price_df["RNN_Direct_Price"], label="RNN-Price (direct)", alpha=0.9)
    plt.plot(price_df.index, price_df["Naive_Direct_Price"], label="Naive (persist)", linestyle="--", alpha=0.8)
    plt.title(f"{symbol} | Test Price Prediction Comparison (ΔP vs Direct)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
