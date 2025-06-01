import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import platform

# 檢查是否有可用的 GPU，否則使用 CPU
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
print(f"Using device: {device}")

# === 1. 資料下載與預處理 ===

# 從 yfinance 抓取比特幣一年的歷史收盤價
data = yf.download("ETH-USD", start="2023-01-01", end="2024-01-01")['Close'].values
data = data.reshape(-1, 1)  # 將資料 reshape 為 2D

# 正規化資料到 [0, 1]
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 設定時間步長
sequence_length = 30

# 建立序列資料（時間步長長度的資料作為 X，下一天作為 y）
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, sequence_length)

# 轉為 PyTorch tensor，並移到 GPU（如有）
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 轉換維度為 (seq_len, batch_size, feature)
X = X.permute(1, 0, 2)

# === 2. 定義 Transformer 模型 ===

class TransformerPredictor(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(feature_size, 64)
        self.pos_encoder = nn.Sequential()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(64, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        out = output[-1, :, :]
        return self.decoder(out)

# === 3. 建立模型與訓練設定 ===

model = TransformerPredictor().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 4. 模型訓練 ===

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# === 5. 預測與可視化 ===

model.eval()
with torch.no_grad():
    predicted = model(X).cpu().detach().numpy()
    real = y.cpu().detach().numpy()

predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(real)

plt.figure(figsize=(12, 6))
plt.plot(real, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("BTC Price Prediction with Transformer")
plt.show()