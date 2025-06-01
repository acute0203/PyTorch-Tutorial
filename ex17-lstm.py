import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import platform
# 裝置設定
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
# 股票代碼與下載區間
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')
close_prices = data['Close'].values.reshape(-1, 1)

# 正規化價格資料
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

# 建立時間序列資料
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

seq_length = 30
X, Y = create_sequences(scaled, seq_length)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# 訓練測試分割
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# LSTM 模型
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = StockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# 訓練
train_losses, test_losses = [], []
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_loss = criterion(test_out, Y_test).item()
        test_losses.append(test_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}")

# 可視化預測
model.eval()
with torch.no_grad():
    predicted = model(X_test).cpu().numpy()
    actual = Y_test.cpu().numpy()

predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(actual)

# 繪圖
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(actual, label="True Price")
plt.plot(predicted, label="Predicted")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title(f"{symbol} LSTM Prediction")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()