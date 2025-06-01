import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
# 超參數
input_size = 1
hidden_size = 32
output_size = 1
seq_length = 50
num_epochs = 100
lr = 0.01

# 產生 sine wave 時間序列資料
def generate_data(seq_length, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    data = np.sin(x)
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y)

X, Y = generate_data(seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, seq, 1]
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

# 訓練 / 測試分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 定義 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最後一個 time step 輸出
        out = self.fc(out)
        return out

model = SimpleRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# 訓練
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 測試
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, Y_test).item()
        test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

# 預測可視化
model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(Y_test.cpu().numpy(), label="True")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Sine Wave Prediction")
plt.tight_layout()
plt.show()