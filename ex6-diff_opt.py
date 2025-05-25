import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import platform

# ---------- 自動選擇 GPU / MPS / CPU ---------- #
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

# ---------- 模型定義 (class-based) ---------- #
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logits

# ---------- 資料處理 ---------- #
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
y_train_raw = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_raw = torch.tensor(y_test, dtype=torch.long).to(device)

# ---------- 可以修改：Loss 與 Optimizer ---------- #
loss_name = "cross_entropy"   # 選擇： "cross_entropy" 或 "mse"
optimizer_name = "adam"       # 選擇： "sgd", "adam", "rmsprop"

# ---------- 損失函數設定 ---------- #
if loss_name == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
    y_train = y_train_raw
    y_test = y_test_raw
elif loss_name == "mse":
    criterion = nn.MSELoss()
    y_train = F.one_hot(y_train_raw, num_classes=3).float()
    y_test = F.one_hot(y_test_raw, num_classes=3).float()
else:
    raise ValueError("Unsupported loss type")

# ---------- 初始化模型 ---------- #
model = IrisNet().to(device)

# ---------- Optimizer 選擇 ---------- #
if optimizer_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=0.1)
elif optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=0.01)
elif optimizer_name == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
elif optimizer_name == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)
else:
    raise ValueError("Unsupported optimizer")

# ---------- 訓練 ---------- #
num_epochs = 100
loss_list = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"[{epoch+1:3d}/{num_epochs}] Loss: {loss.item():.4f}")

# ---------- 測試準確率 ---------- #
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)

    if loss_name == "mse":
        y_true = torch.argmax(y_test, dim=1)
    else:
        y_true = y_test

    acc = (predicted == y_true).float().mean()
    print(f"Test Accuracy: {acc:.4f}")

# ---------- 畫出訓練損失曲線 ---------- #
plt.plot(loss_list, label=f'Loss ({loss_name})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Loss Curve with {optimizer_name.upper()}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
