import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import platform

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

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, output_size=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 可以改的區域
loss_name = "cross_entropy"         # "cross_entropy" 或 "mse"
optimizer_name = "adam"             # "sgd", "adam", "rmsprop", "adagrad"
feed_mode = "dataloader"         # "full", "manual_batch", "dataloader", "single_sample"
batch_size = 32

# 資料處理
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train_raw = torch.tensor(y_train, dtype=torch.long)
y_test_raw = torch.tensor(y_test, dtype=torch.long)

if loss_name == "mse":
    criterion = nn.MSELoss()
    y_train = F.one_hot(y_train_raw, num_classes=3).float()
    y_test = F.one_hot(y_test_raw, num_classes=3).float()
else:
    criterion = nn.CrossEntropyLoss()
    y_train = y_train_raw
    y_test = y_test_raw

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

model = IrisNet().to(device)

if optimizer_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=0.01)
elif optimizer_name == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
elif optimizer_name == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)
else:
    raise ValueError("Unsupported optimizer")

# 訓練
num_epochs = 100
loss_list = []

for epoch in range(num_epochs):
    model.train()

    if feed_mode == "full":
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elif feed_mode == "manual_batch":
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elif feed_mode == "dataloader":
        dataset = IrisDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elif feed_mode == "single_sample":
        for i in range(len(X_train)):
            x_i = X_train[i].unsqueeze(0)
            y_i = y_train[i].unsqueeze(0)
            outputs = model(x_i)
            loss = criterion(outputs, y_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        raise ValueError("Unsupported feed mode")

    loss_list.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"[{epoch+1:3d}/{num_epochs}] Loss: {loss.item():.4f}")

# 測試
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

# 畫圖
plt.plot(loss_list, label=f'Loss ({loss_name})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Loss Curve | {optimizer_name.upper()} | {feed_mode}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()