import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
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
# 參數設定
batch_size = 64
num_epochs = 5
learning_rate = 0.001

# 資料預處理
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet 需要 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 簡化版本
])

# 載入 CIFAR-10 資料集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 使用預訓練的 ResNet18
model = models.resnet18(pretrained=True)

# 凍結前面的參數（可選）
for param in model.parameters():
    # Q:把 requires_grad=False 拿掉，只凍結 conv1~conv4
    param.requires_grad = False

# 替換最後一層為 CIFAR-10 分類（10 類）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# 訓練
train_loss_list = []
train_acc_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# 測試
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")

# 畫圖
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="Train Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()