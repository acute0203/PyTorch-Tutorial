import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1. 載入資料集與前處理
# 載入 Iris 資料集
iris = load_iris()
X = iris.data      # 特徵: shape = (150, 4)
y = iris.target    # 標籤: 3 類別 (0, 1, 2)

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化（對每個特徵 zero-mean, unit-variance）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 轉換為 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)     # 注意分類要用 long
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
#2. 建立模型（用 nn.Sequential）
model = nn.Sequential(
    nn.Linear(4, 10),  # 4 維輸入 → 10 個神經元
    nn.ReLU(),
    nn.Linear(10, 3)   # 10 → 3 類別輸出
)
#3. 定義損失函數與優化器（使用 SGD）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 100
# 4. 訓練模型
for epoch in range(num_epochs):
    # Forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# 5. 測試準確率
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
