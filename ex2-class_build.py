import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 全連接層1
        self.relu = nn.ReLU()                         # 啟用函數
        self.fc2 = nn.Linear(hidden_size, output_size)  # 全連接層2

    def forward(self, x):
        out = self.fc1(x)      # 輸入 → 隱藏層
        out = self.relu(out)   # ReLU
        out = self.fc2(out)    # 隱藏層 → 輸出
        return out

# 假設輸入維度是 10，隱藏層維度是 20，輸出維度是 3
model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)

# 建立一個假資料 (batch_size=5, input_size=10)
x = torch.randn(5, 10)

# 執行前向推論
output = model(x)
print("模型輸出：", output)
