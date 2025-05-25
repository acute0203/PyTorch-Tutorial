import torch
import torch.nn as nn

# 使用 nn.Sequential 建立模型
model = nn.Sequential(
    nn.Linear(10, 20),   # 輸入層 → 隱藏層
    nn.ReLU(),           # 啟用函數
    nn.Linear(20, 3)     # 隱藏層 → 輸出層
)
# init each layer's weight & bias
nn.init.normal_(model[0].weight)
nn.init.normal_(model[0].bias)
nn.init.normal_(model[2].weight)
nn.init.normal_(model[2].bias)
# 如果你希望每一層有名字，可以這樣寫：
'''
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(20, 3))
]))

# 初始化函數：這邊以 Xavier 初始化為例
def init_weights(m):
    # 確保只對線性層初始化，避免對 ReLU、BatchNorm 等不需要初始化的層下手
    if isinstance(m, nn.Linear):
        # 適用於激活函數為 ReLU 或 Sigmoid 的情況
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 套用初始化
# model.apply(fn)：會自動遞迴呼叫模型中的每一層（包含子模組
model.apply(init_weights)
'''

# 建立輸入資料 (batch_size = 5, feature = 10)
x = torch.randn(5, 10)

# 前向傳遞
output = model(x)
print("模型輸出：", output)
