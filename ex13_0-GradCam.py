import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 1. 資料前處理與 CIFAR-10 載入 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# === 2. 修改 ResNet18 用於 CIFAR-10 ===
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
'''
# === 3. 簡單訓練10輪 ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("開始訓練...")
model.train()
for epoch in range(10):
    print(epoch)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("訓練完成。")
torch.save(model.state_dict(), "resnet18_weights.pth")
'''
model.load_state_dict(torch.load("resnet18_weights.pth", map_location=device))
model.eval()  # 設定為評估模式
# === 4. Grad-CAM 實作 ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_grad = self.gradients.mean(dim=[0, 2, 3])
        weighted_activations = self.activations[0] * pooled_grad[:, None, None]
        cam = weighted_activations.sum(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam

# === 5. 可視化 Grad-CAM 與原圖比較 ===
def visualize_cam(img_tensor, mask, save_path=None):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
    img = (img - img.min()) / (img.max() - img.min())  # normalize to 0-1

    # === 修正：把 mask resize 到與 img 同樣大小 ===
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # (W, H)

    # === heatmap 視覺化處理 ===
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0  # BGR -> RGB, normalize to 0~1

    # === 疊圖 ===
    cam = heatmap * 0.4 + img * 0.6
    cam = np.clip(cam, 0, 1)

    # === 顯示圖像 ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(cam)
    axs[1].set_title("Grad-CAM")
    axs[1].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

# === 6. 測試一張圖片 ===
model.eval()
test_iter = iter(testloader)
img, label = next(test_iter)
img = img.to(device)

cam_generator = GradCAM(model, model.layer4)
mask = cam_generator.generate_cam(img)

print(f"真實標籤: {label.item()}")
visualize_cam(img, mask)


# === 💡 練習題目 ===
# 1. 修改模型為 resnet34、resnet50，比較 Grad-CAM 效果
# 2. 使用不同類別圖片，觀察熱區變化
# 3. 改用自訂資料集進行熱區分析