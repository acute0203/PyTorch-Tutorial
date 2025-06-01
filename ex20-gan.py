import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os,platform

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

# === 1. 資料前處理與載入 ===

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 將像素值正規化至 [-1, 1]
])

# 下載 MNIST 資料集
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# === 2. 定義 Generator 模型 ===

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # 輸出值在 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# === 3. 定義 Discriminator 模型 ===

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 判斷為真或假
        )

    def forward(self, x):
        return self.net(x)

# === 4. 初始化模型與訓練參數 ===

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

noise_dim = 100
epochs = 50
sample_dir = "./gan_samples"
os.makedirs(sample_dir, exist_ok=True)

# === 5. 訓練迴圈 ===

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)

        # 將圖片展平成一維向量
        real_imgs = real_imgs.view(batch_size, -1).to(device)

        # 真實與假資料的標籤
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # === 訓練 Discriminator ===
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = G(z)

        real_loss = loss_fn(D(real_imgs), real_labels)
        fake_loss = loss_fn(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # === 訓練 Generator ===
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = G(z)
        g_loss = loss_fn(D(fake_imgs), real_labels)  # 希望 D 判斷為真

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    # 每 10 回儲存一張生成圖
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, noise_dim).to(device)
            samples = G(z).view(-1, 1, 28, 28).cpu()
            grid = torch.cat([s for s in samples[:16]], dim=2).squeeze()
            plt.imshow(grid.numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Epoch {epoch+1}')
            plt.savefig(f"{sample_dir}/sample_epoch_{epoch+1}.png")
            plt.close()