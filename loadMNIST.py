import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 定義 SimpleCNN 模型（如果沒有 buildCNN.py）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 模型版本管理
folder_path = "Models"
base_filename = "mnist_cnn"
version = 1

# 確保Models目錄存在
os.makedirs(folder_path, exist_ok=True)

# 查找最新版本號
while os.path.exists(os.path.join(folder_path, f"{base_filename}_v{version}.pth")):
    version += 1

print(f"📦 準備訓練模型版本: v{version}")

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的標準化參數
])

# 載入 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# 資料載入器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"📊 訓練資料: {len(train_dataset)} 筆")
print(f"📊 測試資料: {len(test_dataset)} 筆")

# 初始化模型
model = SimpleCNN()

# 嘗試加載最新版本的模型
latest_version = version - 1
if latest_version > 0:
    latest_model_path = os.path.join(folder_path, f"{base_filename}_v{latest_version}.pth")
    if os.path.isfile(latest_model_path):
        try:
            model.load_state_dict(torch.load(latest_model_path, weights_only=True))
            print(f"✅ 已加載模型: {latest_model_path}")
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用設備: {device}")
model.to(device)

# 設定損失函數與優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 測試函數
def test_model(model, test_loader, device):
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
    
    accuracy = 100 * correct / total
    print(f'🎯 測試準確率: {accuracy:.2f}%')
    return accuracy

# 訓練前測試
print("\n📋 訓練前測試:")
test_model(model, test_loader, device)

# 訓練模型
print(f"\n🚀 開始訓練 {10} 個 epochs...")
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 統計
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # 每100個batch顯示進度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{10}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 每個epoch的統計
    train_accuracy = 100 * correct_train / total_train
    avg_loss = running_loss / len(train_loader)
    print(f'📈 Epoch {epoch+1}/{10} 完成 - Loss: {avg_loss:.4f}, 訓練準確率: {train_accuracy:.2f}%')

# 保存模型
new_model_path = os.path.join(folder_path, f"{base_filename}_v{version}.pth")
torch.save(model.state_dict(), new_model_path)
print(f"💾 模型已保存為: {new_model_path}")

# 最終測試
print("\n📋 訓練後最終測試:")
final_accuracy = test_model(model, test_loader, device)

print(f"\n🎉 訓練完成！最終測試準確率: {final_accuracy:.2f}%")