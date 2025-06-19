import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset
#from buildCNN import SimpleCNN
from PIL import Image

if __name__ == '__main__':
    # 所有训练代码放在这里

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=11):  # 0-9 數字 + 空白類
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)  # 11 類輸出
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # 自定義空白資料集類別
    class BlankDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=10000, transform=None):
            self.num_samples = num_samples
            self.transform = transform
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = np.zeros((28, 28), dtype=np.float32)
            
            if np.random.random() < 0.3:
                noise_type = np.random.choice(["gaussian", "uniform", "line", "dot"])
                if noise_type == "gaussian":
                    image += np.random.normal(0, 0.05, (28, 28))
                elif noise_type == "uniform":
                    image += np.random.uniform(0, 0.1, (28, 28))
                elif noise_type == "line":
                    for _ in range(1):
                        x = np.random.randint(28)
                        image[x] = np.random.rand(28) * 0.2
                elif noise_type == "dot":
                    for _ in range(random.randint(5, 15)):
                        x, y = random.randint(0, 27), random.randint(0, 27)
                        image[x, y] = random.random() * 0.5
                
                image = np.clip(image, 0, 1)
            
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')
            if self.transform:
                image = self.transform(image)
            return image, 10  # 空白类标签为10

    # 自定義資料集包裝器，用於重新標記 MNIST
    class MNISTWithBlank(torch.utils.data.Dataset):
        def __init__(self, mnist_dataset):
            self.mnist_dataset = mnist_dataset
            
        def __len__(self):
            return len(self.mnist_dataset)
        
        def __getitem__(self, idx):
            image, label = self.mnist_dataset[idx]
            # MNIST 標籤保持 0-9，空白類將是 10
            return image, label

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    folder_path = "Models"
    base_filename = "mnist_cnn"
    version = 1

    # 确保Models目录存在
    os.makedirs(folder_path, exist_ok=True)

    # 查找最新版本号
    while os.path.exists(os.path.join(folder_path, f"{base_filename}_v{version}.pth")):
        version += 1

    print(f"📦 準備訓練模型版本: v{version}")

    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 載入 MNIST 資料集
    mnist_train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    mnist_test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # 包裝 MNIST 資料集
    train_mnist = MNISTWithBlank(mnist_train_dataset)
    test_mnist = MNISTWithBlank(mnist_test_dataset)

    blank_train = BlankDataset(num_samples=6000, transform=train_transform)  # 10% 的訓練資料為空白
    blank_test = BlankDataset(num_samples=1000, transform=test_transform)   # 10% 的測試資料為空白

    train_dataset = ConcatDataset([train_mnist, blank_train])
    test_dataset = ConcatDataset([test_mnist, blank_test])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"📊 訓練資料: {len(train_dataset)} 筆")
    print(f"📊 測試資料: {len(test_dataset)} 筆")

    # 載入你的 CNN 模型
    model = SimpleCNN(num_classes=11)

    # 尝试加载最新版本的模型he
    latest_version = version - 1
    if latest_version > 0:
        latest_model_path = os.path.join(folder_path, f"{base_filename}_v{latest_version}.pth")
        if os.path.isfile(latest_model_path):
            try:
                checkpoint = torch.load(latest_model_path)
                # 修复：兼容两种保存格式
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 新格式：包含完整检查点信息
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 已加载模型: {latest_model_path}")
                    print(f"   上次准确率: {checkpoint.get('accuracy', '未知')}%")
                else:
                    # 旧格式：只包含state_dict
                    model.load_state_dict(checkpoint)
                    print(f"✅ 已加载模型: {latest_model_path}")
                    print(f"   上次准确率: 未知")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")

    # 3. 使用 CPU 或 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用設備: {device}")
    model.to(device)

    # 4. 設定 loss function 與 optimizer
    import torch.optim as optim
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    def test(model, test_loader, device, save_wrong=False):  # 添加save_wrong参数
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 11
        class_total = [0] * 11
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 错误样本保存逻辑
                wrong_count = 0
                max_wrong_samples = 50
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    pred = predicted[i].item()
                    
                    if pred != label and save_wrong and wrong_count < max_wrong_samples:
                        img = images[i].cpu().numpy().squeeze()
                        img = ((img * 0.3081 + 0.1307) * 255).astype(np.uint8)
                        img = Image.fromarray(img, mode='L')
                        os.makedirs("wrong_preds", exist_ok=True)
                        img.save(f"wrong_preds/wrong_{wrong_count}_true{label}_pred{pred}.png")
                        wrong_count += 1
                    
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

        accuracy = 100 * correct / total
        print(f'测试准确率: {accuracy:.2f}%')
        
        # 显示各类别准确率
        class_names = [str(i) for i in range(10)] + ['空白']
        for i in range(11):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'   类别 {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        
        return accuracy

    print("\n📋 訓練前測試:")
    initial_accuracy = test(model, test_loader, device, save_wrong=True)

    # 训练参数
    num_epochs = 20
    best_accuracy = initial_accuracy
    patience = 3
    no_improve = 0

    # 5. 訓練模型
    for epoch in range(num_epochs):  # 訓練次數
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

            # 計算 loss
            loss = criterion(outputs, labels)

            # 反向傳播 + 更新權重
            loss.backward()
            optimizer.step()

            # 統計
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 每200個batch顯示進度
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # 更新学习率
        scheduler.step()

        train_accuracy = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        print(f'📈 Epoch {epoch+1}/{num_epochs} 完成 - Loss: {avg_loss:.4f}, 訓練準確率: {train_accuracy:.2f}%')
        
        # 每3个epoch测试一次
        if (epoch + 1) % 3 == 0:
            print(f"\n📋 Epoch {epoch+1} 测试:")
            current_accuracy = test(model, test_loader, device, save_wrong=(epoch == num_epochs-1))
            
            # 早停机制
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                no_improve = 0
                # 保存最佳模型
                best_model_path = os.path.join(folder_path, f"{base_filename}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"💾 保存最佳模型: {best_model_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️ 早停触发，已连续{no_improve}个epoch没有改进")
                    break
            print()

    new_model_path = os.path.join(folder_path, f"{base_filename}_v{version}.pth")

    torch.save(model.state_dict(), new_model_path)
    # 保存模型时添加检查点
    """
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'accuracy': best_accuracy
    }, new_model_path)
    """
    print(f"✅ 模型已保存为: {new_model_path}")

    # 最終測試
    print("\n📋 訓練後最終測試:")
    final_accuracy = test(model, test_loader, device, save_wrong=True)

    print(f"\n🎉 訓練完成！")
    print(f"📊 支援類別: 0-9 數字 + 空白類")
    print(f"🎯 最佳测试准确率: {best_accuracy:.2f}%")
    print(f"🏁 最终测试准确率: {final_accuracy:.2f}%")

    # 保存類別標籤對應
    label_map = {i: str(i) for i in range(10)}
    label_map[10] = 'blank'
    print(f"\n📝 類別對應: {label_map}")

    # 预测函数
    def predict_image(model, image_path, device):
        try:
            img = Image.open(image_path).convert('L').resize((28, 28))
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
            pred = output.argmax(1).item()
            return label_map.get(pred, "未知类别")
        except Exception as e:
            return f"预测失败: {str(e)}"

    # 示例预测
    test_image_path = "picture/sudoku.jpg"  # 替换为实际路径
    if os.path.exists(test_image_path):
        print(f"\n🔍 测试图像预测结果: {predict_image(model, test_image_path, device)}")
    else:
        print("\n⚠️ 未找到测试图像，请检查路径")