import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from buildCNN import SimpleCNN

# 1. 模型加载函数（增强版）
def load_model(model_path, num_classes=11, device='cpu'):
    """加载训练好的模型（兼容PyTorch 2.6+的安全加载）"""
    model = SimpleCNN(num_classes=num_classes)
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
            
        # 方法1：关闭安全模式（需确保模型来源可信）
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 方法2：使用安全上下文（PyTorch 2.6+）
        # from torch.serialization import safe_globals
        # with safe_globals([torch.nn.modules.loss.CrossEntropyLoss]):
        #     checkpoint = torch.load(model_path, map_location=device)
        
        # 加载权重
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        model.to(device)
        print(f"✅ 模型载入成功: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print("尝试解决方案:")
        print("1. 确保模型文件未损坏")
        print("2. 使用 torch.save(model.state_dict(), ...) 重新保存模型")
        print("3. 如果信任模型来源，设置 weights_only=False")
        return None
    
# 2. 图像预处理函数
def get_transform():
    """返回与训练时一致的预处理流程"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])

# 3. 预测函数（增强版）
def predict_single_image(model, image, transform=None, device='cpu'):
    """
    增强版单图像预测函数
    支持输入类型: PIL.Image | np.ndarray | torch.Tensor
    返回: (预测类别, 置信度, 所有类别概率)
    """
    try:
        # 输入类型转换
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')
        elif isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
            image = ((image * 0.3081 + 0.1307) * 255).astype(np.uint8)  # 反标准化
            image = Image.fromarray(image, mode='L')
        elif not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL.Image、np.ndarray或torch.Tensor")
        
        # 确保图像尺寸
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.LANCZOS)
        
        # 应用预处理
        if transform is None:
            transform = get_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item(), probabilities.cpu().squeeze()
        
    except Exception as e:
        print(f"⚠️ 预测出错: {str(e)}")
        return -1, 0.0, None

# 4. 测试图像生成器（增强版）
def create_test_images():
    """生成更全面的测试图像集"""
    test_images = []
    rng = np.random.RandomState(42)  # 固定随机种子
    
    # 1. 纯空白
    blank = np.zeros((28, 28), dtype=np.uint8)
    test_images.append(("纯空白", blank))
    
    # 2. 低噪声空白
    noisy_blank = rng.uniform(0, 30, (28, 28)).astype(np.uint8)
    test_images.append(("低噪声空白", noisy_blank))
    
    # 3. 高噪声空白
    high_noise = rng.uniform(0, 100, (28, 28)).astype(np.uint8)
    test_images.append(("高噪声空白", high_noise))
    
    # 4-8. 简单数字 (1, 3, 5, 7, 9)
    digits = {
        "数字0": lambda x, y: ((x == 7) | (x == 21)) & (y >= 5) & (y <= 23) | 
                         ((y == 5) | (y == 23)) & (x >= 7) & (x <= 21),
        "数字1": lambda x, y: (x == 14) & (y >= 5) & (y <= 23),
        "数字2": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 14) & (x == 21) & (y >= 5) | (y >= 14) & (x == 7) & (y <= 23),
        "数字3": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 5),
        "数字4": lambda x, y: (y == 14) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 5) | (y >= 5) & (x == 7) & (y <= 14),
        "数字5": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 14) | (y >= 5) & (x == 7) & (y <= 14),
        "数字6": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 14) | (y >= 5) & (x == 7) & (y <= 23),
        "数字7": lambda x, y: (y == 5) & (x >= 7) & (x <= 21) | 
                          (y <= 23) & (x == 21) & (y >= 5),
        "数字8": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 5) | (y >= 5) & (x == 7) & (y <= 23),
        "数字9": lambda x, y: ((y == 5) | (y == 14) | (y == 23)) & (x >= 7) & (x <= 21) |
                         (y <= 23) & (x == 21) & (y >= 5) | (y >= 5) & (x == 7) & (y <= 14)
    }
    
    
    xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    for name, func in digits.items():
        digit = np.where(func(xx, yy), 255, 0).astype(np.uint8)
        test_images.append((name, digit))
    
    # 9. 部分填充
    partial = np.zeros((28, 28), dtype=np.uint8)
    partial[10:18, 10:18] = 128
    test_images.append(("部分填充", partial))
    
    # 10. 边缘案例
    edge_case = np.zeros((28, 28), dtype=np.uint8)
    edge_case[:, :4] = 200  # 左侧边缘
    test_images.append(("边缘填充", edge_case))
    
    return test_images

# 5. 可视化函数（增强版）
def visualize_predictions(model, test_images, transform, device, save_dir="predictions"):
    """增强版可视化，包含概率分布和混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    class_names = [str(i) for i in range(10)] + ['空白']
    
    def get_true_class(name):
        name = str(name).lower()
        if "空白" in name or "blank" in name:
            return 10
        digits = ''.join(c for c in name if c.isdigit())
        return int(digits) if digits else 10
    
    # 收集预测结果
    y_true = []
    y_pred = []
    all_probs = []
    
    plt.figure(figsize=(18, 12))
    
    for idx, (name, image) in enumerate(test_images):
        pred_class, confidence, probs = predict_single_image(
            model, image, transform, device
        )
        true_class = get_true_class(name)
        
        y_true.append(true_class)
        y_pred.append(pred_class)
        all_probs.append(probs.numpy() if probs is not None else np.zeros(11))

        # 在matplotlib绘图前添加中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 绘制子图
        plt.subplot(5, 4, idx + 1)
        plt.imshow(image, cmap='gray')
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f'{name}\n预测: {class_names[pred_class]}\n置信度: {confidence:.2f}', 
                 color=color)
        plt.axis('off')
        
        # 保存错误样本
        if pred_class != true_class:
            Image.fromarray(image).save(f"{save_dir}/wrong_{name.replace(' ', '_')}.png")
    
    # 绘制混淆矩阵
    plt.subplot(5, 4, 20)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_summary.png")
    plt.show()
    
    # 打印详细报告
    print("\n📊 预测结果统计:")
    for i, (name, _) in enumerate(test_images):
        print(f"{name:>12}: 真实={class_names[y_true[i]]}, 预测={class_names[y_pred[i]]}, 置信度={all_probs[i][y_pred[i]]:.2f}")

# 6. 主函数
def main():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型（修改为你的模型路径）
    model_path = "Models/mnist_cnn_v1.pth"  
    model = load_model(model_path, num_classes=11, device=device)
    if model is None:
        return
    
    # 获取预处理流程
    transform = get_transform()
    
    # 生成测试图像
    test_images = create_test_images()
    
    # 可视化预测结果
    visualize_predictions(model, test_images, transform, device)
    
    # 模型验证测试
    test_model_outputs(model, device)

# 7. 模型验证函数
def test_model_outputs(model, device):
    """验证模型输入输出是否符合预期"""
    print("\n🔍 正在验证模型结构...")
    
    # 测试标准输入
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
            assert output.shape == (1, 11), "输出形状应为(1,11)"
            probs = torch.softmax(output, dim=1)
            assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0)), "概率和应为1"
        print("✅ 模型验证通过")
        return True
    except Exception as e:
        print(f"❌ 模型验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    main()
    print("\n✨ 测试完成！检查predictions/目录下的结果")