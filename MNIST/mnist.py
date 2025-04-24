//使用卷积神经网络框架+pytorch

#导入relevant lībraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

#mnist.npz在database文件夹下，是mnist.py的下级文件
data = np.load('database/mnist.npz')
print(data.files)
#了解训练集和测试集长啥样
x_train = data['x_train']
print(x_train.shape)
y_train = data['y_train']
print(y_train.shape)
x_test = data['x_test']
print(x_test.shape)
y_test = data['y_test']
print(y_test.shape)

#step1: 定义数据预处理，这里自定义一个Dataset类，也可以直接使用torchvision.datasets.MNIST
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        # 转为 float tensor，并归一化（从 0~255 → 0~1）
        self.images = torch.tensor(images, dtype = torch.float32) / 255.0
        # 转为 long tensor（神经网络分类时需要 long 类型标签）
        self.labels = torch.tensor(labels, dtype = torch.long)
        # 增加通道维度：(N, 28, 28) → (N, 1, 28, 28)，符合 CNN 输入格式
        self.images = self.images.unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

#step2: 创建Dataset和DataLoader
train_dataset = MNISTDataset(x_train, y_train)
test_dataset = MNISTDataset(x_test, y_test)

# step3: 用 DataLoader 封装数据，方便训练时分批读取
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

#测试是否加载成功
images, labels = next(iter(train_loader))
print(images.shape)  # torch.Size([64, 1, 28, 28])
print(labels.shape)  # torch.Size([64])

# 初始化权重和偏置（手动实现）
input_size = 28 * 28
hidden_size = 128
output_size = 10

W1 = torch.randn(input_size, hidden_size, dtype=torch.float32, requires_grad=True)
b1 = torch.zeros(hidden_size, dtype=torch.float32, requires_grad=True)
W2 = torch.randn(hidden_size, output_size, dtype=torch.float32, requires_grad=True)
b2 = torch.zeros(output_size, dtype=torch.float32, requires_grad=True)

# 定义超参数
learning_rate = 0.1
epochs = 100
loss_fn = torch.nn.CrossEntropyLoss()

# 训练过程
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        # 展平图像：64 x 1 x 28 x 28 -> 64 x 784
        x = images.view(images.size(0), -1)  # batch_size x 784

        # 前向传播
        z1 = x @ W1 + b1              # batch_size x hidden_size
        a1 = torch.relu(z1)           # ReLU激活
        z2 = a1 @ W2 + b2             # batch_size x output_size (10类)

        # 损失计算
        loss = loss_fn(z2, labels)

        # 反向传播
        loss.backward()

        # 参数更新
        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            b1 -= learning_rate * b1.grad
            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad

            # 清空梯度
            W1.grad.zero_()
            b1.grad.zero_()
            W2.grad.zero_()
            b2.grad.zero_()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#评估模型
# 测试准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        x = images.view(images.size(0), -1)
        z1 = x @ W1 + b1
        a1 = torch.relu(z1)
        z2 = a1 @ W2 + b2
        preds = torch.argmax(z2, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")




