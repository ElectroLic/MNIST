//纯数学方案（梯度下降算法+反向传播+一层隐藏层），不采用任何现有框架
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Dataloader():
    def get_data(self):
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/database/mnist.npz") as f:
            images, labels = f['x_train'], f["y_train"]
        images = images.astype("float32") / 255.0
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]
        return images, labels

if __name__ == "__main__":
    #通过Dataloader类获取数据
    dataloader = Dataloader()
    images, labels = dataloader.get_data()

    #创建模型
    #权重数组
    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    b_i_h = np.zeros((20, 1))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_h_o = np.zeros((10, 1))

    #设置超参数
    learning_rate = 0.01
    epochs = 30
    correct_count = 0

    #训练模型
    for epoch in tqdm(range(epochs), desc = "Training", unit = "epoch"):
        for img, lb in zip(images, labels):
            #转置
            img.shape += (1,)
            lb.shape += (1,)

            #前向传播
            # -输入层 -> 隐藏层
            h_pre = b_i_h + w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))
            # -隐藏层 -> 输出层
            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))
            # 计算损失
            loss = 1 / len(o) * np.sum((o - lb) ** 2, axis = 0)
            correct_count += int(np.argmax(o) == np.argmax(lb))

            #反向传播
            # -输出层 -> 隐藏层
            delta_o = 0.2 * (o - lb)
            delta_z = (o * (1 - o))
            delta_w_h = np.transpose(h)
            w_h_o += -learning_rate * delta_o @ delta_w_h * delta_z
            b_h_o += -learning_rate * delta_o
            # -隐藏层 -> 输入
            delta_h = np.transpose(w_h_o)
            delta_z_2 = (h * (1 - h))
            delta_w_i = np.transpose(img)
            w_i_h += -learning_rate * delta_h @ delta_o * delta_z_2 @ delta_w_i
            b_i_h += -learning_rate * delta_h @ delta_o * delta_z_2

        # 输出精准度
        temp = round((correct_count / images.shape[0]) * 100, 2)
        correct_count = 0
    print(f"  Accuracy: {temp}%")
    print(w_i_h)


    # 展示效果
    T = 20
    while T:
        T -= 1
        index = int(input("输入编号进行预测 (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        # 前向传播
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        plt.title(f"This figure is predicted to be: {o.argmax()} :")
        plt.show()
