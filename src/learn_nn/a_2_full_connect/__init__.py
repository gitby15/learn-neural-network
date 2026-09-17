import os

import numpy as np
from tqdm import tqdm

from learn_nn.a_2_full_connect.dataset import get_test_dataset, get_train_dataset
from learn_nn.utils.activetion.relu import Relu
from learn_nn.utils.activetion.softmax import Softmax


# 神经网络定义
# 为了简化问题，我们不实现根据数据集自动调整网络结构的功能
# 输入固定位MNIST，输入28*28 + 一个颜色通道，输出固定是0-9这10个类别
input_size = 28 * 28  # 输入层节点数
hidden_size = 128  # 隐藏层节点数，可以调整
output_size = 10  # 输出层节点数

# 超参数
learning_rate = 0.1  # 学习率，可以调整
num_epochs = 5  # 训练轮数，可以调整
batch_size = 32  # mini-batch的批处理大小，可以调整
momentum = 0.9  # 动量，可以调整


# 创建one hot标签, 把image和label关联起来
def one_hot_encode(labels, num_classes=10):
    # 创建一个全零矩阵，行数为标签数量，列数为类别数量
    one_hot = np.zeros((labels.shape[0], num_classes))
    # 将对应类别的位置设为1
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


# 我们设计一个三层的神经网络，输入层-隐藏层-输出层

# 初始化权重和偏置，使用了He初始化
np.random.seed(42)  # 为了结果可复现，设置随机种子
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))  # 隐藏层偏置
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))  # 输出层偏置
velocity_W1 = np.zeros_like(W1)
velocity_b1 = np.zeros_like(b1)
velocity_W2 = np.zeros_like(W2)
velocity_b2 = np.zeros_like(b2)


# 损失函数用交叉熵损失
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]

    # 避免log(0)的情况
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 选出每一个样本，对应真实类别的预测概率
    loss = -np.log(y_pred_clip[range(m), np.argmax(y_true, axis=1)])

    # 求平均值
    loss = np.sum(loss) / m
    return loss


# 向前传播
def forward(x):
    z1 = np.dot(x, W1) + b1
    a1 = Relu.activation(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = Softmax.activation(z2)
    return z1, a1, z2, a2


# 向后传播
def backward(x, y, z1, a1, z2, a2):
    m = y.shape[0]

    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * Relu.derivative(z1)
    dW1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def update_parameters(dW1, db1, dW2, db2):
    global W1, b1, W2, b2, velocity_W1, velocity_b1, velocity_W2, velocity_b2
    # 使用动量法更新参数
    velocity_W1 = momentum * velocity_W1 + learning_rate * dW1
    velocity_b1 = momentum * velocity_b1 + learning_rate * db1
    velocity_W2 = momentum * velocity_W2 + learning_rate * dW2
    velocity_b2 = momentum * velocity_b2 + learning_rate * db2
    W1 -= velocity_W1
    b1 -= velocity_b1
    W2 -= velocity_W2
    b2 -= velocity_b2


# 训练模型

count = 0


def train_model():


    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        train_datasets = get_train_dataset(batch_size)
        batch_progress = tqdm(
            enumerate(train_datasets),
            desc=f"Batch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for count, (images, labels) in batch_progress:
            label_one_hot = one_hot_encode(labels)
            # 向前传播
            z1, a1, z2, a2 = forward(images)

            # 计算损失
            loss = compute_loss(label_one_hot, a2)

            # 向后传播
            dW1, db1, dW2, db2 = backward(images, label_one_hot, z1, a1, z2, a2)

            # 更新参数
            update_parameters(dW1, db1, dW2, db2)
            batch_progress.set_postfix(loss=f"{loss:.4f}")

    # 在测试集上评估模型
    test_datasets = get_test_dataset(batch_size)
    test_loss = 0
    test_accuracy = 0
    test_count = 0
    for images, labels in test_datasets:
        label_one_hot = one_hot_encode(labels)
        z1, a1, z2, a2 = forward(images)
        test_loss += compute_loss(label_one_hot, a2)
        test_accuracy += np.mean(np.argmax(a2, axis=1) == labels)
        test_count += 1
    test_loss /= test_count
    test_accuracy /= test_count
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def predict(image, label):
    # image是一个一维数组，长度是28*28=784，值是0-255
    # 需要把它转换成和训练时一样的格式
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, -1)  # 转换成二维数组，行数是1，列数是784

    # 向前传播
    _, _, _, a2 = forward(image)

    # 取最大值的索引作为预测结果
    prediction = np.argmax(a2, axis=1)[0]
    return prediction, label


if __name__ == "__main__":
    train_model()
    test_datasets = get_test_dataset(1)
    for image, label in test_datasets:
        prediction, label = predict(image, label)
        print(f"Prediction: {prediction}, Label: {label}")


# 训练效果：1000个样本，测试5轮，准确率能干到90%

# 跟上一章节的区别：
# 1. 增加了mini-batch的支持
# 2. 增加了He初始化
# 3. 增加了动量法
# 4. numpy不支持GPU加速，这两个章节大概学会原理后，后面的神经网络都用torch来实现了