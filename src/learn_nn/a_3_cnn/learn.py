# 卷积神经网络是一种用于处理图像数据的方案n
# 卷积层
# 池化层
# 全连接层
import numpy as np

from learn_nn.a_3_cnn.dataset import get_single_image, get_train_dataset
from learn_nn.utils.activetion.relu import Relu
from learn_nn.utils.activetion.softmax import Softmax

# 超参数
learning_rate = 0.01
num_epochs = 5
momentum = 0.9
mini_batch_size = 64

# 初始化神经网络的参数
np.random.seed(42)  # 固定好随机种子，这样每次运行结果都一样，方便调试

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

class FC:
    def __init__(self, input_size, hidden_size, output_size):
        # 全连接层参数
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))  # 隐藏层偏置
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))  # 输出层偏置
        self.velocity_W1 = np.zeros_like(self.W1)
        self.velocity_b1 = np.zeros_like(self.b1)
        self.velocity_W2 = np.zeros_like(self.W2)
        self.velocity_b2 = np.zeros_like(self.b2)

    # 损失函数用交叉熵损失
    def compute_loss(self, y_pred, y_truth):
        loss = compute_loss(y_pred, y_truth)
        return loss

    def forward(self, x):
        # 全连接层前向传播
        self.z1 = np.dot(x, self.W1) + self.b1  # 隐藏层线性变换
        self.a1 = Relu.activation(self.z1)  # ReLU激活函数
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 输出层线性变换
        self.a2 = Softmax.activation(self.z2)  # Softmax激活函数
        return self.a2

    def backward(self, x, y_true, y_pred):
        m = y_true.shape[0]

        # 输出层梯度
        dz2 = y_pred - y_true  # 交叉熵损失对z2的梯度
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * Relu.derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新参数，使用SGD + Momentum
        self.velocity_W1 = momentum * self.velocity_W1 + learning_rate * dW1
        self.velocity_b1 = momentum * self.velocity_b1 + learning_rate * db1
        self.velocity_W2 = momentum * self.velocity_W2 + learning_rate * dW2
        self.velocity_b2 = momentum * self.velocity_b2 + learning_rate * db2
        self.W1 -= self.velocity_W1
        self.b1 -= self.velocity_b1
        self.W2 -= self.velocity_W2
        self.b2 -= self.velocity_b2


class CNN:
    def __init__(self):
        # 卷积层参数
        self.num_kernels = 3  # 卷积核数量
        self.kernel_size = 3  # 卷积核大小
        self.stride = 1  # 步长
        self.padding = 1  # 填充
        self.kernels = (
            np.random.randn(self.num_kernels, self.kernel_size, self.kernel_size) * 0.1
        )
        self.biases = np.zeros((self.num_kernels, 1))  # 每个卷积核一个偏置
        # 池化层参数
        self.pool_size = 2  # 池化窗口大小
        self.pool_stride = 2  # 池化步长

    def _conv2d(self, x, kernel, bias):
        # x的形状是(N, H, W)，颜色通道是1
        # kernel的形状是(F, HH, WW) F是卷积核数量
        # bias的形状是(F, 1)
        N, H, W = x.shape
        F, HH, WW = kernel.shape
        # 计算输出尺寸
        # a//b 表示向下取整
        out_H = (H + 2 * self.padding - HH) // self.stride + 1
        out_W = (W + 2 * self.padding - WW) // self.stride + 1
        out = np.zeros((N, F, out_H, out_W))
        # 填充输入
        # 往每张图的上下左右各插入padding行/列，插入的数值是0
        x_padded = np.pad(
            x,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )
        # 卷积操作
        # count = 0
        for n in range(N):  # 遍历每个样本
            for f in range(F):  # 遍历每个卷积核
                for i in range(out_H):  # H和W 是遍历每一个像素点
                    for j in range(out_W):
                        h_start = i * self.stride
                        h_end = h_start + HH
                        w_start = j * self.stride
                        w_end = w_start + WW
                        out[n, f, i, j] = (
                            x_padded[n, h_start:h_end, w_start:w_end] * kernel[f]
                        ).sum() + bias[f].item()
                # count += 1
                # print("count: {count}, n: {n}, f: {f}".format(count=count, n=n, f=f))
        return out  # (N, F, out_H, out_W)，表示N张图，每张图经过F个卷积核，变成F个特征图，每个特征图的尺寸是(out_H, out_W)

    def _max_pool2d(self, x):
        # x的形状是(N, F, H, W)，颜色通道是1
        N, F, H, W = x.shape
        pool_H = self.pool_size
        pool_W = self.pool_size
        stride = self.pool_stride
        # 计算输出尺寸
        out_H = (H - pool_H) // stride + 1
        out_W = (W - pool_W) // stride + 1
        out = np.zeros((N, F, out_H, out_W))
        for n in range(N):  # 遍历每个样本
            for f in range(F):  # 遍历每个卷积核
                for i in range(out_H):  # H和W 是遍历每一个像素点
                    for j in range(out_W):
                        h_start = i * stride
                        h_end = h_start + pool_H
                        w_start = j * stride
                        w_end = w_start + pool_W
                        out[n, f, i, j] = np.max(x[n, f, h_start:h_end, w_start:w_end])
        return out  # (N, F, out_H, out_W) N*F张图，每张图经过池化，变成一张特征图，尺寸是(out_H, out_W)

    def forward(self, x):
        # x的形状是(N, H, W)，颜色通道是1
        self.conv_out = self._conv2d(x, self.kernels, self.biases)  # (N, F, H, W)
        self.relu_out = Relu.activation(self.conv_out)  # ReLU激活函数, (N, F, H, W)
        # 对每个卷积核的输出，进行池化操作
        self.pool_out = self._max_pool2d(self.relu_out)  # (N, F, HP, WP)
        return self.pool_out

    def backward(self, x, dout):
        # 反向传播暂时不实现
        pass


def train_model():
    cnn = CNN()
    # 拿一张图算出pool_out的尺寸，用于计算全连接层的输入尺寸
    
    image, label = get_single_image()
    cnn.forward(image)

    pool_out = cnn.pool_out  # (10, 3, 14, 14)

    input_size = cnn.kernel_size * pool_out.shape[2] * pool_out.shape[3]
    hidden_size = 64
    output_size = 10

    fc = FC(input_size, hidden_size, output_size)


    # 训练
    for epoch in range(num_epochs):
        for image_batch, label_batch in get_train_dataset(mini_batch_size):
            # 前向传播
            cnn_out = cnn.forward(image_batch)  # (mini_batch_size, F, HP, WP)
            fc_input = cnn_out.reshape(mini_batch_size, -1)  # 展平 (F*HP*WP)
            y_pred = fc.forward(fc_input)  # (mini_batch_size, 10)
            loss = fc.compute_loss(y_pred, label_batch)
            # 反向传播
            fc.backward(fc_input, label_batch, y_pred)
            # cnn_for_calculate.backward(x_batch, dout)  # 卷积层的反向传播暂时不实现
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")




    # # 测试
    # cnn.forward(test_images[0:10]) # 先随便拿10张图测试
    # pool_out = cnn.pool_out  # (10, 3, 14, 14)
    # fc_input = pool_out.reshape(10, -1)  # 展平 (10, F*HP*WP)
    # y_pred = fc.forward(fc_input)  # (10, 10)
    # y_pred_labels = np.argmax(y_pred, axis=1)
    # print("Predicted labels:", y_pred_labels)
    # print("True labels:", np.argmax(test_labels[0:10], axis=1))
    # return
    
    
if __name__ == "__main__":
    train_model()
