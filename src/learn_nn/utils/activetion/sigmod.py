# sigmoid 函数: f(x) = 1/(1+e^(-x))
# 优点：1）输出范围在(0,1)之间，这使得它非常适合用于二分类问题。2）函数是单调递增的，这使得它在反向传播中很方便计算梯度。
# 缺点：1）输出不是0均值的，这在某些情况下可能会影响模型的性能。2）当输入非常大或非常小的时候，函数的梯度会非常小，这会导致训练过程中的梯度消失问题。
# 适用场景：1）二分类问题。2）需要将输入映射到(0,1)之间的问题。
import numpy as np


class Sigmod:
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmod.activation(x) * (1 - Sigmod.activation(x))
