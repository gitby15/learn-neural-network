# Softmax: 是一种放大优势的归一化，将所有值映射到0-1，所有值相加，总和为1
# 优点：放大差异，让有优势的参数突出；处处可导，不像relu函数在0区域不可导
# 缺点：梯度消失；训练过程中容易造成个别神经元虹吸其他神经元

import numpy as np


class Softmax:
    @staticmethod
    def activation(x):
        # 分子：e^x
        # 分母：e^x的和
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def derivative(x):
        # 求导：y' = y * (1 - y)
        return Softmax.activation(x) * (1 - Softmax.activation(x))
    