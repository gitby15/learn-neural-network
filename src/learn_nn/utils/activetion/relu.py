# relu激活函数：y = max(0, x)
# 优点：1）计算高效，求导简单
# 缺点：1）死区问题：神经元容易进入0区域，然后就一直不更新了（因为倒数是0）；

import numpy as np


class Relu:
    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        # x < 0 时，y = 0，所以y' = 0
        # x >= 0 时，y = x，所以y' = 1
        return (x > 0).astype(np.float32)
