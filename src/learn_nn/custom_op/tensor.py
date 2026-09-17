# 简单学习torch是怎么处理张量的
import numpy as np
import torch

def iter_indices(shape):
    index = [0]*len(shape)
    while True:
        yield tuple(index)

        # 从最后一维开始进位
        dim = len(shape) - 1

        while dim >= 0:
            index[dim] += 1

            if index[dim] < shape[dim]:
                break

            index[dim] = 0
            dim -= 1

        if dim < 0:
            break

        

# 暂时只支持二维，后面再支持更高维度
class TimTensor:

    # 简单处理，暂时不考虑太边界的case
    def __init__(self, data: list[any], shape=None, stride=None):
        _list, _shape, _stride = self._flat_array(data)
        self.storage = _list
        
        if shape is None:
            self.shape = _shape
        else:
            self.shape = tuple(shape)
            
        if stride is None:
            self.stride = _stride
        else:
            self.stride = tuple(stride)
            
    # 输入多维数组，输出一维数组、shape、stride
    @staticmethod
    def _flat_array(data: list[any]) -> (list[any], list[int], list[int]):
        def _flatten(arr) -> (list[any], list[int]):
            if not isinstance(arr, list):
                return None
            if not isinstance(arr[0], list):
                return arr, [len(arr)]

            result = []
            _len = len(arr)
            len_list = [_len]
            for item in arr:
                r, s = _flatten(item)
                result.extend(r)
            print(f"=== len_list: {len_list}, r: {r}, s: {s}")
            # 简单处理，我们认为同一层的数组，长度都是一样的
            len_list.extend(s)
            
            return result, len_list
        _list, _shape = _flatten(data)
        _stride = []
        _temp = 1
        for item in reversed(_shape):
            _stride.insert(0, _temp)
            _temp *= item
            

        return _list, _shape, _stride

    # 根据坐标计算offset
    def offset(self, indices):
        offset = 0
        for i, s in zip(indices, self.stride):
            offset += i * s

        return offset

    # 暂时简单实现，不考虑越界情况
    def get(self, indices):
        if len(indices) != len(self.shape):
            raise ValueError("取数形状对不上")
        return self.storage[self.offset(indices)]


    # 只改变逻辑形状，底层不产生数据拷贝
    def transpose(self):
        new_shape = self.shape[::-1]
        new_strides = self.stride[::-1]
        return TimTensor(self.storage, shape=new_shape, stride=new_strides)

    # 判断逻辑存储和物理存储是否连续
    def is_contiguous(self):
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride)):
            if stride != expected:
                return False
            expected *= size
        return True

    @staticmethod
    def _compute_contiguous_stride(shape):
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
        return tuple(stride)

    # 只改变逻辑形状，底层不产生数据拷贝
    def view(self, *shape):
        if not self.is_contiguous():
            raise ValueError("view is not possible on non-contiguous tensor")
        old_numel = np.prod(self.shape)
        new_numel = np.prod(shape)
        if old_numel != new_numel:
            raise ValueError("view is not possible with different number of elements")
        
        new_stride = self._compute_contiguous_stride(shape)
        return TimTensor(self.storage, shape=shape, stride=new_stride)



    # 重新整理物理布局，跟现在的逻辑布局同步
    def contiguous(self):
        if self.is_contiguous():
            return self

        new_storage = np.empty(
            np.prod(self.shape),
            dtype=self.storage.dtype
        )

        write_pos = 0

        for idx in iter_indices(self.shape):
            old_offset = self.offset(idx)
            new_storage[write_pos] = (
                self.storage[old_offset]
            )

            write_pos += 1

        return TimTensor(
            new_storage,
            shape=self.shape
        )

    def __repr__(self):
        return (
            f"TimTensor("
            f"shape={self.shape}, "
            f"stride={self.stride}, "
            f"data=\n{self.storage}"
            f")"
        )


def check_my_tensor():
    raw_data = [[1, 2, 3], [4, 5, 6],[7,8,9]]
    np_data = np.array(raw_data)
    tim_tensor = TimTensor(raw_data)
    tensor = torch.tensor(np_data)

    print(f"shape: {tim_tensor.shape} == {tensor.shape}")
    print(f"stride: {tim_tensor.stride} == {tensor.stride}")
    print("== transpose ==")
    tim_tensor.transpose()
    tensor.transpose(-2, -1)
    print(f"shape: {tim_tensor.shape} == {tensor.shape}")
    print(f"stride: {tim_tensor.stride} == {tensor.stride}")




    

def main():
    list = [
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    ]
    (arr, shape, stride) = TimTensor._flat_array(list)
    print(arr)
    print(shape)
    print(stride)

if __name__ == "__main__":
    main()
