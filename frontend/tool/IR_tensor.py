import numpy as np
# H高度，w宽度，C通道数，M卷积核数量，也是输出的深度
# batch一次处理的数据个数，每次迭代中处理32张图像，那么输入张量的形状可能会是 [32, height, width, channels]，其中 32 就是batch大小
# reshape函数用于改变张量的形状，即在不改变张量数据内容的情况下，重新排列张量的维度。
# resize函数用于改变张量的形状，并且在必要时可以增减张量中的数据。
# https://www.bilibili.com/video/BV16N411y7cV/?spm_id_from=333.337.search-card.all.click&vd_source=81863602d1b31de5a4149c4065401902


class Format(object):  # 高维张量的展开模式
    NCHW = 0
    NHWC = 1


class DataType(object):  # 数据类型ID
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9
    FLOAT64 = 10
    COMPLEX128 = 11
    BFLOAT16 = 12
    BFLOAT32 = 13


class Shape:  # 专门用于表示张量形状的class
    N = None
    C = None
    H = None
    W = None

    def __init__(self, batch, channel, width, height):
        self.N = batch
        self.C = channel
        self.W = width
        self.H = height

    def get_shape(self):
        return self

    def __repr__(self):
        return f"{[self.N, self.C, self.W, self.H]}"

    def get_shape_as_np(self):
        np_shape = np.zeros(4).astype(int)
        np_shape[0] = self.N
        np_shape[1] = self.H
        np_shape[2] = self.W
        np_shape[3] = self.C
        return np_shape


class TensorType(object):
    Intermediate = 0
    Weight = 1
    Bias = 2
    Parameter = 3
    Input = 4
    Output = 5


class IRTensor:  # IR中，表示张量的class
    Name = "top_ir_tensor"
    Format = Format.NHWC
    Type = TensorType.Intermediate
    DataType = DataType.INT8
    ZeroPoint = None
    Scale = None
    Q_min = None
    Q_max = None
    Data = None
    # 输出该张量的 算子的 在AllTensors中的 索引
    OwnerOp = None
    # NumpyData = None
    # Tensor_id = None
    Tensor_idx = None

    def __init__(self):
        self.Shape = Shape(0, 0, 0, 0)

    def __repr__(self):
        return (f'############## Tensor.{self.Tensor_idx} ##############\n'
                f"Name:{self.Name}\n"
                f"Type:{self.Type}\n"
                f"Shape:{self.Shape}\n"
                f"Format:{self.Format}\n"
                f"Data:{self.Data}\n"
                f'############## Tensor.{self.Tensor_idx} ##############\n'
                )

    def load_data(self, np_data):
        self.Data = np_data.copy()

    def load_value(self, value):
        self.Data = value
