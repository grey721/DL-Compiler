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

    def __init__(self, tensor_shape):
        self.list = tensor_shape
        dims = len(tensor_shape)
        if dims == 4:  # 不同情况下赋值赋值shape
            self.N, self.C, self.H, self.W = tensor_shape
        elif dims == 3:
            self.C, self.H, self.W = tensor_shape
            self.N = 1
        elif dims == 2:
            self.N, self.C = tensor_shape
            self.W = 1
            self.H = 1
        elif dims == 1:
            self.C = tensor_shape[0]
            self.N = 1
            self.H = 1
            self.W = 1
        elif dims == 5:
            self.N, self.BoxNum, self.BoxInfo, self.fiH, self.fiW = tensor_shape
        #     # (x, y, w h,confidence)
        #     # 批次，3个预测框，预选框信息，尺度且单位像素
        # 维度i,j,f,k,v含义：第 i 张图，第 j 个锚框， 锚框的信息f，第 k 行单元格，第 v 列单元格
        # 网格尺度有三种，小、中、大三个尺度检测目标对象

        else:
            pass

    def __bool__(self):
        return bool(self.list)

    def get_shape(self):
        return self

    def __repr__(self):
        return f"{self.list}"

    def get_shape_as_np(self):
        return np.array(self.list)

    def get_n_shape(self, tensor_format=Format.NCHW):
        """tensor_format = 0:NCHW, 1: NHWC"""
        if len(self.list) < 5:
            if tensor_format == Format.NCHW:
                return [self.N, self.H, self.W, self.C]
            else:
                return [self.N, self.C, self.H, self.W]
        else:
            return self.list

    def get_size(self):
        res = 1
        for i in self.list:
            res *= i
        return res


class TensorType(object):
    Intermediate = 0
    Weight = 1
    Bias = 2
    Parameter = 3
    Const = 4
    Input = 5
    Output = 6


class IRTensor:  # IR中，表示张量的class
    Name = "top_ir_tensor"
    Id = None
    Format = Format.NHWC
    Type = TensorType.Intermediate
    DataType = DataType.INT8
    ZeroPoint = None
    Scale = None
    Q_min = None
    Q_max = None
    Data = None
    # 输出该张量的 算子的 在AllTensors中的 索引
    # NumpyData = None
    # Tensor_id = None
    idx = None

    def __init__(self):
        self.Shape = None
        self.ConsumerOp = None
        self.OwnerOp = None

    def __repr__(self):
        mapping = {
            0: 'Intermediate',
            1: 'Weight',
            2: 'Bias',
            3: 'Parameter',
            5: 'Constant',
            4: 'Input',
            6: 'Output'
        }
        return (
            f'############## Tensor.{self.idx} ##############\n'
            f"Name:{self.Name}\n"
            f"Type:{mapping[self.Type]}\n"
            f'{self.OwnerOp} -> {self.ConsumerOp}\n'
            f"Shape:{self.Shape}\n"
            f"Format:{self.Format}\n"
            # f"Data:{self.Data}\n"
            f'############## Tensor.{self.idx} ##############\n'
        )

    def load_data(self, np_data):
        self.Data = np_data.copy()

    def load_value(self, value):
        self.Data = value
