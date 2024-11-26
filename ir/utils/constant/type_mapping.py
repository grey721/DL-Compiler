from enum import Enum


class Layout(Enum):  # 高维张量的展开模式
    NCHW = 0
    NHWC = 1


# NCHW -> NHWC
axis_map = {
    0: 0,
    1: 3,
    2: 1,
    3: 2
}


class TensorType(Enum):
    Intermediate = 0
    Weight = 1
    Bias = 2
    Parameter = 3
    Const = 4
    Input = 5
    Output = 6


class DataType(Enum):  # 数据类型ID
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
