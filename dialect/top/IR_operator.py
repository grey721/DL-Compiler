from .IR_tensor import *


class OperatorType(object):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    SIGMOID = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115
    ROUND = 116
    HARD_SWISH = 117
    IF = 118
    WHILE = 119
    NON_MAX_SUPPRESSION_V4 = 120
    NON_MAX_SUPPRESSION_V5 = 121
    SCATTER_ND = 122
    SELECT_V2 = 123
    DENSIFY = 124
    SEGMENT_SUM = 125
    BATCH_MATMUL = 126
    PLACEHOLDER_FOR_GREATER_OP_CODES = 127
    CUMSUM = 128
    CALL_ONCE = 129
    BROADCAST_TO = 130
    CONSTANT = 131
    Unsqueeze = 132

    # vpu post op set
    NPU_POST_OP_SET = 150


class OpBase:  # 算子基类
    Skip = False

    TopOpId = None  # 标记该算子在TopIR中的哈希值

    def __init__(self) -> None:
        self.Name = None
        self.InTensors = []
        self.InputShape = []
        self.OutTensors = []
        self.OutputShape = []
        # self.PreOpId = []  # 上一个op id
        # self.PostOpId = []  # 下一个op id

    def load_input_id(self, ir_tensor_id):
        self.InTensors.append(ir_tensor_id)

    def load_output_id(self, ir_tensor_id):
        self.OutTensors.append(ir_tensor_id)

    def get_fmi_size(self):
        fmi_size = self.InputShape[0].C * self.InputShape[0].H * self.InputShape[0].W
        return fmi_size

    def get_fmo_size(self):
        fmo_size = self.OutputShape[0].C * self.OutputShape[0].H * self.OutputShape[0].W
        return fmo_size


# ########################### Constant ########################
class Constant(OpBase):
    Type = "Constant"

    def __init__(self):
        super().__init__()
        self.Mode = None

    def __repr__(self):
        return (
            f'############## Constant.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Constant.{self.TopOpId} ##############\n'
        )

    def shape_inference(self) -> list:
        shape = self.OutputShape[0]
        return [shape.H, shape.W, shape.C]


# ########################### ElemWise ########################
class ElementWiseMode(object):  # 元操作代码
    ELW_ADD = 0
    ELW_SUB = 1
    ELW_MUL = 2  # 元素乘法
    ELW_DIV = 3
    ELW_POW = 4


class ElemWise(OpBase):
    Type = "ElemWise"
    Do_relu = False
    FusedActFunc = 0

    def __init__(self):
        super().__init__()  # 用于在子类实例中初始化父类中的__init__，必须先调用，子类才能有父类的初始化
        self.Mode = None

    def __repr__(self):
        return (
            f'############## ElemWise.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Mode:{self.Mode}'
            f'Input tensor Id:{self.InTensors}\n'
            f'Input shape:{self.InputShape}\n'
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## ElemWise.{self.TopOpId} ##############\n'
        )

    # def get_fmo_size(self):
    # fmi_size = self.InputShape[0].C * self.InputShape[0].H * self.InputShape[0].W
    # return 2 * fmi_size
    def get_input0_scale_numpy(self, graph):
        assert len(self.InTensors), f'No input in {self.Name}'
        return graph.Alltenors[self.InTensors[0]].Scale

    def get_input0_zero_point_numpy(self, graph):
        assert len(self.InTensors), f'No input in {self.Name}'
        return graph.Alltenors[self.InTensors[0]].ZeroPoint

    def get_input1_scale_numpy(self, graph):
        assert len(self.InTensors), f'No input in {self.Name}'
        return graph.Alltenors[self.InTensors[1]].Scale

    def get_input1_zero_point_numpy(self, graph):
        assert len(self.InTensors), f'No input in {self.Name}'
        return graph.Alltenors[self.InTensors[1]].ZeroPoint

    def get_output_scale_numpy(self, graph):
        assert len(self.OutTensors), f'No output in {self.Name}'
        return graph.Alltenors[self.OutTensors[0]].Scale

    def get_output_zero_point_numpy(self, graph):
        assert len(self.OutTensors), f'No output in {self.Name}'
        return graph.Alltenors[self.OutTensors[0]].ZeroPoint

    def shape_inference(self) -> list:
        return [self.InputShape[0].H, self.InputShape[0].W, self.InputShape[0].C]


# ###################  conv related ############################
class Conv2d(OpBase):
    Type = "Conv2d"  # ConvRelu2d
    # Pad
    Padding = None  # 开关
    PadH = None  # 分别表示高度和宽度方向上的填充大小
    PadW = None
    Auto_pads = "PAD_ZERO"
    #  卷积核在相应维度上的膨胀
    Dilation = None
    # 将输入和卷积核分组
    Group = None
    # 偏置值,None则无偏置
    Bias = None
    # 卷积核权重值
    # WeightValue = None
    # 偏置项
    # BiasValue = None
    # 是否是首层
    FirstLayer = False
    KerM_16 = False
    # 激活函数
    Do_relu = False

    def __init__(self):
        super().__init__()
        # 卷积核尺寸
        self.KerH = None
        self.KerW = None

        # 卷积核在输入特征图上滑动的步长
        self.StrideH = None
        self.StrideW = None

    def __repr__(self):
        return (
            f'############## {self.Type}.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Kernel Shape: {[self.KerH, self.KerW]}\n'
            f'Stride: {[self.StrideH, self.StrideW]}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Conv2d.{self.TopOpId} ##############\n'
        )

    # TODO confirm 哪个正确？
    def get_mac(self):
        # mac = self.OutputShape[0].H * self.OutputShape[0].W * self.OutputShape[0].C \
        # * self.KerH * self.KerW * self.InputShape[0].C / self.Group * 2
        ov = self.OutputShape[0].H * self.OutputShape[0].W * self.OutputShape[0].C
        mac = ov * self.KerH * self.KerW * self.InputShape[0].C / self.Group
        if self.Bias:
            mac = mac + ov
        return mac

    def get_weight_size(self):
        weight_size = self.InputShape[0].C * self.OutputShape[0].C * self.KerW * self.KerH
        return weight_size

    def get_weight_numpy(self, graph):
        assert len(self.InTensors)
        weight_tensor = self.InTensors[1]
        weight = graph.get_tensor(weight_tensor).NumpyData
        if len(weight.shape) == 2:
            weight = np.transpose(weight, (1, 0))
            # axis：这是你想要增加新维度的位置。axis=0意味着新维度将被添加到张量的最前面。
            weight = np.expand_dims(weight, axis=0)
            weight = np.expand_dims(weight, axis=0)
            # 它根据提供的元组(3, 0, 1, 2)来改变weight数组维度的顺序。
            weight = np.transpose(weight, (3, 0, 1, 2))
        return weight

    def get_bias_numpy(self, graph):
        if self.Bias:
            bias_tensor = self.InTensors[2]
            bias = graph.get_tensor(bias_tensor).NumpyData
            return bias
        return np.zeros(self.OutputShape[0].C, dtype=np.int32)

    def get_input_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.Alltenors[self.InTensors[0]].Scale

    def get_output_scale_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.Alltenors[self.OutTensors[0]].Scale

    def get_input_zero_point_numpy(self, graph):
        assert len(self.InTensors)
        return graph.Alltenors[self.InTensors[0]].ZeroPoint

    def get_output_zero_point_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.Alltenors[self.OutTensors[0]].ZeroPoint

    def get_weight_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.Alltenors[self.InTensors[1]].Scale

    # def GetQuantWeightZeroPointNumpy(self):
    #     return self.InTensors[1].ZeroPoint

    def get_bias_scale_numpy(self, graph):
        if self.Bias:
            return graph.Alltenors[self.InTensors[2]].Scale
        return None

    # 推断经过padding和卷积后图像的新形状
    def shape_inference(self) -> list:
        h = self.InputShape[0].H
        w = self.InputShape[0].W

        pad_h = self.PadH
        pad_w = self.PadW
        n_h = int((h + pad_h * 2 - self.KerH) / self.StrideH + 1)
        n_w = int((w + pad_w * 2 - self.KerW) / self.StrideW + 1)
        n_c = self.InputShape[0].C

        return [n_h, n_w, n_c]

    # def GetQuantBiasZeroPointNumpy(self):
    #     if self.Bias:
    #         return self.InTensors[2].ZeroPoint
    #     return None


# ###################### Pool ######################
class PoolMode(object):
    POOL_AVG = 1
    POOL_MAX = 0


class Pool(OpBase):
    Type = 'Pool'
    KerH = None
    KerW = None
    StrideH = None
    StrideW = None
    Padding = None
    PadH = None
    PadW = None
    do_relu = False

    def __init__(self):
        super().__init__()
        self.Mode = None

    def __repr__(self):
        return (
            f'############## {self.Type}.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Mode:{self.Mode}\n'
            f'Kernel Shape: {[self.KerH, self.KerW]}\n'
            f'Stride: {[self.StrideH, self.StrideW]}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Conv2d.{self.TopOpId} ##############\n'
        )

    def shape_inference(self) -> list:
        h = self.InputShape[0].H
        w = self.InputShape[0].W

        n_h = int(h / self.StrideH)
        n_w = int(w / self.StrideW)
        n_c = self.InputShape[0].C

        return [n_h, n_w, n_c]
    

# ############################ activation ###########################
class RELUMode(object):
    RELU = 0
    PRELU = 1
    LEAKY_RELU = 2
    # SIGMOID = 3
    # HARDSWISH = 4
    # SOFTMAX = 5


class RELU(OpBase):
    Type = "RELU"
    Mode = RELUMode.RELU
    Alpha = 0
    MaxLimit = None

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return (
            f'############## {self.Type}.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'            
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## {self.Type}.{self.TopOpId} ##############\n'
        )

    def get_input_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].Scale

    def get_input_zero_point_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].ZeroPoint

    def get_output_scale_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].Scale

    def get_output_zero_point_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].ZeroPoint

    def shape_inference(self) -> list:
        return [self.InputShape[0].H, self.InputShape[0].W, self.InputShape[0].C]


class Sigmoid(OpBase):
    Type = 'Sigmoid'

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return (
            f'############## Sigmoid.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'            
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Sigmoid.{self.TopOpId} ##############\n'
        )

    def get_input_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].Scale

    def get_input_zero_point_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].ZeroPoint

    def get_output_scale_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].Scale

    def get_output_zero_point_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].ZeroPoint

    def shape_inference(self) -> list:
        return [self.InputShape[0].H, self.InputShape[0].W, self.InputShape[0].C]


# ###################  tensor related ############################
class ResizeMode(object):
    RESIZE_BILINEAR = 1  # 双线性插值
    RESIZE_NEAREST = 0  # 最近邻插值


class Resize(OpBase):
    Type = "Resize"

    AlignCorners = False  # 对齐角点，考虑图像角点的精确对齐
    HalfPixelCenters = False  # 半像素中心，像素中心点位于像素网格的半像素位置

    def __init__(self):
        super().__init__()
        self.Name = None
        self.Mode = None
        self.ScaleFactor = None  # 缩放因子

    def __repr__(self):
        mapping = {
            1: 'BILINEAR',
            0: 'NEAREST'
        }
        return (
            f'############## Resize.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Mode:{mapping[self.Mode]}\n'
            f'ScaleFactor:{self.ScaleFactor}\n'
            f'AlignCorners:{self.AlignCorners}; HalfPixelCenters:{self.HalfPixelCenters}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'            
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Resize.{self.TopOpId} ##############\n'
        )

    def shape_inference(self, shape_list):
        h, w, c = shape_list
        n_h = h * self.ScaleFactor
        n_w = w * self.ScaleFactor
        n_c = c
        return [n_h, n_w, n_c]


class Concat(OpBase):
    Type = "Concat"
    # 链接的轴
    Axis = None
    FusedActFunc = False

    # TODO what this
    # quant_param
    RescaleInput = -1

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return_map = {
            0: 'N',
            1: 'H',
            2: 'W',
            3: 'C'
        }
        return (
            f'############## Concat.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Concat Axis:{return_map[self.Axis]}\n'
            f'Input tensors Id:{self.InTensors}\n'
            f'Inputs shape:{self.InputShape}\n'
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Concat.{self.TopOpId} ##############\n'
        )

    def get_fmi_size(self):
        fmi_size = self.InputShape[0].C * self.InputShape[0].H * self.InputShape[0].W
        fmi1_size = self.InputShape[1].C * self.InputShape[1].H * self.InputShape[1].W
        return fmi_size + fmi1_size

    def get_input_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].Scale

    def get_input_zero_point_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[0]].ZeroPoint

    def get_input1_scale_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[1]].Scale

    def get_input1_zero_point_numpy(self, graph):
        assert len(self.InTensors)
        return graph.AllTensors[self.InTensors[1]].ZeroPoint

    def get_output_scale_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].Scale

    def get_output_zero_point_numpy(self, graph):
        assert len(self.OutTensors)
        return graph.AllTensors[self.OutTensors[0]].ZeroPoint

    def shape_inference(self):
        shape1_list = [self.InputShape[0].N, self.InputShape[0].H, self.InputShape[0].W, self.InputShape[0].C]
        shape2_list = [self.InputShape[1].N, self.InputShape[1].H, self.InputShape[1].W, self.InputShape[1].C]

        for i in range(4):
            if i != self.Axis:
                assert shape1_list[i] == shape2_list[i]
            else:
                shape1_list[i] = shape1_list[i] + shape2_list[i]
        return shape1_list


class Reshape(OpBase):
    Type = "Reshape"

    def __init__(self):
        super().__init__()
        self.Name = None
        self.out_shape = None

    def __repr__(self):
        return (
            f'############## Reshape.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Reshape:{self.out_shape}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'    
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Reshape.{self.TopOpId} ##############\n'
        )


class Transpose(OpBase):
    Type = "Transpose"
    OutDimOrder = None

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return (
            f'############## Transpose.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'ReDim(NHWC):{self.OutDimOrder}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'    
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Transpose.{self.TopOpId} ##############\n'
        )


class Pad(OpBase):
    Type = "Pad"
    pad_val = None
    pad_mode = "constant"
    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return (
            f'############## Pad.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Pad_val:{self.pad_val}\n'
            f'          Top:{self.pad_top}\n'
            f'Left:{self.pad_left}                    Right:{self.pad_right}\n'
            f'          Bottom:{self.pad_bottom}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'    
            f'Output tensor Id:{self.OutTensors[0]}\n'
            f'Output shape:{self.OutputShape[0]}\n'
            f'############## Pad.{self.TopOpId} ##############\n'
        )


class Split(OpBase):
    Type = "Split"
    Axis = None
    split_shape = None

    def __init__(self):
        super().__init__()
        self.Name = None

    def __repr__(self):
        return (
            f'############## Split.{self.TopOpId} ##############\n'
            f'Op Name:{self.Name}\n'
            f'Split:{self.split_shape}; Axis:{self.Axis}\n'
            f'Input tensor Id:{self.InTensors[0]}\n'
            f'Input shape:{self.InputShape[0]}\n'    
            f'Output tensor Id:{self.OutTensors}\n'
            f'Output shape:{self.OutputShape}\n'
            f'############## Split.{self.TopOpId} ##############\n'
        )

    def shape_inference(self):
        shape_list = [self.InputShape[0].N, self.InputShape[0].H, self.InputShape[0].W, self.InputShape[0].C]
        n_shape = []
        for i in range(len(self.split_shape)):
            out = []
            for j in range(4):
                if j == self.Axis:
                    out.append(self.split_shape[i])
                else:
                    out.append(shape_list[j])
            n_shape.append(out)

        return n_shape
