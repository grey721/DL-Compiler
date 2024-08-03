import onnx
from .Graph_IR import *
from .IR_tensor import *
from .IR_operator import *

SKIP_OPs = []
# TODO: Handle RESHAPE during codegen
SUPPORTED_OPs = [
    OperatorType.CONV_2D,
    OperatorType.AVERAGE_POOL_2D,
    OperatorType.DEPTHWISE_CONV_2D,
    OperatorType.MAX_POOL_2D,
    OperatorType.RESHAPE,
    # OperatorType.FULLY_CONNECTED,
    # OperatorType.SOFTMAX,
    OperatorType.LEAKY_RELU,
    # OperatorType.RESIZE_NEAREST_NEIGHBOR,
    # OperatorType.RESIZE_BILINEAR,
    OperatorType.CONCATENATION,
    OperatorType.TRANSPOSE,
    # OperatorType.MEAN,
    # OperatorType.PAD,
    OperatorType.ADD,
    # OperatorType.DEQUANTIZE,
    # OperatorType.CUSTOM,
    # OperatorType.QUANTIZE,
    OperatorType.RELU,

    # New
    OperatorType.MUL,
    OperatorType.CONSTANT,
    # OperatorType.SPLIT,
    OperatorType.SIGMOID,
    OperatorType.PRELU,
    OperatorType.POW
]

FUSIBLE_OPs = [
    OperatorType.RELU,
    OperatorType.RELU6,
    OperatorType.FULLY_CONNECTED,
    OperatorType.AVERAGE_POOL_2D,
    OperatorType.MEAN
]

ONNXType2OperatorType = {
    'Add': 0,
    'AveragePool': 1,
    'GlobalAveragePool': 1,
    'Concat': 2,
    'Conv': 3,
    # 'DEPTHWISE_CONV_2D' : 4, # use Conv
    'DepthToSpace': 5,
    'DequantizeLinear': 6,
    'EMBEDDING_LOOKUP': 7,  #
    'Floor': 8,
    'Gemm': 9,
    'HASHTABLE_LOOKUP': 10,  #
    'LpNormalization': 11,
    'LpPool': 12,
    'LOCAL_RESPONSE_NORMALIZATION': 13,  #
    'Sigmoid': 14,
    'LSH_PROJECTION': 15,  #
    'LSTM': 16,
    'MaxPool': 17,
    'Mul': 18,
    'Relu': 19,
    'RELU_N1_TO_1': 20,  #
    'Clip': 21,
    'Reshape': 22,
    'Resize': 23,
    'RNN': 24,
    'Softmax': 25,
    'SpaceToDepth': 26,
    'SVDF': 27,  #
    'Tanh': 28,
    'CONCAT_EMBEDDINGS': 29,  #
    'SKIP_GRAM': 30,  #
    'CALL': 31,  #
    'CUSTOM': 32,  #
    'EMBEDDING_LOOKUP_SPARSE': 33,  #
    'Pad': 34,
    'UNIDIRECTIONAL_SEQUENCE_RNN': 35,  #
    'Gather': 36,
    'BATCH_TO_SPACE_ND': 37,  #
    'SPACE_TO_BATCH_ND': 38,  #
    'Transpose': 39,
    'Mean': 40,
    'Flatten': 40,
    'Sub': 41,
    'Div': 42,
    'Squeeze': 43,
    'UNIDIRECTIONAL_SEQUENCE_LSTM': 44,  #
    'STRIDED_SLICE': 45,
    'BIDIRECTIONAL_SEQUENCE_RNN': 46,  #
    'Exp': 47,
    'TopK': 48,
    'Split': 49,
    'LogSoftmax': 50,
    'DELEGATE': 51,  #
    'BIDIRECTIONAL_SEQUENCE_LSTM': 52,  #
    'Cast': 53,
    'PRelu': 54,
    'MAXIMUM': 55,  #
    'ArgMax': 56,
    'MINIMUM': 57,  #
    'Less': 58,
    'Neg': 59,
    'PADV2': 60,  #
    'Greater': 61,
    'GreaterOrEqual': 62,
    'LessOrEqual': 63,
    'SELECT': 64,  #
    'Slice': 65,
    'Sin': 66,
    'ConvTranspose': 67,
    'SPARSE_TO_DENSE': 68,  #
    'Tile': 69,
    'EXPAND_DIMS': 70,  #
    'Equal': 71,
    'NOT_EQUAL': 72,  #
    'Log': 73,
    'Sum': 74,
    'Sqrt': 75,
    'RSQRT': 76,  #
    'Shape': 77,
    'Pow': 78,
    'ArgMin': 79,
    'FAKE_QUANT': 80,  #
    'ReduceProd': 81,
    'ReduceMax': 82,
    'PACK': 83,  #
    'Or': 84,
    'OneHot': 85,
    'And': 86,
    'Not': 87,
    'UNPACK': 88,  #
    'ReduceMin': 89,
    'FLOOR_DIV': 90,  #
    'REDUCE_ANY': 91,  #
    'SQUARE': 92,  #
    'ZEROS_LIKE': 93,  #
    'FILL': 94,
    'FLOOR_MOD': 95,  #
    'Range': 96,
    'RESIZE_NEAREST_NEIGHBOR': 97,  #
    'LeakyRelu': 98,
    'SQUARED_DIFFERENCE': 99,  #
    'MIRROR_PAD': 100,  #
    'Abs': 101,
    'SPLIT_V': 102,  #
    'Unique': 103,
    'Ceil': 104,
    'REVERSE_V2': 105,  #
    'ADD_N': 106,  #
    'GatherND': 107,
    'Cos': 108,
    'Where': 109,
    'RANK': 110,  #
    'Elu': 111,
    'ReverseSequence': 112,
    'MATRIX_DIAG': 113,  #
    'QuantizeLinear': 114,
    'MATRIX_SET_DIAG': 115,  #
    'Round': 116,
    'HardSwish': 117,
    'If': 118,
    'WHILE': 119,  #
    'NON_MAX_SUPPRESSION_V4': 120,  #
    'NON_MAX_SUPPRESSION_V5': 121,  #
    'ScatterND': 122,
    'SELECT_V2': 123,  #
    'DENSIFY': 124,  #
    'SEGMENT_SUM': 125,  #
    'MatMul': 126,
    'PLACEHOLDER_FOR_GREATER_OP_CODES': 127,  #
    'CumSum': 128,
    'CALL_ONCE': 129,  #
    'BROADCAST_TO': 130,  #
    'Constant': 131,
    'Unsqueeze': 132,

    # vpu post op set
    'NPU_POST_OP_SET': 150  #

}
# 整型也可以做字典的键，该字典用于转换数据类型，mapping：映射   onnx.TensorProto，ONNX提供的数据类型枚举
onnx2np_dtype_mapping = {
    # pylint: disable=no-member
    None: None,
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.UINT16: np.uint16,
}
# TODO confirm it
parameter_input = {
    OperatorType.PAD,
    OperatorType.PADV2,
    OperatorType.RESHAPE,
    OperatorType.RESIZE_NEAREST_NEIGHBOR,
    OperatorType.RESIZE_BILINEAR,
}
# NCHW -> NHWC
axis_map = {
    0: 0,
    1: 3,
    2: 1,
    3: 2
}

