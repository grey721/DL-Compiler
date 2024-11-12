from ir.conversion.top2npu.operator.transpose import *
from ir.conversion.top2npu.operator.reshape import *
from ir.conversion.top2npu.operator.concat import *
from ir.conversion.top2npu.operator.base import *
from ir.conversion.top2npu.operator.mean import *
from ir.conversion.top2npu.operator.pool import *
from ir.conversion.top2npu.operator.pad import *
from ir.conversion.top2npu.operator.conv import *
from ir.conversion.top2npu.operator.resize import *
from ir.conversion.top2npu.operator.logistic import *
from ir.conversion.top2npu.operator.elemwise import *
from ir.conversion.top2npu.operator.split import *
# TODO  以后实现
from ir.conversion.top2npu.operator.leakyrelu import *
# from ir.conversion.top2npu.operator.softmax import *
# from ir.conversion.top2npu.operator.fullconnect import *
# from compiler.conversion.top2npu.operator.relu import *


lowing_ada200_list = [
    OpTransformRule.TRANSPOSE_LOWERING,
    OpTransformRule.RESHAPE_LOWERING,
    OpTransformRule.CONCAT_LOWERING,
    OpTransformRule.CONV_LOWERING,
    OpTransformRule.POOL_LOWERING,
    OpTransformRule.MEAM_LOWERING,
    OpTransformRule.PAD_LOWERING,
    OpTransformRule.RESIZE_LOWERING,
    OpTransformRule.ELEMWISE_LOWERING,
    OpTransformRule.LOGISTIC_LOWERING,
    OpTransformRule.SPLIT_LOWERING,
    OpTransformRule.LEAKYRELU_LOWERING,
]
