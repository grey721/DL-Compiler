from ir.conversion.top2npu.ada200.operator_lowing.transpose import *
from ir.conversion.top2npu.ada200.operator_lowing.reshape import *
from ir.conversion.top2npu.ada200.operator_lowing.concat import *
from ir.conversion.top2npu.ada200.operator_lowing.split import *
from ir.conversion.top2npu.ada200.operator_lowing.base import *
from ir.conversion.top2npu.ada200.operator_lowing.mean import *
from ir.conversion.top2npu.ada200.operator_lowing.pool import *
from ir.conversion.top2npu.ada200.operator_lowing.pad import *
from ir.conversion.top2npu.ada200.operator_lowing.conv import *
from ir.conversion.top2npu.ada200.operator_lowing.resize import *
from ir.conversion.top2npu.ada200.operator_lowing.logistic import *
from ir.conversion.top2npu.ada200.operator_lowing.elemwise import *
# TODO  以后实现
# from ir.conversion.top2npu.ada200.operator_lowing.leakyrelu import *
# from ir.conversion.top2npu.ada200.operator_lowing.softmax import *
# from ir.conversion.top2npu.ada200.operator_lowing.fullconnect import *
# from compiler.conversion.top2npu.ada200.relu import *


lowing_ada200_list = [
    OpTransformRule.TRANSPOSE_LOWERING,
    OpTransformRule.RESHAPE_LOWERING,
    OpTransformRule.CONCAT_LOWERING,
    OpTransformRule.SPLIT_LOWERING,
    OpTransformRule.CONV_LOWERING,
    OpTransformRule.POOL_LOWERING,
    OpTransformRule.MEAM_LOWERING,
    OpTransformRule.PAD_LOWERING,
    OpTransformRule.RESIZE_LOWERING,
    OpTransformRule.ELEMWISE_LOWERING,
    OpTransformRule.LOGISTIC_LOWERING
    # TODO 以后实现
    # OpTransformRule.LEAKYRELU_LOWERING,
    # OpTransformRule.FULLCONNECT_LOWERING,
    # OpTransformRule.SOFTMAX_LOWERING,
    # OpTransformRule.RELU_LOWERING
]
