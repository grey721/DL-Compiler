from ir.conversion.top2npu.ada200.operator_lowing.base import *
# from ir.conversion.top2npu.ada200.operator_lowing.conv import *
# from ir.conversion.top2npu.ada200.operator_lowing.leakyrelu import *
from ir.conversion.top2npu.ada200.operator_lowing.pool import *
# from ir.conversion.top2npu.ada200.operator_lowing.resize import *
from ir.conversion.top2npu.ada200.operator_lowing.sigmoid import *
from ir.conversion.top2npu.ada200.operator_lowing.concat import *
from ir.conversion.top2npu.ada200.operator_lowing.reshape import *
from ir.conversion.top2npu.ada200.operator_lowing.transpose import *
# from ir.conversion.top2npu.ada200.operator_lowing.elemwise import *
from ir.conversion.top2npu.ada200.operator_lowing.pad import *

# from ir.conversion.top2npu.ada200.operator_lowing.softmax import *
# from ir.conversion.top2npu.ada200.operator_lowing.mean import *
# from ir.conversion.top2npu.ada200.operator_lowing.fullconnect import *
# todo
#  from compiler.conversion.top2npu.ada200.relu import *


lowing_ada200_list = [OpTransformRule.CONV_LOWERING,
                      OpTransformRule.LEAKYRELU_LOWERING,
                      OpTransformRule.POOL_LOWERING,
                      OpTransformRule.RESIZE_LOWERING,
                      OpTransformRule.CONCAT_LOWERING,
                      OpTransformRule.RESHAPE_LOWERING,
                      OpTransformRule.TRANSPOSE_LOWERING,
                      OpTransformRule.ELEMWISE_LOWERING,
                      OpTransformRule.SOFTMAX_LOWERING,
                      OpTransformRule.FULLCONNECT_LOWERING,
                      OpTransformRule.MEAM_LOWERING,
                      OpTransformRule.PAD_LOWERING,
                      OpTransformRule.LOGISTIC_LOWERING
                      ]
# lowing_ada200_map.append(OpTransformRule.RELU_LOWERING)