from ir.conversion.top2npu.ada200.base import *
# from ir.conversion.top2npu.ada200.conv import *
# from ir.conversion.top2npu.ada200.leakyrelu import *
# from ir.conversion.top2npu.ada200.pool import *
# from ir.conversion.top2npu.ada200.resize import *
# from ir.conversion.top2npu.ada200.logistic import *
# from ir.conversion.top2npu.ada200.concat import *
# from ir.conversion.top2npu.ada200.reshape import *
# from ir.conversion.top2npu.ada200.transpose import *
# from ir.conversion.top2npu.ada200.softmax import *
# from ir.conversion.top2npu.ada200.mean import *
# from ir.conversion.top2npu.ada200.elemwise import *
# from ir.conversion.top2npu.ada200.pad import *
# from ir.conversion.top2npu.ada200.fullconnect import *
#todo
# from compiler.conversion.top2npu.ada200.relu import *

lowing_ada200_map = []
lowing_ada200_map.append(TransformRule.CONV_LOWERING)
lowing_ada200_map.append(TransformRule.LEAKYRELU_LOWERING)
lowing_ada200_map.append(TransformRule.POOL_LOWERING)
lowing_ada200_map.append(TransformRule.RESIZE_LOWERING)
lowing_ada200_map.append(TransformRule.CONCAT_LOWERING)
lowing_ada200_map.append(TransformRule.RESHAPE_LOWERING)
lowing_ada200_map.append(TransformRule.TRANSPOSE_LOWERING)
lowing_ada200_map.append(TransformRule.ELEMWISE_LOWERING)
lowing_ada200_map.append(TransformRule.SOFTMAX_LOWERING)
lowing_ada200_map.append(TransformRule.FULLCONNECT_LOWERING)
lowing_ada200_map.append(TransformRule.MEAM_LOWERING)
lowing_ada200_map.append(TransformRule.PAD_LOWERING)
lowing_ada200_map.append(TransformRule.LOGISTIC_LOWERING)
# lowing_ada200_map.append(TransformRule.RELU_LOWERING)