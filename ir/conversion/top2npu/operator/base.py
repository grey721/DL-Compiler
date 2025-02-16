from typing import Callable, Dict
from ir.utils.constant.type_mapping import *


class OpTransformRule(Enum):
    NOPE = 0
    GENERAL_LOWERING = 1
    CONV_LOWERING = 2
    LEAKYRELU_LOWERING = 3
    POOL_LOWERING = 4
    RESIZE_LOWERING = 5
    CONCAT_LOWERING = 6
    RESHAPE_LOWERING = 7
    TRANSPOSE_LOWERING = 8
    ELEMWISE_LOWERING = 9
    SOFTMAX_LOWERING = 10
    FULLCONNECT_LOWERING = 11
    MEAM_LOWERING = 12
    PAD_LOWERING = 13
    LOGISTIC_LOWERING = 14
    RELU_LOWERING = 15
    SPLIT_LOWERING = 16
    CONSTANT_LOWERING = 17


ADA300_TRANSFORM_MAP: Dict[Enum, Callable] = {}


# 装饰器首次应用于函数时，装饰器函数会被调用一次
def _register_op_transformation_rule(transform_rule):  # 装饰器工厂，ADA200_TRANSFORM_MAP[rule]赋值为函数impl
    def callback(impl):
        ADA300_TRANSFORM_MAP[transform_rule] = impl

    return callback
