import math

from ir.conversion.ir_transform import _register_ir_transformation_rule
from ir.dialect.npu.IR_operator import *
from ir.graph.Graph_IR import *


class TransformRule(Enum):
    NOPE = 1

    WEIGHT_MAPPING = 2


@_register_ir_transformation_rule(TransformRule.WEIGHT_MAPPING)
def _weight_mapping(net: GraphIR):
    for npu_op in net.AllOps:
        if isinstance(npu_op, NpuOp):
            npu_conv_op = npu_op.NpuOpConvOp
            if npu_conv_op is None:
                continue
            npu_op_id = npu_op.NpuOpId
            weight = {
                "weight": npu_conv_op.WeightValue.tolist(),

                "bias": npu_conv_op.BiasValue.tolist()
            }
            net.add_weight_tensor(npu_op_id, weight)


# weight_mapping_pass
weight_mapping_transform = [
    TransformRule.WEIGHT_MAPPING,
]
