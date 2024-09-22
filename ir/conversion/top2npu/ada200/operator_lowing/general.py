from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule,  _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *


@_register_op_transformation_rule(OpTransformRule.GENERAL_LOWERING)
def _lowering(net, mode):

    applicable_ops = (Reshape, Split, Mean, Concat, Pad, Transpose, Slice, Shape, Floor)

    for op in net.AllOps:
        if isinstance(op, applicable_ops):
            NpuOp = _lowering_none(op)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_none(op):
    npu_op = NpuReshape()
    npu_op.__dict__.update(op.__dict__)
    # npu_op.Name = "Npu" + op.Type
    return npu_op
