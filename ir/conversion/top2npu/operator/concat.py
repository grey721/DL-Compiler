from ir.conversion.top2npu.operator.base import OpTransformRule, _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *


@_register_op_transformation_rule(OpTransformRule.CONCAT_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Concat):
            if mode is None:
                NpuOp = _lowering_none(op)

            elif mode == DataType.INT8:
                NpuOp = _lowering_int8(op)

            elif mode == DataType.FLOAT32:
                NpuOp = _lowering_fp32(op)

            else:
                raise NotImplementedError('Unsupported lowing mode')

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op):
    npu_concat = NpuConcat()
    npu_concat.__dict__.update(op.__dict__)
    # npu_concat.Name = "NpuConcat"
    return npu_concat


def _lowering_fp32(op):
    raise NotImplementedError


def _lowering_none(op):
    npu_concat = NpuConcat()
    npu_concat.__dict__.update(op.__dict__)
    # npu_concat.Name = "NpuConcat"
    return npu_concat
