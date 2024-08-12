from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.dialect.npu.ir.ir_operator import *


@_register_tranformation_rule(TransformRule.MEAM_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Mean):
            if mode == "int8":
                NpuOp = _lowering_int8(op)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op):
    npu_mean = NpuMean()
    npu_mean.__dict__.update(op.__dict__)
    npu_mean.Name = "NpuMean"
    return npu_mean


def _lowering_fp32(op):
    npu_mean = NpuMean()
    npu_mean.__dict__.update(op.__dict__)
    npu_mean.Name = "NpuMean"
    return npu_mean