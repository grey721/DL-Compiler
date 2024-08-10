from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.conversion.top2npu.top_lowing import *
from compiler.dialect.npu.ir.ir_operator import *


@_register_tranformation_rule(TransformRule.POOL_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Pool):
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    npu_pool = NpuPool()
    npu_pool.__dict__.update(op.__dict__)
    npu_pool.Name = "NpuPool"
    OutputH = op.OutputH
    OutputW = op.OutputW
    InputH = op.InputH
    InputW = op.InputW
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    H = OutputH*StrideH - StrideH + KerH
    W = OutputW*StrideW - StrideW + KerW
    npu_pool.pad_top = op.PadH
    npu_pool.pad_bottom = H - op.PadH - InputH
    npu_pool.pad_left = op.PadW
    npu_pool.pad_right = W - op.PadW - InputW

    # output_scale = op.GetQuantOutputScaleNumpy(net)

    # if npu_pool.do_relu == True:
    #     npu_pool.quantized_activation_max, npu_pool.quantized_activation_min \
    #         = get_activation_range(output_scale, npu_pool.output_offset, "RELU")
    # else:
    #     npu_pool.quantized_activation_max, npu_pool.quantized_activation_min \
    #         = get_activation_range(output_scale, npu_pool.output_offset, None)

    return npu_pool


def _lowering_fp32(op):
    npu_pool = NpuPool()
    npu_pool.__dict__.update(op.__dict__)
    npu_pool.Name = "NpuPool"
    return npu_pool