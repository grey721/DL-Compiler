from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.dialect.npu.ir.ir_operator import *


def get_ratio_r(input_shape, output_shape):
    in_height = input_shape[1]
    out_height = output_shape[1]
    in_width = input_shape[2]
    out_width = output_shape[2]
    ratio_h = (int)(((1 << 10) * in_height + out_height / 2) / out_height)
    ratio_w = (int)(((1 << 10) * in_width + out_width / 2) / out_width)

    return ratio_w, ratio_h


@_register_tranformation_rule(TransformRule.RESIZE_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Resize):
            if mode == "int8":
                NpuOp = _lowering_int8(op)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op):
    npu_resize = NpuResize()
    npu_resize.__dict__.update(op.__dict__)
    npu_resize.Name = "NpuResize"

    input_shape = [npu_resize.InputN, npu_resize.InputH, npu_resize.InputW, npu_resize.InputC]
    output_shape = [npu_resize.outputN, npu_resize.OutputH, npu_resize.OutputW, npu_resize.OutputC]
    ratio_w, ratio_h = get_ratio_r(input_shape, output_shape)
    npu_resize.ratio_w = ratio_w
    npu_resize.ratio_h = ratio_h

    return npu_resize


def _lowering_fp32(op):
    npu_resize = NpuResize()
    npu_resize.__dict__.update(op.__dict__)
    npu_resize.Name = "NpuResize"
    return npu_resize