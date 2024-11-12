from ir.conversion.top2npu.operator.base import OpTransformRule,  _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *


def get_ratio_r(input_shape, output_shape):
    in_height = input_shape[1]
    out_height = output_shape[1]
    in_width = input_shape[2]
    out_width = output_shape[2]
    # TODO ????
    ratio_h = int(((1 << 10) * in_height + out_height / 2) / out_height)
    ratio_w = int(((1 << 10) * in_width + out_width / 2) / out_width)

    return ratio_w, ratio_h


@_register_op_transformation_rule(OpTransformRule.RESIZE_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Resize):
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
    npu_resize = NpuResize()
    npu_resize.__dict__.update(op.__dict__)
    # npu_resize.Name = "NpuResize"

    input_shape = [npu_resize.InputShape[0].N,
                   npu_resize.InputShape[0].H,
                   npu_resize.InputShape[0].W,
                   npu_resize.InputShape[0].C]

    output_shape = [npu_resize.OutputShape[0].N,
                    npu_resize.OutputShape[0].H,
                    npu_resize.OutputShape[0].W,
                    npu_resize.OutputShape[0].C]

    ratio_w, ratio_h = get_ratio_r(input_shape, output_shape)
    npu_resize.ratio_w = ratio_w
    npu_resize.ratio_h = ratio_h

    return npu_resize


def _lowering_fp32(op):
    npu_resize = NpuResize()
    npu_resize.__dict__.update(op.__dict__)
    # npu_resize.Name = "NpuResize"
    return npu_resize


def _lowering_none(op):
    npu_resize = NpuResize()
    npu_resize.__dict__.update(op.__dict__)
    # npu_resize.Name = "NpuResize"
    return npu_resize

