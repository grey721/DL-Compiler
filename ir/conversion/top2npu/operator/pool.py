from ir.conversion.top2npu.operator.base import OpTransformRule, _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *


@_register_op_transformation_rule(OpTransformRule.POOL_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Pool):
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
    npu_pool = NpuPool()
    npu_pool.__dict__.update(op.__dict__)
    # npu_pool.Name = "NpuPool"
    OutputH = op.OutputShape[0].H
    OutputW = op.OutputShape[0].W
    InputH = op.InputShape[0].H
    InputW = op.InputShape[0].W
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    # OutputH * StrideH，OutputH的每一个格子代表滑动过一次，则原图最大边长为滑动次数OutputH * 每次滑动的步长StrideH
    # 但最后一次滑动后，filter应该正好覆盖图片（因为pad过），所以最后一个长度不一定是步长，而是filter的长度
    H = OutputH * StrideH - StrideH + KerH
    W = OutputW * StrideW - StrideW + KerW

    # TODO 为什么不直接用Top IR中获取
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


def _lowering_none(op):
    npu_pool = NpuPool()
    npu_pool.__dict__.update(op.__dict__)
    # npu_pool.Name = "NpuPool"
    OutputH = op.OutputShape[0].H
    OutputW = op.OutputShape[0].W
    InputH = op.InputShape[0].H
    InputW = op.InputShape[0].W
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    # OutputH * StrideH，OutputH的每一个格子代表滑动过一次，则原图最大边长为滑动次数OutputH * 每次滑动的步长StrideH
    # 但最后一次滑动后，filter应该正好覆盖图片（因为pad过），所以最后一个长度不一定是步长，而是filter的长度
    H = OutputH * StrideH - StrideH + KerH
    W = OutputW * StrideW - StrideW + KerW

    # TODO 为什么不直接用Top IR中获取
    npu_pool.pad_top = op.PadH
    npu_pool.pad_bottom = H - op.PadH - InputH
    npu_pool.pad_left = op.PadW
    npu_pool.pad_right = W - op.PadW - InputW

    return npu_pool


def _lowering_fp32(op):
    npu_pool = NpuPool()
    npu_pool.__dict__.update(op.__dict__)
    # npu_pool.Name = "NpuPool"
    return npu_pool
