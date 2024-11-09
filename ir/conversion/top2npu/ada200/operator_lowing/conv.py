import numpy as np

from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule,  _register_op_transformation_rule
from ir.conversion.top2npu.top_lowing import *
from ir.dialect.npu.IR_operator import *


@ _register_op_transformation_rule(OpTransformRule.CONV_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, ConvBase):
            if mode is None:
                NpuOp = _lowering_none(op, net)

            elif mode == DataType.INT8:
                NpuOp = _lowering_int8(op, net)

            elif mode == DataType.FLOAT32:
                NpuOp = _lowering_fp32(op, net)

            else:
                raise NotImplementedError('Unsupported lowing mode')

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    NpuConv = NpuConv2d()
    NpuConv.__dict__.update(op.__dict__)
    # NpuConv.Name = "NpuConv2d"

    weight_scale = op.get_weight_scale_numpy(net)
    input_scale = op.get_input_scale_numpy(net)
    output_scale = op.get_output_scale_numpy(net)

    # 浮点数Scale, 可以转换成Multiplier和rshift，替代传统的浮点数运算
    # 使用乘法器 M 的目的是在整数运算中实现对原始浮点数乘法的近似。
    # 这样，在执行矩阵乘法或其他涉及权重的运算时，可以通过整数乘法和后续的右移位操作来模拟浮点数的乘法。
    scale = norm(weight_scale) * norm(input_scale) / norm(output_scale)
    # bias + weight * input_zp
    weight = op.get_weight_numpy(net)
    zp = op.get_input_zero_point_numpy(net)
    bias = op.get_bias_numpy(net)
    # TODO 溢出了 是否需要改为int32和int16存储offset？
    for oc in range(op.OutputShape[0].C):
        offset_sum = np.int32(0)
        for ic in range(int(op.InputShape[0].C / op.Group)):
            for kh in range(op.KerH):
                for kw in range(op.KerW):
                    offset_sum += np.int32(weight[oc][kh][kw][ic]) * np.int32(zp[0])
        # TODO 推导中为什么直接将B_3 换成了 B ?
        bias[oc] -= offset_sum
    NpuConv.BiasValue = bias
    NpuConv.WeightValue = NpuConv.get_weight_numpy(net)

    NpuConv.input_offset = op.get_input_zero_point_numpy(net)
    NpuConv.weights_offset = op.get_weight_scale_numpy(net)
    NpuConv.output_offset = op.get_output_zero_point_numpy(net)

    if len(scale) != op.OutputShape[0].C:  # 逐层量化
        assert len(scale) == 1
        assert len(NpuConv.weights_offset) == 1

        scale = np.full(op.OutputShape[0].C, scale[0], dtype=scale.dtype)

        NpuConv.weights_offset = np.full(op.OutputShape[0].C,  # 新张量的形状，整型则是一维数组
                                         NpuConv.weights_offset[0],  # 设置张量中的值
                                         dtype=NpuConv.weights_offset.dtype)

    (NpuConv.output_multiplier, NpuConv.output_shift) = QuantizeMultiplier(scale)

    if NpuConv.do_relu:
        NpuConv.quantized_activation_max, NpuConv.quantized_activation_min \
            = get_activation_range(output_scale, NpuConv.output_offset, "SIGMOID")

    OutputH = op.OutputShape[0].H
    OutputW = op.OutputShape[0].W
    InputH = op.InputShape[0].H
    InputW = op.InputShape[0].W
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    H = OutputH * StrideH - StrideH + KerH
    W = OutputW * StrideW - StrideW + KerW
    NpuConv.pad_top = op.PadH
    NpuConv.pad_bottom = H - op.PadH - InputH
    NpuConv.pad_left = op.PadW
    NpuConv.pad_right = W - op.PadW - InputW
   
    return NpuConv


def _lowering_fp32(op, net):
    NpuConv = NpuConv2d()
    NpuConv.__dict__.update(op.__dict__)
    # NpuConv.Name = "NpuConv2d"
    raise NotImplementedError


def _lowering_none(op, net):
    NpuConv = NpuConv2d()
    NpuConv.__dict__.update(op.__dict__)
    # NpuConv.Name = "NpuConv2d"

    # 加载参数
    weight = op.get_weight_numpy(net)
    bias = op.get_bias_numpy(net)

    NpuConv.BiasValue = bias
    NpuConv.WeightValue = weight

    # Pad参数
    OutputH = op.OutputShape[0].H
    OutputW = op.OutputShape[0].W
    InputH = op.InputShape[0].H
    InputW = op.InputShape[0].W
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    H = OutputH * StrideH - StrideH + KerH
    W = OutputW * StrideW - StrideW + KerW

    NpuConv.pad_top = op.PadH
    NpuConv.pad_bottom = H - op.PadH - InputH
    NpuConv.pad_left = op.PadW
    NpuConv.pad_right = W - op.PadW - InputW

    return NpuConv

