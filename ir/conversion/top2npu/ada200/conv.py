from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.conversion.top2npu.top_lowing import *
from compiler.dialect.npu.ir.ir_operator import *


@_register_tranformation_rule(TransformRule.CONV_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, ConvBase):
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    NpuConv = NpuConv2d()
    NpuConv.__dict__.update(op.__dict__)
    NpuConv.Name = "NpuConv2d"

    weight_scale = op.GetQuantWeightScaleNumpy(net)
    Input_scale = op.GetQuantInputScaleNumpy(net)
    output_scale = op.GetQuantOutputScaleNumpy(net)

    scale = norm(weight_scale) * norm(Input_scale) / norm(output_scale)
    #bias + weight * input_zp
    weight = op.GetWeightNumpy(net)
    zp = op.GetQuantInputZeroPointNumpy(net)
    bias = op.GetBiasNumpy(net)

    for oc in range(op.OutputC):
        offset_sum = 0
        for ic in range(int(op.InputC / op.Group)):
            for kh in range(op.KerH):
                for kw in range(op.KerW):
                    offset_sum += weight[oc][kh][kw][ic] * zp[0]
        bias[oc] -= offset_sum
    NpuConv.BiasValue = bias
    NpuConv.WeightValue = NpuConv.GetWeightNumpy(net)

    NpuConv.input_offset = op.GetQuantInputZeroPointNumpy(net)
    NpuConv.weights_offset = op.GetQuantWeightScaleNumpy(net)
    NpuConv.output_offset = op.GetQuantOutputZeroPointNumpy(net)
    if len(scale) != op.OutputC:
        assert len(scale) == 1
        assert len(NpuConv.weights_offset) == 1
        scale = np.full((op.OutputC), scale[0], dtype=scale.dtype)
        NpuConv.weights_offset = np.full((op.OutputC), NpuConv.weights_offset[0], dtype=NpuConv.weights_offset.dtype)
    (NpuConv.output_multiplier, NpuConv.output_shift) = QuantizeMultiplier(scale)


    if NpuConv.do_relu == True:
        NpuConv.quantized_activation_max, NpuConv.quantized_activation_min \
            = get_activation_range(output_scale, NpuConv.output_offset, "RELU")

    OutputH = op.OutputH
    OutputW = op.OutputW
    InputH = op.InputH
    InputW = op.InputW
    KerH = op.KerH
    KerW = op.KerW
    StrideH = op.StrideH
    StrideW = op.StrideW
    H_ = OutputH*StrideH - StrideH + KerH
    W_ = OutputW*StrideW - StrideW + KerW
    NpuConv.pad_top = op.PadH
    NpuConv.pad_bottom = H_ - op.PadH - InputH
    NpuConv.pad_left = op.PadW
    NpuConv.pad_right = W_ - op.PadW - InputW
   
    return NpuConv


def _lowering_fp32(op, net):
    raise NotImplementedError
