from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.conversion.top2npu.top_lowing import *
from compiler.dialect.npu.ir.ir_operator import *


@_register_tranformation_rule(TransformRule.FULLCONNECT_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, FullConnected):
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    npu_full_connect = NpuFullConnected()
    npu_full_connect.__dict__.update(op.__dict__)
    npu_full_connect.Name = "NpuFullConnected"

    weight_scale = op.GetQuantWeightScaleNumpy(net)
    Input_scale = op.GetQuantInputScaleNumpy(net)
    output_scale = op.GetQuantOutputScaleNumpy(net)
    npu_full_connect.output_offset = op.GetQuantOutputZeroPointNumpy(net)

    if npu_full_connect.do_relu == True:
        npu_full_connect.quantized_activation_max, npu_full_connect.quantized_activation_min \
            = get_activation_range(output_scale, npu_full_connect.output_offset, "RELU")
    else:
        npu_full_connect.quantized_activation_max, npu_full_connect.quantized_activation_min \
            = get_activation_range(output_scale, npu_full_connect.output_offset, None)

    scale = norm(weight_scale) * norm(Input_scale) / norm(output_scale)
    #bias + weight * input_zp
    weight = op.GetWeightNumpy(net)
    zp = op.GetQuantInputZeroPointNumpy(net)
    bias = op.GetBiasNumpy(net)
    for oc in range(op.OutputC):
        offset_sum = 0
        for ic in range(int(op.InputC)):
            offset_sum += weight[oc][ic] * zp[0]
        bias[oc] -= offset_sum
    npu_full_connect.BiasValue = bias
    npu_full_connect.WeightValue = npu_full_connect.GetWeightNumpy(net)
    (npu_full_connect.output_multiplier, npu_full_connect.output_shift) = QuantizeMultiplier(scale)
    npu_full_connect.input_offset = op.GetQuantInputZeroPointNumpy(net)
    npu_full_connect.weights_offset = op.GetQuantWeightScaleNumpy(net)
    npu_full_connect.output_offset = op.GetQuantOutputZeroPointNumpy(net)

    return npu_full_connect


def _lowering_fp32(op, net):
    raise NotImplementedError