from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule, _register_op_transformation_rule
from ir.conversion.top2npu.top_lowing import *
from ir.dialect.npu.IR_operator import *


@ _register_op_transformation_rule(OpTransformRule.FULLCONNECT_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, FullConnected):
            if mode == DataType.INT8:
                NpuOp = _lowering_int8(op, net)
            elif mode == DataType.FLOAT32:
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    npu_full_connect = NpuFullConnected()
    npu_full_connect.__dict__.update(op.__dict__)
    npu_full_connect.Name = "NpuFullConnected"

    weight_scale = op.get_weight_scale_numpy(net)
    input_scale = op.get_input_scale_numpy(net)
    output_scale = op.get_output_scale_numpy(net)
    npu_full_connect.output_offset = op.get_output_zero_point_numpy(net)

    if npu_full_connect.do_relu:
        npu_full_connect.quantized_activation_max, npu_full_connect.quantized_activation_min \
            = get_activation_range(output_scale, npu_full_connect.output_offset, "RELU")
    else:
        npu_full_connect.quantized_activation_max, npu_full_connect.quantized_activation_min \
            = get_activation_range(output_scale, npu_full_connect.output_offset, None)

    scale = norm(weight_scale) * norm(input_scale) / norm(output_scale)
    #bias + weight * input_zp
    weight = op.get_weight_numpy(net)
    zp = op.get_input_zero_point_numpy(net)
    bias = op.get_bias_numpy(net)
    for oc in range(op.OutputShape[0].C):
        offset_sum = 0
        for ic in range(int(op.InputShape[0].C)):
            offset_sum += weight[oc][ic] * zp[0]
        bias[oc] -= offset_sum
    npu_full_connect.BiasValue = bias
    npu_full_connect.WeightValue = npu_full_connect.get_weight_numpy(net)
    (npu_full_connect.output_multiplier, npu_full_connect.output_shift) = QuantizeMultiplier(scale)
    npu_full_connect.input_offset = op.get_input_zero_point_numpy(net)
    npu_full_connect.weights_offset = op.get_weight_scale_numpy(net)
    npu_full_connect.output_offset = op.get_output_zero_point_numpy(net)

    return npu_full_connect


def _lowering_fp32(op, net):
    raise NotImplementedError