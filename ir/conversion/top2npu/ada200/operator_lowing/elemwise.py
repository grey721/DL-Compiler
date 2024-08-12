from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule, _register_op_transformation_rule
from ir.conversion.top2npu.top_lowing import *
from ir.dialect.npu.IR_operator import *


@_register_op_transformation_rule(OpTransformRule.ELEMWISE_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, ElemWise):
            # if op.ElwMode == ElementwiseMode.ELW_ADD:
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            print(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op, net):
    npu_elemwise = NpuElemWise()
    npu_elemwise.__dict__.update(op.__dict__)
    npu_elemwise.Name = "NpuElemWise"

    npu_elemwise.input_offset = op.get_input0_zero_point_numpy(net)
    npu_elemwise.input1_offset = op.get_input1_zero_point_numpy(net)
    npu_elemwise.output_offset = op.get_output_zero_point_numpy(net)

    input_scale = op.get_input0_scale_numpy(net)
    input1_scale = op.get_input1_scale_numpy(net)
    output_scale = op.get_output_scale_numpy(net)

    if npu_elemwise.do_relu:
        npu_elemwise.quantized_activation_max, npu_elemwise.quantized_activation_min \
            = get_activation_range(output_scale, npu_elemwise.output_offset, "RELU")
    else:
        npu_elemwise.quantized_activation_max, npu_elemwise.quantized_activation_min \
            = get_activation_range(output_scale, npu_elemwise.output_offset, None)

    if op.Mode == ElementWiseMode.ELW_ADD or op.Mode == ElementWiseMode.ELW_SUB:
        twice_input_scale = max(norm(input1_scale), norm(input_scale)) * 2
        real_input_scale = norm(input_scale) / twice_input_scale
        real_input1_scale = norm(input1_scale) / twice_input_scale
        real_output_scale = twice_input_scale / (norm(output_scale) * (1 << 20))
        npu_elemwise.left_shift = 20
        npu_elemwise.input_multiplier, npu_elemwise.input_shift \
            = QuantizeMultiplier(real_input_scale)
        npu_elemwise.input1_multiplier, npu_elemwise.input1_shift \
            = QuantizeMultiplier(real_input1_scale)
        npu_elemwise.output_multiplier, npu_elemwise.output_shift \
            = QuantizeMultiplier(real_output_scale)

    if op.Mode == ElementWiseMode.ELW_MUL:
        real_scale = norm(input_scale) * norm(input1_scale) / norm(output_scale)
        npu_elemwise.output_multiplier, npu_elemwise.output_shift \
            = QuantizeMultiplier(real_scale)

    if op.Mode == ElementWiseMode.ELW_DIV:
        real_scale = norm(input_scale) / (norm(input1_scale) * norm(output_scale))
        npu_elemwise.input_multiplier, npu_elemwise.input_shift \
            = QuantizeMultiplier(real_scale)

    return npu_elemwise


def _lowering_fp32(op, net):
    raise NotImplementedError
