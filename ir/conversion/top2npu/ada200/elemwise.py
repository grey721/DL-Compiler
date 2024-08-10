from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.conversion.top2npu.top_lowing import *
from compiler.dialect.npu.ir.ir_operator import *


@_register_tranformation_rule(TransformRule.ELEMWISE_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Elemwise):
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

    npu_elemwise = NpuElemwise()
    npu_elemwise.__dict__.update(op.__dict__)
    npu_elemwise.Name = "NpuElemwise"

    npu_elemwise.input_offset = op.GetQuantInput0ZeroPointNumpy(net)
    npu_elemwise.input1_offset = op.GetQuantInput1ZeroPointNumpy(net)
    npu_elemwise.output_offset = op.GetQuantOutputZeroPointNumpy(net)
    input_scale = op.GetQuantInput0ScaleNumpy(net)
    input1_scale = op.GetQuantInput1ScaleNumpy(net)
    output_scale = op.GetQuantOutputScaleNumpy(net)

    if npu_elemwise.do_relu == True:
        npu_elemwise.quantized_activation_max, npu_elemwise.quantized_activation_min \
            = get_activation_range(output_scale, npu_elemwise.output_offset, "RELU")
    else:
        npu_elemwise.quantized_activation_max, npu_elemwise.quantized_activation_min \
            = get_activation_range(output_scale, npu_elemwise.output_offset, None)

    if op.ElwMode == ElementwiseMode.ELW_ADD or op.ElwMode == ElementwiseMode.ELW_SUB:
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

    if op.ElwMode == ElementwiseMode.ELW_MUL:
        real_scale = norm(input_scale) * norm(input1_scale) / norm(output_scale)
        npu_elemwise.output_multiplier, npu_elemwise.output_shift \
                                = QuantizeMultiplier(real_scale)

    if op.ElwMode == ElementwiseMode.ELW_DIV:
        real_scale = norm(input_scale) / (norm(input1_scale) * norm(output_scale))
        npu_elemwise.input_multiplier, npu_elemwise.input_shift \
                                = QuantizeMultiplier(real_scale)
    
    return npu_elemwise


def _lowering_fp32(op, net):
    raise NotImplementedError