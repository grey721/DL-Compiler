from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule, _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *

from python.vpu import *
from python.memory import *
from python.util import *


@_register_op_transformation_rule(OpTransformRule.LEAKYRELU_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Activation):
            if op.Name == "leakyrelu":
                if mode == "int8":
                    NpuOp = _lowering_int8(op, net)
                if mode == "fp32":
                    NpuOp = _lowering_fp32(op, net)

                op_id = net.get_op_idx(op)
                net.delete_op(op_id)
                net.insert_op(NpuOp, op_id)


def _get_quant_para(input_scale, output_scale):
    def _norm(x):
        return np.array(x, dtype=np.float64)

    def QuantizeMultiplierSmallerThanOneExp(x):  # 量化乘法器小于1
        assert x > 0., "real_multiplier err"
        q, shift = np.frexp(x)
        q_fixed = np.round(q * (1 << 31)).astype(np.int32)

        if (q_fixed == (1 << 31)).sum() > 0:
            shift[q_fixed == (1 << 31)] += 1
            q_fixed[q_fixed == (1 << 31)] /= 2
        if (shift < -31).sum() > 0:
            q_fixed[shift < -31] = 0
            shift[shift < -31] = 0
        assert shift <= 0, "shift err"
        return q_fixed, shift

    def QuantizeMultiplierGreateThanOne(x):  # 量化乘法器大于1
        q, shift = np.frexp(x)
        q_fixed = np.round(q * (1 << 31)).astype(np.int32)

        if (q_fixed == (1 << 31)).sum() > 0:
            shift[q_fixed == (1 << 31)] += 1
            q_fixed[q_fixed == (1 << 31)] /= 2
        if (shift < -31).sum() > 0:
            q_fixed[shift < -31] = 0
            shift[shift < -31] = 0
        assert shift >= 0, "shift err"
        return q_fixed, shift

    real_multiplier = _norm(input_scale) / _norm(output_scale)
    if real_multiplier < 1.0:
        output_multiplier, output_shift = QuantizeMultiplierSmallerThanOneExp(real_multiplier)
    else:
        output_multiplier, output_shift = QuantizeMultiplierGreateThanOne(real_multiplier)

    return output_multiplier, output_shift


def leaky_relu_int8(x, y, input_shape, shape_len, input_offset, output_offset, output_multiplier_identity,
                    output_multiplier_alpha, output_shift_identity, output_shift_alpha, quantized_activation_min,
                    quantized_activation_max):
    param = LEAKY_RELU_PARAM(input_offset, output_offset, output_multiplier_identity,
                             output_multiplier_alpha, output_shift_identity, output_shift_alpha,
                             quantized_activation_min, quantized_activation_max)  # 结构体
    a = np.array([input_offset, output_offset, output_multiplier_identity,
                  output_multiplier_alpha, output_shift_identity, output_shift_alpha,
                  quantized_activation_min, quantized_activation_max])

    base_param = BASE_PARAM()
    base_param.leaky_relu_param = param
    vpu_type = 1
    vpu_base_param = VPU_BASE_PARAM(base_param, vpu_type)
    py_values = [vpu_base_param]
    vpu_base_param_list = (VPU_BASE_PARAM * len(py_values))(*py_values)
    vpu_mode = 0
    vpu_base_param_list_len = 1
    vpu_parm = VPU_PARAM(vpu_base_param_list, vpu_base_param_list_len, vpu_mode)

    x = ThreeD2OneD(x).tolist()  # x=input data;
    buffer_size = len(x)
    buffer_addr = 0
    memory_type = 0
    data_type = 0  # mean int8
    input_data = (c_int8 * buffer_size)(*x)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(buffer_addr, buffer_size,
                              memory_type, data_type, base_data)

    buffer_size = len(y)
    buffer_addr = 0
    memory_type = 0
    data_type = 0  # mean int8
    output_data = (c_int8 * len(y))(*y)
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(buffer_addr, buffer_size,
                               memory_type, data_type, base_data)

    vpu_interface(vpu_parm, input, output, 1)
    result = output.get_data()
    assert (shape_len == 3), 'shape error'
    h, w, c = input_shape
    result = OneD2ThreeD(result, h, w, c)

    return result


def activation_lut_gen(cmodel, npu_leakyrelu, base_addr="32'h00009", base_addr_int=112):
    input_data_0 = (np.array([i for i in range(0, 128)]).reshape([1, 1, 128])).astype(np.int8)
    input_data_1 = (np.array([i for i in range(-128, 0)]).reshape([1, 1, 128])).astype(np.int8)
    input_data = np.concatenate((input_data_0, input_data_1), axis=2)
    y = cmodel(input_data, (1, 1, 256), npu_leakyrelu)

    return y


def cmodel(x, input_shape, npu_leakyrelu):  # x=input data;
    y = np.zeros([input_shape[0] * input_shape[1] * input_shape[2]], dtype=np.int8)
    input_shape = np.array(input_shape, dtype=np.int32)
    shape_len = np.int32(len(input_shape))

    input_offset = npu_leakyrelu.input_offset[0]
    output_offset = npu_leakyrelu.output_offset[0]
    identity_output_multiplier = npu_leakyrelu.identity_output_multiplier[0]
    alpha_output_multiplier = npu_leakyrelu.alpha_output_multiplier[0][0]
    identity_output_shift = npu_leakyrelu.identity_output_shift[0]
    alpha_output_shift = npu_leakyrelu.alpha_output_shift[0][0]
    quantized_activation_min = -128
    quantized_activation_max = 127

    y = leaky_relu_int8(x, y, input_shape, shape_len, input_offset, output_offset, identity_output_multiplier,
                        alpha_output_multiplier, identity_output_shift, alpha_output_shift, quantized_activation_min,
                        quantized_activation_max)

    return y


def _lowering_int8(op, net):
    npu_leakyrelu = NpuLeakyRelu()
    npu_leakyrelu.__dict__.update(op.__dict__)
    npu_leakyrelu.Type = "NpuLeakyRelu"

    npu_leakyrelu.input_offset = op.get_input_zero_point_numpy(net)
    npu_leakyrelu.output_offset = op.get_output_zero_point_numpy(net)

    input_scale = op.get_input_scale_numpy(net)
    output_scale = op.get_output_scale_numpy(net)
    alpha_input_scale = [input_scale * op.Alpha]  # alpha * S_in

    npu_leakyrelu.identity_output_multiplier, npu_leakyrelu.identity_output_shift \
        = _get_quant_para(input_scale, output_scale)
    npu_leakyrelu.alpha_output_multiplier, npu_leakyrelu.alpha_output_shift \
        = _get_quant_para(alpha_input_scale, output_scale)

    npu_leakyrelu.lut_dict = activation_lut_gen(cmodel, npu_leakyrelu, base_addr="32'h00009", base_addr_int=112)

    return npu_leakyrelu


def _lowering_fp32(op, net):
    raise NotImplementedError
