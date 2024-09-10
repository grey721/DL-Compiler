from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule, _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *

# from python_support.vpu import *
# from python_support.memory import *
# from python_support.util import *


@_register_op_transformation_rule(OpTransformRule.LOGISTIC_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Activation):
            if op.Mode != ActivationMode.SIGMOID:
                continue
            if mode is None:
                NpuOp = _lowering_none(op)
            elif mode == "int8":
                NpuOp = _lowering_int8(op, net)
            elif mode == "fp32":
                NpuOp = _lowering_fp32(op, net)
            else:
                raise NotImplementedError('Unsupported lowing mode')

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


def sigmoid_int8(x, y, input_shape, shape_len, input_offset, input_scale):
    param = SIGMOID_PARAM(input_offset, input_scale)
    base_param = BASE_PARAM()
    base_param.sigmoid_param = param
    vpu_type = 3
    vpu_base_param = VPU_BASE_PARAM(base_param, vpu_type)
    py_values = [vpu_base_param]
    vpu_base_param_list = (VPU_BASE_PARAM * len(py_values))(*py_values)
    vpu_mode = 0
    vpu_base_param_list_len = 1
    vpu_parm = VPU_PARAM(vpu_base_param_list, vpu_base_param_list_len, vpu_mode)

    x = ThreeD2OneD(x).tolist()
    buffer_size = len(x)
    buffer_addr = 0
    memory_type = 0
    data_type = 0  # mean uint8
    input_data = (c_int8 * buffer_size)(*x)
    base_data = BASE_DATA()
    base_data.i8_data = input_data
    input = MEMORY_ADDRES_MAP(buffer_addr, buffer_size,
                              memory_type, data_type, base_data)

    buffer_size = len(y)
    buffer_addr = 0
    memory_type = 0
    data_type = 0  # mean uint8
    output_data = (c_int8 * len(y))(*y)
    base_data = BASE_DATA()
    base_data.i8_data = output_data

    output = MEMORY_ADDRES_MAP(buffer_addr, buffer_size,
                               memory_type, data_type, base_data)

    vpu_interface(vpu_parm, input, output, 1)
    result = output.get_data()
    assert (shape_len == 3), 'shape error'
    h, w, c = input_shape
    result = OneD2ThreeD(result, h, w, c)

    return result


def activation_lut_gen(cmodel, npu_sigmoid, s_in, base_addr="32'h00009", base_addr_int=112):
    input_data_0 = (np.array([i for i in range(0, 128)]).reshape([1, 1, 128])).astype(np.int8)
    input_data_1 = (np.array([i for i in range(-128, 0)]).reshape([1, 1, 128])).astype(np.int8)
    input_data = np.concatenate((input_data_0, input_data_1), axis=2)
    y = cmodel(input_data, (1, 1, 256), npu_sigmoid, s_in)

    return y


def cmodel(x, input_shape, npu_sigmoid, s_in):
    y = np.zeros([input_shape[0] * input_shape[1] * input_shape[2]], dtype=np.int8)
    input_shape = np.array(input_shape, dtype=np.int32)
    shape_len = np.int32(len(input_shape))
    input_offset = npu_sigmoid.input_offset[0]
    input_scale = s_in
    y = sigmoid_int8(x, y, input_shape, shape_len, input_offset, input_scale)
    return y


def _lowering_int8(op, net):
    npu_logistic = NpuLogistic()
    npu_logistic.__dict__.update(op.__dict__)
    npu_logistic.Type = "NpuLogistic"

    input_scale = op.get_input_scale_numpy(net)
    output_scale = op.get_output_scale_numpy(net)

    input_zero_point = op.get_input_zero_point_numpy(net)
    output_zero_point = op.get_output_zero_point_numpy(net)

    npu_logistic.input_offset = input_zero_point
    npu_logistic.output_offset = output_zero_point

    npu_logistic.output_multiplier, npu_logistic.output_shift = _get_quant_para(input_scale, output_scale)
    # todo test logistic in test project?还需要加什么
    npu_logistic.lut_dict = activation_lut_gen(cmodel, npu_logistic, input_scale, base_addr="32'h00009", base_addr_int=112)
    raise NotImplementedError


def _lowering_fp32(op, net):
    raise NotImplementedError


def _lowering_none(op):
    npu_sigmoid = NpuLogistic()
    npu_sigmoid.__dict__.update(op.__dict__)
    npu_sigmoid.Type = "NpuSigmoid"
    return npu_sigmoid
