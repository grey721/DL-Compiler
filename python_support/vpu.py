from python_support.cmodel import *
from python_support.memory import BASE_DATA, MEMORY_ADDRES_MAP


class RELUX_PARAM(Structure):
    _fields_ = [("input_offset", c_int8),
                ("output_offset", c_int8),
                ("output_multiplier", c_int32),
                ("output_shift", c_int8),
                ("quantized_activation_min", c_int8),
                ("quantized_activation_max", c_int8)]


class LEAKY_RELU_PARAM(Structure):
    _fields_ = [("input_offset", c_int8),
                ("output_offset", c_int8),
                ("output_multiplier_identity", c_int32),
                ("output_multiplier_alpha", c_int32),
                ("output_shift_identity", c_int8),
                ("output_shift_alpha", c_int8),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class HARD_SWISH_PARAM(Structure):
    _fields_ = [("input_offset", c_int8),
                ("output_offset", c_int8),
                ("reluish_multiplier_fixedpoint_int16", c_int16),
                ("output_multiplier_fixedpoint_int16", c_int16),
                ("reluish_multiplier_exponent", c_int8),
                ("output_multiplier_exponent", c_int16),
                ("quantized_activation_min", c_int16),
                ("quantized_activation_max", c_int16)]


table = c_uint8 * 256


class SIGMOID_PARAM(Structure):
    _fields_ = [("input_offset", c_int32),
                ("input_scale", c_float),
                ("reluish_multiplier_fixedpoint_int16", table)]


class SILU_PARAM(Structure):
    _fields_ = [("input_offset", c_int32),
                ("logistic_output_offset", c_int32),
                ("logistic_output_multiplier", c_int32),
                ("logistic_input_scale", c_float),
                ("mul_output_offset", c_int32),
                ("mul_output_multiplier", c_int32),
                ("mul_output_shift", c_int),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class ADD_PARAM(Structure):
    _fields_ = [("input1_offset", c_int32),
                ("input2_offset", c_int32),
                ("output_offset", c_int32),
                ("output_multiplier", c_int32),
                ("output_shift", c_int),
                ("left_shift", c_int),
                ("input1_multiplier", c_int32),
                ("input1_shift", c_int),
                ("input2_multiplier", c_int32),
                ("input2_shift", c_int),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class SUB_PARAM(Structure):
    _fields_ = [("input1_offset", c_int32),
                ("input2_offset", c_int32),
                ("output_offset", c_int32),
                ("output_multiplier", c_int32),
                ("output_shift", c_int),
                ("left_shift", c_int),
                ("input1_multiplier", c_int32),
                ("input1_shift", c_int),
                ("input2_multiplier", c_int32),
                ("input2_shift", c_int),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class DIV_PARAM(Structure):
    _fields_ = [("input1_offset", c_int32),
                ("input2_offset", c_int32),
                ("output_offset", c_int32),
                ("output_multiplier", c_int32),
                ("output_shift", c_int),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class MUL_PARAM(Structure):
    _fields_ = [("input1_offset", c_int32),
                ("input2_offset", c_int32),
                ("output_offset", c_int32),
                ("output_multiplier", c_int32),
                ("output_shift", c_int),
                ("quantized_activation_min", c_int32),
                ("quantized_activation_max", c_int32)]


class RESIZE_PARAM(Structure):
    _fields_ = [("cim_mode_r", c_int8),
                ("resize_position_flag_r", c_int8),
                ("resize_bil_nn_sel_flag_r", c_int8),
                ("resize_param_half_pixal_flag_r", c_int8),
                ("resize_param_width_ratio_r", c_int16),
                ("resize_param_height_ratio_r", c_int16),
                ("resize_param_input_width_r", c_int16),
                ("resize_param_input_height_r", c_int16),
                ("resize_param_output_width_r", c_int16),
                ("resize_param_output_height_r", c_int16)]


class POOL_PARAM(Structure):
    _fields_ = [("cim_mode_r", c_int8),
                ("pooling_func_mode_r", c_int8),
                ("pooling_factor_r", c_int8),
                ("pooling_param_input_width_r", c_int16),
                ("pooling_param_input_height_r", c_int16),
                ("pooling_param_output_width_r", c_int16),
                ("pooling_param_output_height_r", c_int16),
                ("pooling_filter_width_r", c_int16),
                ("pooling_filter_height_r", c_int16),
                ("pooling_stride_width_r", c_int16),
                ("pooling_stride_height_r", c_int16),
                ("pooling_padding_values_width_r", c_int16),
                ("pooling_padding_values_height_r", c_int16), ]


class MEAN_PARAM(Structure):
    _fields_ = [("input_offset", c_int32),
                ("output_offset", c_int32),
                ("multiplier", c_int32),
                ("shift", c_int32),
                ("input_width_r", c_int16),
                ("input_height_r", c_int16),
                ("output_width_r", c_int16),
                ("output_height_r", c_int16),
                ("cim_mode_r", c_int8)]


class CONCATENATION_PARAM(Structure):
    _fields_ = [("axis", c_int32),
                ("inputs_count", c_int32),
                ("input_shapes", POINTER(c_int32)),
                ("output_shape", POINTER(c_int32))]


class BASE_PARAM(Union):
    _fields_ = [("div_param", DIV_PARAM),
                ("add_param", ADD_PARAM),
                ("relux_param", RELUX_PARAM),
                ("leaky_relu_param", LEAKY_RELU_PARAM),
                ("hard_swish_param", HARD_SWISH_PARAM),
                ("sigmoid_param", SIGMOID_PARAM),
                ("resize_param", RESIZE_PARAM),
                ("mean_param", MEAN_PARAM),
                ("pool_param", POOL_PARAM),
                ("sub_param", SUB_PARAM),
                ("mul_param", MUL_PARAM),
                ("concatenation_param", CONCATENATION_PARAM),
                ("silu_param", SILU_PARAM)]


class VPU_BASE_PARAM(Structure):
    _fields_ = [("param", BASE_PARAM),
                ("vpu_type", c_int)]


class VPU_PARAM(Structure):
    _fields_ = [("vpu_base_param_list", POINTER(VPU_BASE_PARAM)),
                ("vpu_base_param_list_len", c_int),
                ("vpu_mode", c_int)]


class VPU(Structure):
    _fields_ = [("vpu_param", VPU_PARAM),
                ("input", POINTER(MEMORY_ADDRES_MAP)),
                ("output", MEMORY_ADDRES_MAP)]


vpu_interface = lib.vpu_interface
vpu_interface.argtypes = [VPU_PARAM, POINTER(MEMORY_ADDRES_MAP), MEMORY_ADDRES_MAP, c_int]
vpu_interface.restype = None

if __name__ == "__main__":
    param = RELUX_PARAM(1, 1, 1, 1, 0, 0)
    base_param = BASE_PARAM(param)
    vpu_base_param = VPU_BASE_PARAM(base_param, 0)
    py_values = [vpu_base_param]
    vpu_base_param_list = (VPU_BASE_PARAM * len(py_values))(*py_values)
    vpu_parm = VPU_PARAM(vpu_base_param_list, 1, 0)

    py_values = [100]
    input_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, 1, 0, 0, base_data)

    py_values = [1]
    output_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(0, 1, 0, 0, base_data)

    vpu_interface(vpu_parm, input, output)
    print("-----------------------------")
