from python_support.cmodel import *
from python_support.memory import *
from python_support.cluster_cim import *


class QUANT_PARAM(Structure):
    _fields_ = [("input_offset", c_int),
                ("weights_offset", c_int),
                ("output_offset", c_int),
                ("output_multiplier", POINTER(c_int)),
                ("output_shift", POINTER(c_int)),
                ("quantized_activation_min", c_int),
                ("quantized_activation_max", c_int)]


class CIM_MAC_PARAM(Structure):
    _fields_ = [("cols", c_int),
                ("rows", c_int),
                ("sub_cell_num", c_int),
                ("sraw_row_width", c_int),
                ("sraw_col_width", c_int)]


class CIM_MAC_ACC_PARAM(Structure):
    _fields_ = [("bit_map", c_int),
                ("acc_flag", c_int)]


class CIM_MAC(Structure):
    _fields_ = [("cim_mac_param", CIM_MAC_PARAM),
                ("quant_param", QUANT_PARAM),
                ("cim_mac_acc_param", CIM_MAC_ACC_PARAM),
                ("cim_cluster_acc_param", CIM_MAC_ACC_PARAM),
                ("weight", MEMORY_ADDRES_MAP),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]
