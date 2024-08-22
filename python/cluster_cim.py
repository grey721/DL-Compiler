from python.cmodel import *
from python.memory  import *
from python.cim_mac  import *


class CIM_CLUSTER_PARAM(Structure):
    _fields_ = [("cluster_num", c_int),
                ("cluster_cim_num", c_int)]


class CIM_CLUSTER_ACC_PARAM(Structure):
    _fields_ = [("bit_map", c_int),
                ("cluster_p_sum", c_int),
                ("cluser_cim_p_sum", c_int)]



class CIM_CLUSTER(Structure):
    _fields_ = [("cim_cluster_param", CIM_CLUSTER_PARAM),
                ("quant_param", QUANT_PARAM),
                ("cim_mac", POINTER(CIM_MAC)),
                ("cim_cluster_acc_param", CIM_CLUSTER_ACC_PARAM),
                ("weight", MEMORY_ADDRES_MAP),
                ("bias", MEMORY_ADDRES_MAP),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]



cluster_init = lib.cluster_init_interface
cluster_init.argtypes = [CIM_CLUSTER_PARAM, CIM_CLUSTER_ACC_PARAM]
cluster_init.restype = None

write_weight_interface = lib.write_weight_interface
write_weight_interface.argtypes = [MEMORY_ADDRES_MAP]
write_weight_interface.restype = None

cluster_interface = lib.cluster_run_interface
cluster_interface.argtypes = [MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
cluster_interface.restype = None


if __name__ == "__main__":

    #init input
    py_values = [1 for _ in range(64*8*4*4*4)]
    buffer_size = len(py_values)
    input_data = (c_int8 * buffer_size)(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)

    #init weight
    py_values = [1 for _ in range(64 * 128 * 4 * 4 * 4)]
    buffer_size = len(py_values)
    weight_data = (c_int8 * buffer_size)(*py_values)
    base_data = BASE_DATA(weight_data)
    weight = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)
    
    #init bias
    buffer_size = 16
    py_values = [_+10 for _ in range(16)]
    bias_data = (c_int32 * len(py_values))(*py_values)
    base_data = BASE_DATA(None, None, bias_data)
    bias = MEMORY_ADDRES_MAP(0, buffer_size, 0, 2, base_data)
    #output
    buffer_size = 16
    py_values = [0 for _ in range(16)]
    output_data = (c_int32 * len(py_values))(*py_values)
    base_data = BASE_DATA(None, None, output_data)
    output = MEMORY_ADDRES_MAP(0, buffer_size, 0, 2, base_data)

    cim_cluster_para = CIM_CLUSTER_PARAM(4, 4)
    cim_cluster_acc_param = CIM_CLUSTER_ACC_PARAM(4, 4, 4)

    #quant_param
    output_multiplier = [1 for _ in range(16)]
    output_shift = [1 for _ in range(16)]
    multiplier_data = (c_int * len(output_multiplier))(*output_multiplier)
    shift_data= (c_int * len(output_shift))(*output_shift)
    quant_param = QUANT_PARAM(1, 1, 1, multiplier_data, shift_data, -128, 127)
    cluster_init(cim_cluster_para, cim_cluster_acc_param, quant_param)

    write_weight_interface(weight, bias)

    cluster_interface(input,output)
    data = np.array(output.get_data()).astype(np.int32)
    print(data)

