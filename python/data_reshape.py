from python.cmodel import *
from python.memory import BASE_DATA, MEMORY_ADDRES_MAP
from python.util import Array2Txt_hwc


class DATA_RESHAPE_PARAM(Structure):
    _fields_ = [("cluster_cim_num", c_int),
                ("bit_num", c_int),
                ("cluster_win_num", c_int),
                ("cim_psum", c_int)]


data_reshape_interface = lib.data_reshape_interface
data_reshape_interface.argtypes = [DATA_RESHAPE_PARAM, MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
data_reshape_interface.restype = None


def intput_bit_data_save_to_txt(input_bit_data, data_path):

    input_group_len = int(len(input_bit_data) / 8)
    _input_bit_data = []
    for i in range(input_group_len):
        start_addr = i*8
        end_addr = (i+1)*8
        input_8_bit_data = input_bit_data[start_addr:end_addr]
        x = np.packbits(input_8_bit_data).astype(np.int8)[0]
        _input_bit_data.append(x)

    input_bit_data = np.array(_input_bit_data)
    Array2Txt_hwc(input_bit_data, 8, 64, data_path) 

class DATA_RESHAPE(Structure):
    _fields_ = [("data_reshape_param", DATA_RESHAPE_PARAM),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]

if __name__ == "__main__":

    cluster_cim_num = 4
    bit_num = 8
    cluster_win_num = 1
    cim_psum = 0
    data_reshape_param = DATA_RESHAPE_PARAM(cluster_cim_num, bit_num)

    input_list = [i for i in range(8)]
    py_values = input_list
    buffer_size = len(py_values)
    input_data = (c_int8 * buffer_size)(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)

    buffer_size = 8 * len(input_list)
    py_values = np.zeros(buffer_size).astype(np.int8).tolist()
    output_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)

    data_reshape_interface(data_reshape_param, input, output)
    input_bit_data = output.get_data()
    data_path = "/data/tinynpu/test/data_reshape/input_bit_data.txt"
    intput_bit_data_save_to_txt(input_bit_data, data_path)

    print("-------------------------------------------------")