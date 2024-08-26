from python_support.cmodel import *
from python_support.memory import BASE_DATA, MEMORY_ADDRES_MAP


class PAD_PARAM(Structure):
    _fields_ = [("fmi_h", c_int),
                ("fmi_w", c_int),
                ("fmi_cmod", c_int),
                ("pad_value", c_int),
                ("pad_up", c_int),
                ("pad_down", c_int),
                ("pad_left", c_int),
                ("pad_right", c_int)]


class PAD(Structure):
    _fields_ = [("pad_param", PAD_PARAM),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]


padding_interface = lib.padding_interface
padding_interface.argtypes = [PAD_PARAM, MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
padding_interface.restype = None

if __name__ == "__main__":
    param = PAD_PARAM(3, 3, 3, 0, 1, 1, 1, 1)

    import random

    py_values = random.sample(range(100), 27)
    input_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, 27, 0, 0, base_data)

    py_values = [0 for i in range(75)]
    output_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(0, 75, 0, 0, base_data)
    padding_interface(param, input, output)

    print("----------------------------------")
