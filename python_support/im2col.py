from python_support.cmodel import *
from python_support.tile import *
from python_support.memory import BASE_DATA, MEMORY_ADDRES_MAP
from python_support.util import Array2Txt_hwc


class IM2COL_PARAM(Structure):
    _fields_ = [("kernel_h", c_int),
                ("kernel_w", c_int),
                ("stride", c_int),
                ("fmi_h", c_int),
                ("fmi_w", c_int),
                ("fmi_c", c_int),
                ("cluster_win_num", c_int),
                ("cim_psum", c_int),
                ("fmo_h", c_int),
                ("fmo_w", c_int),
                ("win_num", c_int),
                ("col_size", c_int),
                ("fmo_buffer_size", c_int),
                ("col_bit_m_size", c_int),
                ("win_patch_mod", c_int),
                ("win_patch_size", c_int),
                ("cim_patch_size", c_int),
                ("reuse_factor", c_int),
                ("wins_padding_buffer_size", c_int)]

    # def __init__(self, kernel_h, kernel_w, 
    #              stride, fmi_h, fmi_w, fmi_c):
    #     self.kernel_h = kernel_h
    #     self.kernel_w = kernel_w
    #     self.stride = stride
    #     self.fmi_h = fmi_h
    #     self.fmi_w = fmi_w
    #     self.fmi_c = fmi_c
    #     self.fmo_h = int((self.fmi_h - self.kernel_h) / self.stride) + 1
    #     self.fmo_w = int((self.fmi_w - self.kernel_w) / self.stride) + 1
    #     self.col_size = self.fmi_c * self.kernel_h * self.kernel_w
    #     self.fmo_buffer_size = self.fmo_h * self.fmo_w  * self.col_size


class IM2COL(Structure):
    _fields_ = [("im2col_param", IM2COL_PARAM),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]


im2col_param_init = lib.im2col_init
im2col_param_init.argtypes = [POINTER(IM2COL_PARAM), IM2COL_PARAM]
im2col_param_init.restype = None

im2col_interface = lib.im2col_interface
im2col_interface.argtypes = [IM2COL_PARAM, MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
im2col_interface.restype = None

if __name__ == "__main__":
    ## hwcm: 1x1x1024x64 for 1-4
    tile_num = 1
    conv_type = 0
    fmi_raddr = 0
    fmi_cmod = int(1024 * 8 / 64)
    fmi_w = 8
    fmi_h = 8
    fmi_coffset = 0
    fmi_woffset = 0
    stride = 1
    kernel_m = int(64 / 4 / 16)  ## meaning kernel_m for 1 cluster div 16
    kernel_cmod = int(1024 * 8 / 64)
    kernel_w = 1
    kernel_h = 1
    pad_up = 0
    pad_down = 0
    pad_left = 0
    pad_right = 0

    seed = 20221011

    tile_param = TILE_PARAM(tile_num,
                            conv_type,
                            fmi_raddr,
                            fmi_cmod,
                            fmi_w,
                            fmi_h,
                            fmi_coffset,
                            fmi_woffset,
                            stride,
                            kernel_m,
                            kernel_cmod,
                            kernel_w,
                            kernel_h,
                            pad_up,
                            pad_down,
                            pad_left,
                            pad_right)

    tile_fmi_data_path = "/data/tinynpu/test/im2col/tile_data.txt"
    tile_param._fmi_data_to_txt(seed, tile_fmi_data_path)

    tile_param_path = "/data/tinynpu/test/im2col/tile_para.txt"
    tile_param.param_to_txt(tile_param_path)

    fmi_data = tile_param.fmi_data

    fmi_c = int(fmi_cmod * 64 / 8)
    cluster_win_num = 1
    cim_psum = 4
    im2col_param = IM2COL_PARAM(kernel_h, kernel_w, stride,
                                fmi_h, fmi_w, fmi_c,
                                cluster_win_num, cim_psum)

    im2col_param_init(byref(im2col_param), im2col_param)

    py_values = fmi_data.tolist()
    buffer_size = len(py_values)
    input_data = (c_int8 * buffer_size)(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)

    buffer_size = im2col_param.wins_padding_buffer_size
    py_values = np.zeros(buffer_size).astype(np.int8).tolist()
    output_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(0, buffer_size, 0, 0, base_data)

    im2col_interface(im2col_param, input, output)

    im2col_data = np.array(output.get_data()).astype(np.int8)
    im2col_data_path = "/data/tinynpu/test/im2col/test_data/im2col_data.txt"
    Array2Txt_hwc(im2col_data, 8, 64, im2col_data_path)

    print("----------------------------------")
