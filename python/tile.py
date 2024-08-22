from python.cmodel import *
from python.memory import *
from python.util import *
from python.pad import *
from python.im2col import *
from python.data_reshape import *
from python.util import *
from python.cluster_cim import *
from python.cim_mac import *

class TILE_PARAM(Structure):
    _fields_ = [("tile_num", c_int),    
                ("conv_type", c_int),
                ("fmi_raddr", c_int),
                ("fmi_cmod", c_int),
                ("fmi_w", c_int),
                ("fmi_h", c_int),
                ("fmi_coffset", c_int),
                ("fmi_woffset", c_int),
                ("stride", c_int),
                ("kernel_m", c_int),
                ("kernel_cmod", c_int),
                ("kernel_w", c_int),
                ("kernel_h", c_int),
                ("pad_up", c_int),
                ("pad_down", c_int),
                ("pad_left", c_int),
                ("pad_right", c_int),
                ("cluster_win_num", c_int),
                ("cim_psum", c_int)]


    def param_to_txt(self, path):
        param_list = [  self.tile_num,
                        self.conv_type,
                        self.fmi_raddr,
                        self.fmi_cmod,
                        self.fmi_w,
                        self.fmi_h,
                        self.fmi_coffset,
                        self.fmi_woffset,
                        self.stride,
                        self.kernel_m,
                        self.kernel_cmod,
                        self.kernel_w,
                        self.kernel_h,
                        self.pad_up,
                        self.pad_down,
                        self.pad_left,
                        self.pad_right ]

        param_array = np.array(param_list).astype(np.int32)
        Array2Txt_hwc(param_array, 32, 32, path) 


    def _gen_fmi_data(self, seed):
        input_channel = np.int32(self.fmi_cmod * 64 / 8)
        input_h = self.fmi_h
        input_w = self.fmi_w
        np.random.seed(seed)
        fmi_data = (np.random.rand(input_h * input_w * input_channel)\
                        .reshape([input_h, input_w, input_channel]) * 255 - 128).astype(np.int8)
        self.fmi_data = ThreeD2OneD(fmi_data)


    def _fmi_data_to_txt(self, seed, path):
        self._gen_fmi_data(seed)
        Array2Txt_hwc(self.fmi_data, 8, 64, path)


    def fmi_data_to_txt(self, data, path):
        Array2Txt_hwc(data, 8, 64, path)  

class TILE(Structure):
    _fields_ = [("vpu_base_param_list", TILE_PARAM*4),
                ("quant_param", QUANT_PARAM),
                ("pad", PAD*4),
                ("im2col", IM2COL*4),
                ("data_reshape", DATA_RESHAPE*4),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]

tile_read = lib.tile_read
tile_read.argtypes = [POINTER(TILE_PARAM), MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
tile_read.restype = None


if __name__ == '__main__':

    ## hwcm: 1x1x1024x16 for 1-4
    tile_num = 1
    conv_type = 0
    fmi_raddr = 0
    fmi_cmod = int(1024 * 8 / 64)
    fmi_w = 64
    fmi_h = 64
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
    
    seed = 20221011
    tile_fmi_data_path = "/data/tinynpu/test/tile/tile_data.txt"
    tile_param._fmi_data_to_txt(seed, tile_fmi_data_path)

    tile_param_path = "/data/tinynpu/test/tile/tile_para.txt"
    tile_param.param_to_txt(tile_param_path)
    print("---------------------------------------")

