from python_support.cmodel import *
from python_support.memory import *
from python_support.util import *
from python_support.tile import *
from python_support.vpu import *
from python_support.cluster_cim import *
from python_support.cim_mac import *


class NPU_PARAM(Structure):
    _fields_ = [("tile_compute_mode", c_int),
                ("tile_param", TILE_PARAM * 4),
                ("quant_param", QUANT_PARAM),
                ("cim_cluster_param", CIM_CLUSTER_PARAM),
                ("cim_cluster_acc_param", CIM_CLUSTER_ACC_PARAM),
                ("vpu_param", VPU_PARAM * 4)]


class NPU(Structure):
    _fields_ = [("npu_param", NPU_PARAM),
                ("tile", TILE),
                ("cim_cluster", CIM_CLUSTER),
                ("vpu", VPU * 4),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]
