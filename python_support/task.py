from python_support.cmodel import *
from python_support.memory import *
from python_support.util import *
from python_support.npu import *
from python_support.memory import *


class NPU_BASE_PARAM(Structure):
    _fields_ = [("weight_read_param", WEIGHT_READ_PARAM),
                ("npu_param", NPU_PARAM)]


class TASK_PARAM(Union):
    _fields_ = [("dma_param", DMA_PARAM),
                ("npu_base_param", NPU_BASE_PARAM),
                ("param_type", c_int)]


class TASK_LIST(Structure):
    _fields_ = [("task_param_list", POINTER(TASK_PARAM)),
                ("task_num", c_int)]


class TASK(Structure):
    _fields_ = [("task_list", TASK_LIST),
                ("memory", MEMORY),
                ("npu", NPU),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]


task_init = lib.task_init
task_init.argtypes = [POINTER(TASK), TASK_LIST]
task_init.restype = None

task_dma = lib.task_dma
task_dma.argtypes = [POINTER(TASK), DMA_PARAM]
task_dma.restype = None

task_run = lib.task_run
task_run.argtypes = [POINTER(TASK)]
task_run.restype = None

task_get_input = lib.task_get_input
task_get_input.argtypes = [POINTER(TASK), MEMORY_ADDRES_MAP]
task_get_input.restype = None

task_get_output = lib.task_get_output
task_get_output.argtypes = [POINTER(TASK), MEMORY_ADDRES_MAP]
task_get_output.restype = None

task_destory = lib.task_destory
task_destory.argtypes = [POINTER(TASK)]
task_destory.restype = None

task_interface = lib.task_interface
task_interface.argtypes = [TASK_LIST, MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP, MEMORY_ADDRES_MAP]
task_interface.restype = None
