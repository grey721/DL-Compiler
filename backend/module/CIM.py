from ir.dialect.npu.IR_operator import NpuOp, NpuConv2d, NpuFullConnected
from ir.constant.type_mapping import *
import numpy as np


class CIM:
    def __init__(self, h=64, w=16, subcell_num=4, data_type=DataType.FLOAT32):
        self.H = h
        self.W = w
        self.SubCell = subcell_num
        self.DataType = data_type

        self.Capacity = h * w * subcell_num
        self.H_equ = h * subcell_num  # 等效高度

    def get_usage(self, op):
        """
        return n_cim, times_load
        n_cim: cim num per window
        times_load: The number of times required to load weights for n_cim CIMs
        """

        if isinstance(op, NpuOp):
            if op.NpuOpConv:
                op = op.NpuOpConvOp
            elif op.NpuOpFc:
                op = op.NpuOpFcOp

        n_cim = 0
        times_load = 0
        if isinstance(op, NpuConv2d):
            k_map_hwc = op.KerH * op.KerW * op.KerM
            k_map_m = op.KerM

            # 该op在一个cycle内需要的CIM数
            n_cim = np.ceil(k_map_hwc / self.H_equ)
            times_load = np.ceil(k_map_m/self.W)

        elif isinstance(op, NpuFullConnected):
            pass

        return n_cim, times_load
