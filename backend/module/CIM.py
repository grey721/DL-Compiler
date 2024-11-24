from ir.dialect.npu.IR_operator import NpuOp, NpuConv2d, NpuFullConnected
from ir.constant.type_mapping import *
import numpy as np
import math


class CIM:
    def __init__(self, h=64, w=16, subcell_num=4, data_type=DataType.FLOAT32):
        self.H = h
        self.W = w
        self.SubCell = subcell_num
        self.DataType = data_type

        self.Capacity = h * w * subcell_num
        self.H_equ = h * subcell_num  # 等效高度

    def map_weight_and_get_cim_usage(self, op):
        """
        Weight to col and padding
        return n_cim, times_load
        n_cim: cim num per window
        times_load: The number of times required to load weights for n_cim CIMs
        """

        k_n = op.WeightValue.shape[0]
        weight_map = op.WeightValue.reshape(k_n, -1).T

        n_cim = 0
        times_load = 0
        if isinstance(op, NpuConv2d):

            k_map_hwc, k_map_m = weight_map.shape

            # 该op在一个cycle内需要的CIM数
            n_cim = math.ceil(k_map_hwc / self.H_equ)
            # 权重需要加载的次数
            times_load = math.ceil(k_map_m/self.W)

            # 根据次数进行权重Padding
            n_k_n = times_load * self.W
            if n_k_n != k_n:
                bias = op.BiasValue

                _weight = np.zeros((k_map_hwc, n_k_n))  # .astype(np.int8)
                _weight[:, 0: k_map_m] = weight_map
                weight_map = _weight  # 给原weight填充0

                _bias = np.zeros(n_k_n)  # .astype(np.int32)
                _bias[0:k_n] = bias
                op.BiasValue = _bias

                # self.graph.AllTensors[conv_op.OutTensors[0]].Shape.reshape("C", n_k_n)

        elif isinstance(op, NpuFullConnected):
            pass

        op.WeightValue = weight_map

        return n_cim, times_load


if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    a = a.reshape((1, -1))
    print(a.shape)

