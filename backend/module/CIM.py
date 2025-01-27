from ir.dialect.npu.IR_operator import *
from ir.utils.constant.type_mapping import *
from ir.utils.utils import *
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

    def generate_unit(self, op, cim_total):
        # 若算子有权重，计算hwc方向上所需的CIM个数和权重的加载次数，并且对权重Pad，weight2col
        n_cim, times_load = self.padding_and_get_cim_usage(op)

        # 适应加法树
        n_cim = within_n_base_2(cim_total, n_cim)

        n_block = n_cim * times_load

        # 复用次数，若n_cim大于16，则获得最后一个block的n_cim复用次数
        repeat = 1
        last = n_block % 16
        if last != 0 and cim_total % last == 0:
            repeat = cim_total // last

        # 每次加载中，加载n_cim的次数
        times_per_load = cim_total // n_cim
        if times_per_load > times_load:
            times_per_load = times_load

        # 权重分割
        # weight[加载次数/需要的核心数][当次加载所需要的CIM数量][CIM中使用的H][CIM中使用的W]
        # bias[加载次数][当次所需要的偏置]
        sub_weight = np.array(
            [np.array_split(i, n_cim) for i in np.array_split(op.WeightValue, times_load, axis=1)])
        sub_bias = np.array(np.array_split(op.BiasValue, times_load))

        op.WeightValue = sub_weight
        op.BiasValue = sub_bias

        # 输出Reshape
        w_shape = sub_weight.shape
        if n_block > cim_total:
            sub_weight = sub_weight.reshape((-1, cim_total, w_shape[2], w_shape[3]))
            sub_bias = sub_bias.reshape((-1, times_per_load, self.W))
        else:
            sub_weight = sub_weight.reshape((1, -1, w_shape[2], w_shape[3]))
            sub_bias = sub_bias.reshape((1, -1, self.W))

        # op.times_per_load = times_per_load
        tree_flag = round(math.log2(n_cim))

        print(f"    Kernel Shape：{op.InputShape[1]}\n"
              f"    Output Shape：{op.OutputShape[0]}\n"
              f"    窗在hwc方向上需要的CIM数：{n_cim} \n"
              f"    需要加载权重的次数：{times_load}\n"
              f"    最后次加载中可复用次数：{repeat}\n"
              f"    SubBlock形状：{op.WeightValue.shape}\n"
              f"    -SubBias形状：{op.BiasValue.shape}\n"
              )
        sub_block_list = []
        for n, block in enumerate(sub_weight):
            sub_block = SubBlock()
            sub_block.BlockId = (op.TopOpId, n)
            sub_block.tree_flag = tree_flag

            sub_block.WeightValue = block
            if op.Bias:
                sub_block.Bias = True
                sub_block.BiasValue = sub_bias[n]

            sub_block_list.append(sub_block)

        sub_block_list[-1].repeat = repeat
        return sub_block_list

    def padding_and_get_cim_usage(self, op):
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
            times_load = math.ceil(k_map_m / self.W)

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

                op.OutputShape[0].reshape("C", n_k_n)

        elif isinstance(op, NpuFullConnected):
            pass

        op.WeightValue = weight_map

        return n_cim, times_load


if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    a = a.reshape((1, -1))
    print(a.shape)

