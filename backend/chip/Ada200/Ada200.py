from ir.dialect.npu.IR_operator import *
from backend.module.CIM import *
from ir.graph.Graph_IR import *


class Ada200:
    num_cluster = 4
    num_cim_per_cluster = 4
    num_cim = 4 * 4

    def __init__(self):
        self.CIM = CIM()

    def node_partition(self, op):
        n_cim, times_load = self.CIM.get_usage(op)
        # Padding 以适应加法树
        # 1 << compensation.bit_length())是compensation最近的以2为底的数
        compensation = n_cim % self.num_cim
        if compensation:
            n_cim += (1 << compensation.bit_length()) - compensation

        return n_cim, times_load
