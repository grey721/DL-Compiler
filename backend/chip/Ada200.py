from ir.dialect.npu.IR_operator import *
from backend.module.CIM import *
from ir.tool.utils import within_n_base_2
from ir.constant.type_mapping import Layout
from ir.graph.Graph_IR import *


class Ada200:
    num_cluster = 4
    num_cim_per_cluster = 4
    num_cim = 4 * 4
    DataLayout = Layout.NCHW

    def __init__(self):
        self.CIM = CIM()

    def node_partition(self, op):
        n_cim, times_load = self.CIM.get_usage(op)

        # Padding 以适应加法树
        n_cim = within_n_base_2(self.num_cim, n_cim)

        # 复用次数，若n_cim大于16，则获得最后一个block的n_cim复用次数
        replication = 0
        last = (n_cim * times_load) % 16
        if last != 0 and self.num_cim % last == 0:
            replication = int(self.num_cim / last) - 1

        return n_cim, times_load, replication

    # def get_replication_numbers(self, n_cim, times_load):
    #
    #         pass


if __name__ == "__main__":
    import pickle
    with open('output/yolov5s/npu_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    chip = Ada200()
    for layer, npu_op in enumerate(graph.AllOps):
        usage, times, n_re = chip.node_partition(npu_op)
        if usage:
            print(f"layer_{layer}:\n窗在hwc方向上需要的CIM数：{usage} \n需要加载权重的次数：{times}\n复用次数：{n_re}")
