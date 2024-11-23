import numpy as np

from backend.module.CIM import *
from ir.graph.Graph_IR import *
from ir.tool.utils import within_n_base_2


class Ada300:
    num_cluster = 4
    num_cim_per_cluster = 4
    num_cim = 4 * 4
    DataLayout = Layout.NCHW
    CIM = CIM(
        h=64,
        w=16,
        subcell_num=4,
        data_type=DataType.FLOAT32
    )

    def __init__(self, npu_graph):
        self.graph = npu_graph

        self.node_partition()
        # self.get_replication_numbers
        self.memory_assign()

    def node_partition(self):
        print("----start TransformRule NODE_PARTITION and WEIGHT_PADDING-----")
        for layer, npu_op in enumerate(self.graph.AllOps):
            if isinstance(npu_op, NpuOp):
                if npu_op.NpuOpConv:
                    op = npu_op.NpuOpConvOp
                elif npu_op.NpuOpFc:
                    op = npu_op.NpuOpFcOp
                else:
                    continue
            elif isinstance(npu_op, (NpuConv2d, NpuFullConnected)):
                op = npu_op
            else:
                continue

            # 若算子有权重，计算hwc方向上所需的CIM个数和权重的加载次数，并且对权重Pad
            n_cim, times_load = self.CIM.get_usage_map(op)

            # 适应加法树
            n_cim = within_n_base_2(self.num_cim, n_cim)

            # 复用次数，若n_cim大于16，则获得最后一个block的n_cim复用次数
            replication = 0
            last = (n_cim * times_load) % 16
            if last != 0 and self.num_cim % last == 0:
                replication = int(self.num_cim / last)

            self.gen_block_param(op, n_cim, times_load, replication)

            print(f"layer_{layer}:\n"
                  f"    窗在hwc方向上需要的CIM数：{n_cim} \n"
                  f"    需要加载权重的次数：{times_load}\n"
                  f"    最后次加载中可复用次数：{replication-1}\n")

    def gen_block_param(self, op, n_cim, times_load, repeat):
        weight = op.WeightValue
        sub_block = np.array([np.array_split(i, n_cim, axis=0) for i in np.array_split(weight, times_load, axis=1)])
        sub_bias = np.array(np.array_split(op.BiasValue, times_load))

        if repeat and len(sub_block) < self.num_cim:
            sub_block = np.tile(sub_block, (repeat, 1))
            sub_bias = np.tile(sub_bias, (repeat, 1))

        op.WeightValue = sub_block
        op.BiasValue = sub_bias

    def get_replication_numbers(self, n_cim, times_load):
        # NSGA3
        pass

    def memory_assign(self):
        pass


if __name__ == "__main__":
    import pickle

    with open('output/yolov5s/npu_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    chip = Ada300(graph)

    for t in graph.AllTensors:
        print(t.DataType)
