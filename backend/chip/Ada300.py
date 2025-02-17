from backend.module.CIM import *
from ir.graph.Graph_IR import *
from ir.utils.NSGA import *


class Ada300:
    num_core = 16
    num_cluster = 4  # 未来将这部分改入CIM模块中
    num_cim_per_cluster = 4
    num_cim = num_cluster * num_cim_per_cluster
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
        mvm_op = []
        ag = 0
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
                npu_op = NpuOp()
                npu_op.fuse_ops(npu_op)
                del self.graph.AllOps[layer]
                self.graph.AllOps.insert(layer, npu_op)
            else:
                continue
            print(f"layer_{layer}:")
            mvm_op.append(layer)
            sub_block_list = self.CIM.generate_unit(op, self.num_cim)
            # self.graph.AllTensors[op.OutTensors[0]].Shape.reshape("C", n_k_n)
            npu_op.sub_block_list = sub_block_list
            ag += op.WeightValue.shape[0] * op.WeightValue.shape[1]
        self.graph.num_mvm_op_unit = ag
        self.graph.mvm_op_idx = mvm_op
        print(f"总计含矩阵乘的算子有：{len(mvm_op)}个")
        print(f"总计AG个数：{ag}")
        # NSGA(self.graph, mvm_op, Ada300)

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
