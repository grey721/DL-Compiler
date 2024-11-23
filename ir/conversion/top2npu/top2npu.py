from ir.conversion.top2npu.chip.lowing_ada200 import lowing_ada200_list, ADA300_TRANSFORM_MAP
from backend.chip.Ada300 import Ada300
from ir.graph.Graph_IR import GraphIR


class Top2Npu:

    def __init__(self, chip="ada300", mode=None):
        self.Chip = chip
        self.Mode = mode

    def transform(self, ir_graph: GraphIR):
        # 设置芯片支持的算子
        if self.Chip == "ada300":
            operation_list = lowing_ada200_list
            op_transform_map = ADA300_TRANSFORM_MAP
            ir_graph.DataLayout = Ada300.DataLayout
        else:
            raise NotImplementedError('Unknown chip')

        # 转换
        for operation in operation_list:
            op_transform_map[operation](ir_graph, self.Mode)
        return ir_graph
