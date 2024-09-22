from ir.conversion.top2npu.ada200.lowing_ada200 import lowing_ada200_list, ADA200_TRANSFORM_MAP


class Top2Npu:
    Chip = "ada200"

    def __init__(self, chip=None, mode=None):
        if chip is not None:
            self.Chip = chip
        self.Mode = mode

    def transform(self, ir_graph):
        # parse
        if self.Chip == "ada200":
            operation_list = lowing_ada200_list
            op_transform_map = ADA200_TRANSFORM_MAP
        else:
            raise NotImplementedError('Unknown chip')

        # 转换
        for operation in operation_list:
            op_transform_map[operation](ir_graph, self.Mode)
        return ir_graph
