

class Top2Npu:

    Chip = "ada200"
    Mode = "int8"

    def __init__(self, chip=None, mode=None):
        if chip is not None:
            self.Chip = chip
        if mode is not None:
            self.Mode = mode

    def add_transform_option(self, transform_map):
        self.transform_map = transform_map
        for option in transform_map:
            print(option)

    def transform(self, ir_graph):
        if self.Chip == "ada200":
            self.add_transform_option(lowing_ada200_map)
            TRANSFORMMAP = ADA200_TRANSFORMMAP

        for option in self.transform_map:
            TRANSFORMMAP[option](ir_graph, self.Mode)
        return ir_graph
