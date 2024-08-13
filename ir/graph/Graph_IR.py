from ir.dialect.top.IR_tensor import *


class GraphIR:

    def __init__(self) -> None:
        # 网络输入输出
        self.NetInTensors = []  # List[IRTensor]
        self.NetOutTensors = []  #
        # 算子、中间张量
        self.AllTensorIds = []  # Name:tensor_idx
        self.AllTensors = []  # List[IRTensor]

        self.AllOpIds = []
        self.AllOps = []  # List[OpBase]

        self.AllCpuOps = []
        self.NetOutOpId = []
        self.NetInputOpId = []
        # 参数
        # self.WeightTensorNames =    []
        # self.WeightTensors =        []
        # Npu
        self.NetOutNpuOpId = []
        self.NetInputNpuOpId = []
        self.SubGraphs = []

    def load_input_id(self, tensor_id):
        self.NetInTensors.append(tensor_id)

    # 图输出张量
    def load_output_id(self, tensor_id):
        self.NetOutTensors.append(tensor_id)

    # 在op_idx位置插入算子
    def insert_op(self, op, op_idx):
        assert len(self.AllOpIds) == len(self.AllOps)  # 确保之前的操作都是正确执行
        assert op_idx is not None
        self.AllOps.insert(op_idx, op)
        self.AllOpIds.insert(op_idx, id(op))  # op唯一身份标识

    def delete_op(self, op_idx):
        del self.AllOps[op_idx]
        del self.AllOpIds[op_idx]

    def add_tensor(self, tensor):
        if tensor.Id not in self.AllTensorIds:
            self.AllTensorIds.append(tensor.Id)
            self.AllTensors.append(tensor)

    def check_tensor(self, tensor_name) -> bool:
        if tensor_name in self.AllTensorIds:
            return True

    # 根据哈希值, 找张量
    def get_tensor(self, tensor_name) -> IRTensor:
        """Find a tensor in AllTensors"""
        assert tensor_name in self.AllTensorIds, f'{tensor_name} not in AllTensors'
        index = self.AllTensorIds.index(tensor_name)  # .index(id)，返回内容是id在列表中首次出现的索引值
        return self.AllTensors[index]

    def get_op(self, op_id):
        return self.AllOps[op_id]

    def get_op_idx(self, op):
        if op in self.AllOps:
            return self.AllOps.index(op)