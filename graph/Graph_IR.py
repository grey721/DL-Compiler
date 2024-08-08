from dialect.top.IR_tensor import *


class GraphIR:
    def __init__(self) -> None:
        # 网络输入输出
        self.NetInTensors = []  # List[IRTensor]
        self.NetOutTensors = []  #
        # 算子、中间张量
        self.AllTensorNames = {}  # Name:tensor_idx
        self.AllOpIds = []
        self.AllTensors = []  # List[IRTensor]
        self.AllOps = []  # List[OpBase]

        self.AllCpuOps = []
        self.NetOutOpId = []
        self.NetInputOpId = []
        # 参数
        # self.WeightTensorNames =    {}
        # self.WeightTensors =        []
        # Npu
        self.NetOutNpuOpId = []
        self.NetInputNpuOpId = []
        self.SubGraphs = []

    def load_input_id(self, tensor_name):
        idx = self.AllTensorNames[tensor_name]
        self.NetInTensors.append(idx)

    # 图输出张量
    def load_output_id(self, tensor_name):
        idx = self.AllTensorNames[tensor_name]
        self.NetOutTensors.append(idx)

    # 在op_idx位置插入算子
    def insert_op(self, op, op_idx):
        assert len(self.AllOpIds) == len(self.AllOps)  # 确保之前的操作都是正确执行
        assert op_idx is not None
        self.AllOps.insert(op_idx, op)
        self.AllOpIds.insert(op_idx, id(op))  # op唯一身份标识

    def add_tensor(self, tensor):
        if tensor.Name not in self.AllTensorNames:
            self.AllTensorNames[tensor.Name] = tensor.Tensor_idx
            self.AllTensors.append(tensor)

    def check_tensor(self, tensor_name) -> bool:
        if tensor_name in self.AllTensorNames:
            return True

    # 根据哈希值, 找张量
    def get_tensor(self, tensor_name) -> IRTensor:
        """Find tensor from AllTensors"""
        assert tensor_name in self.AllTensorNames, f'{tensor_name} not in AllTensors'
        index = self.AllTensorNames[tensor_name]  # .index(id)，返回内容是id在列表中首次出现的索引值
        return self.AllTensors[index]
