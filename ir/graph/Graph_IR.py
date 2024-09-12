from ir.dialect.npu.IR_operator import *
from ir.dialect.top.IR_tensor import *


class GraphIR:

    def __init__(self):
        self.name = None
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
        self.WeightTensorIds = []
        self.WeightTensors = []
        self.WeightFormat = []
        self.WeightBaseAddress = []
        # Npu
        self.NetOutNpuOpId = []
        self.NetInputNpuOpId = []
        self.SubGraphs = []
        # config
        self.codegen_path = None

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

    def get_npu_op(self, npu_op_id):
        for npu_op in self.AllOps:
            if isinstance(npu_op, NpuOp):
                if npu_op.NpuOpId == npu_op_id:
                    return npu_op

            elif isinstance(npu_op, npu_op_group):
                for op in npu_op.npu_op_list:
                    if op.NpuOpId == npu_op_id:
                        return op
            else:
                raise Exception(NotImplementedError)

        raise ValueError

    def get_input_shape(self):
        return self.NetInTensors[0].Shape

    def add_weight_tensor(self, npu_op_id, weight_tensor):
        if npu_op_id not in self.WeightTensorIds:
            self.WeightTensorIds.append(npu_op_id)
            self.WeightTensors.append(weight_tensor)

    def get_weight_tensor(self, npu_op_id):
        if npu_op_id not in self.WeightTensorIds:
            return None
        index = self.WeightTensorIds.index(npu_op_id)
        return self.WeightTensors[index]

    def get_weight_base_addr(self, npu_op_id):
        if npu_op_id not in self.WeightTensorIds:
            return None
        index = self.WeightTensorIds.index(npu_op_id)
        return self.WeightBaseAddress[index]

    def check_weight_tensor(self, npu_op_id):
        if npu_op_id in self.WeightTensorIds:
            return True

    def add_weight_format(self, weight_format):
        self.WeightFormat.append(weight_format)

    def get_weight_format(self, npu_op_id):
        assert npu_op_id in self.WeightTensorIds
        index = self.WeightTensorIds.index(npu_op_id)
        return self.WeightFormat[index]

    def replace_tensor(self, tensor, tensor_id):
        assert tensor_id in self.AllTensorIds
        index = self.AllTensorIds.index(tensor_id)
        self.AllTensors[index] = tensor
