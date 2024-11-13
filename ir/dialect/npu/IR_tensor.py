# 分析内存
import matplotlib
from ir.dialect.top.IR_tensor import *


class NpuIRTensor(IRTensor):
    def __init__(self, top_ir_tensor=None):
        super().__init__()

        self.life = None  # 生命周期
        self.num_data = None
        self.block_shape = []

        if top_ir_tensor:
            self.convert_from_top_ir_tensor(top_ir_tensor)

        self.address_info = None  # [b_h,b_w,b_c],
        # [h_s,h_l,w_s,w_l,c_s,c_l],
        # [m_addr,m_len]]

    def convert_from_top_ir_tensor(self, top_ir_tensor):
        self.__dict__.update(top_ir_tensor.__dict__)  # 将属性值复制

        self.num_data = np.prod(top_ir_tensor.Shape.list)
