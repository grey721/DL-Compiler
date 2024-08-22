from ir.dialect.top.IR_tensor import *
from copy import deepcopy


class NpuIRTensor(IRTensor):
    idx = None
    tensor_type = None
    tensor_size = None

    # from_shared_memory = True
    # from_global_memory = False


    def __init__(self, top_ir_tensor=None):
        super().__init__()
        self.Name = "npu_ir_tensor"

        self.tensor_life_cycle_for_block_list = None  # 生命周期
        self.intermediate_tensor_life_cycle_for_block_list = []
        self.intermediate_tensor_id_list = []
        self.tensor_info_list = []
        self.tensor_info_list_for_global_memory = []
        self.tensor_info_block_id_list = []

        if top_ir_tensor:
            self.convert_from_top_ir_tensor(top_ir_tensor)

        self.address_info = None  # [b_h,b_w,b_c],
        # [h_s,h_l,w_s,w_l,c_s,c_l],
        # [m_addr,m_len]]

    def convert_from_top_ir_tensor(self, top_ir_tensor):
        self.__dict__.update(top_ir_tensor.__dict__)  # 将属性值复制

        if top_ir_tensor.DataType == DataType.INT8:
            tensor_size = np.prod(top_ir_tensor.Shape.get_shape_as_np())  # 列表内所有元素乘积，单位字节
        if top_ir_tensor.DataType == DataType.INT32:
            tensor_size = np.prod(top_ir_tensor.Shape.get_shape_as_np()) * 4

        self.tensor_size = tensor_size
        self.Name = top_ir_tensor.Name

    def add_intermediate_tensor_life_cycle(self, group_block_id, life_cycle):
        if group_block_id not in self.intermediate_tensor_id_list:
            self.intermediate_tensor_id_list.append(group_block_id)
            self.intermediate_tensor_life_cycle_for_block_list.append(life_cycle)

    def get_intermediate_tensor_life_cycle(self, group_block_id):
        assert group_block_id in self.intermediate_tensor_id_list
        index = self.intermediate_tensor_id_list.index(group_block_id)
        tensor_life_cycle = self.intermediate_tensor_life_cycle_for_block_list[index]
        return tensor_life_cycle

    def check_intermediate_tensor(self, group_block_id):
        if group_block_id in self.intermediate_tensor_id_list:
            return True
        else:
            return False

    def get_tensor_info(self, block_id, memory_type="shared_memory"):
        assert block_id in self.tensor_info_block_id_list
        index = self.tensor_info_block_id_list.index(block_id)
        if memory_type == "shared_memory":
            return self.tensor_info_list[index]
        elif memory_type == "dma":
            return self.tensor_info_list_for_global_memory[index]
        else:
            raise Exception(NotImplementedError)

    def replace_tensor_info(self, tensor_info, block_id, memory_type="shared_memory"):
        assert block_id in self.tensor_info_block_id_list
        index = self.tensor_info_block_id_list.index(block_id)
        del self.tensor_info_block_id_list[index]
        del self.tensor_info_list[index]

        if memory_type == "dma":
            del self.tensor_info_list_for_global_memory[index]
            self.tensor_info_list_for_global_memory.append(tensor_info)

        self.tensor_info_list.append(tensor_info)
        self.tensor_info_block_id_list.append(block_id)

    def add_tensor_info(self, tensor_info, block_id, memory_type="shared_memory"):
        if len(self.tensor_info_list) > 0:

            if block_id not in self.tensor_info_block_id_list:
                self.tensor_info_list.append(tensor_info)
                if memory_type == "dma":
                    self.tensor_info_list_for_global_memory.append(tensor_info)
                self.tensor_info_block_id_list.append(block_id)
            else:
                post_tensor_info = deepcopy(tensor_info)
                post_tensor_info['block_address_list'] = post_tensor_info['block_address_list'].tolist()
                if isinstance(post_tensor_info.get("block_pad_list", None), np.ndarray):
                    post_tensor_info['block_pad_list'] = post_tensor_info['block_pad_list'].tolist()
                former_tensor_info = deepcopy(self.get_tensor_info(block_id, memory_type))
                former_tensor_info['block_address_list'] = former_tensor_info['block_address_list'].tolist()
                if isinstance(former_tensor_info.get("block_pad_list", None), np.ndarray):
                    former_tensor_info['block_pad_list'] = former_tensor_info['block_pad_list'].tolist()
                if isinstance(post_tensor_info.get("dma_block_address_list", None), np.ndarray):
                    post_tensor_info['dma_block_address_list'] = post_tensor_info['dma_block_address_list'].tolist()
                if isinstance(former_tensor_info.get("dma_block_address_list", None), np.ndarray):
                    former_tensor_info['dma_block_address_list'] = former_tensor_info['dma_block_address_list'].tolist()
                if post_tensor_info != former_tensor_info:
                    self.replace_tensor_info(tensor_info, block_id, memory_type)

        else:
            self.tensor_info_list.append(tensor_info)
            if memory_type == "dma":
                self.tensor_info_list_for_global_memory.append(tensor_info)
            self.tensor_info_block_id_list.append(block_id)

    def get_block_tensor_size(self, block_id):
        tensor_info = self.get_tensor_info(block_id)
        block_address_list = tensor_info["block_address_list"][0]
        h_l = block_address_list[1]
        w_l = block_address_list[3]
        c_l = block_address_list[5]
        block_tensor_size = h_l * w_l * c_l
        return block_tensor_size

    def update_tensor_info(self, tensor_info, block_id, memory_type="shared_memory"):
        assert tensor_info['tensor_id'] == self.tensor_id
        assert tensor_info['bank_polling_num'] in [1, 2, 4]
        s_b, e_b = self.tensor_life_cycle_for_block_list
        life_cycle_len = e_b - block_id + 1
        for _block_id in range(life_cycle_len):
            _block_id = _block_id + block_id
            self.add_tensor_info(tensor_info, _block_id, memory_type)
        print("tensor_info:", tensor_info)

    def update_concat_output_tensor_info(self, concat_input_tensor_info,
                                         concat_in_tensor_list,
                                         concat_input_tensor_id,
                                         output_tensor_info,
                                         block_id):
        # update_tensor_info
        concat_input_tensor_info['tensor_id'] = self.tensor_id
        concat_input_tensor_info['life_cycle'] = self.tensor_life_cycle_for_block_list

        assert len(concat_in_tensor_list) == 2
        if concat_in_tensor_list.index(concat_input_tensor_id) == 0:
            concat_input_tensor_info['block_address_list'][4] = concat_input_tensor_info['block_address_list'][5]
            concat_input_tensor_info['block_address_list'][5] = (output_tensor_info['block_address_list'][5]
                                                                 - concat_input_tensor_info['block_address_list'][4])
        else:
            concat_input_tensor_info['block_address_list'][4] = 0
            concat_input_tensor_info['block_address_list'][5] = (output_tensor_info['block_address_list'][5]
                                                                 - concat_input_tensor_info['block_address_list'][5])

        h = concat_input_tensor_info['block_address_list'][1]
        w = concat_input_tensor_info['block_address_list'][3]
        c = concat_input_tensor_info['block_address_list'][5]
        concat_input_tensor_info['tensor_size'] = h * w * c
        concat_input_tensor_info['concat_status'] = True

        life_cycle_len = concat_input_tensor_info['life_cycle'][1] - concat_input_tensor_info['life_cycle'][0]
        life_cycle_len = life_cycle_len + 1
        if life_cycle_len == 1:
            self.add_tensor_info(concat_input_tensor_info, block_id)
        else:
            for block_id in range(life_cycle_len):
                block_id = block_id + concat_input_tensor_info['life_cycle'][0]
                self.add_tensor_info(deepcopy(concat_input_tensor_info), block_id)
        print("tensor_info:", concat_input_tensor_info)
