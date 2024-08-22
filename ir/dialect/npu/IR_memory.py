import matplotlib

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import MultipleLocator
from copy import deepcopy

FIGURE_CONFIG = {
    "TRAIN_TENSOR_COLOR": "#68ac14",  # green
    "TRAIN_WEIGHT_COLOR": "#61a78f",  # light green:
    "TRAIN_ACTIVATION_COLOR": "#b9c8d8",  # light blue
    "TRAIN_GRADIENT_COLOR": "#35619f",  # blue
    "INFERENCE_COLOR": "#eab11f",  # yellow
    "FIGURE_W_INCH": 19,
    "FIGURE_H_INCH": 7,
    "DPI": 800,
    "Y_STEP": 96,
    "Y_MAX": 384,
    "X_STEP": 30,
    "X_MAX": 270,
    "SHOW_INDEX": False,
    "FONT_SIZE": 32,
    "INDEX_FONT_SIZE": 24,
}


class SharedMemory(object):
    group_bank_num = 4
    bank_size = 128 * 1024
    bank_group_size = 4 * 128 * 1024
    bank_group_num = 3

    def __init__(self):
        self.memory_type = "shared_memory"
        self.bank_num = 12
        self.bank_size = 128
        self.usage_record_dict = dict()
        self.usage_record_timestep_list = []
        self.available_bank_group_id_list = [0, 1, 2]
        self.used_bank_group_id_list = []
        self.bank_group_used_status_dict = dict()
        self.memory_init()

    def memory_init(self):
        for k in self.available_bank_group_id_list:
            addr = 0
            length = self.bank_group_size
            self.usage_record_dict[k] = []
            self.bank_group_used_status_dict[k] = [addr, length]

    def bank_group_mem_alloc(self, tensor_info, bank_group_id, block_id):
        tensor_size = tensor_info['tensor_size']
        self.get_bank_group_used_status_dict(block_id)
        addr, length = self.bank_group_used_status_dict[bank_group_id]
        if tensor_size <= length:
            alloc_len = tensor_size
            alloc_addr = addr
            tensor_info['addr'] = alloc_addr
            tensor_info['len'] = alloc_len
            self.usage_record_dict[bank_group_id].append(tensor_info)

        else:
            error_info = "share memory bank group id:{} memory allocate error".format(bank_group_id)
            raise Exception(error_info)

    def allocate(self, tensor, tensor_info, block_id, bank_groud_id=None):
        tensor_info['life_cycle'] = tensor.tensor_life_cycle_for_block_list
        if bank_groud_id == None:
            bank_group_id_list = self.get_available_bank_group_id_list()
        else:
            bank_group_id_list = [bank_groud_id]

        for bank_group_id in bank_group_id_list:
            tensor_info['bank_group_id'] = bank_group_id
            self.bank_group_mem_alloc(tensor_info, bank_group_id, block_id)
            if bank_group_id not in self.available_bank_group_id_list:
                raise KeyError
            self.available_bank_group_id_list.remove(bank_group_id)
            life_cycle_len = tensor_info['life_cycle'][1] - tensor_info['life_cycle'][0]
            life_cycle_len = life_cycle_len + 1
            if life_cycle_len == 1:
                tensor.add_tensor_info(tensor_info, block_id)
            else:
                for block_id in range(life_cycle_len):
                    block_id = block_id + tensor_info['life_cycle'][0]
                    tensor.add_tensor_info(tensor_info, block_id)
            break

    def allocate2(self, tensor, tensor_info, block_id, bank_groud_id=None):

        if tensor_info.get('life_cycle', None) == None:
            tensor_info['life_cycle'] = tensor.tensor_life_cycle_for_block_list
        if bank_groud_id == None:
            bank_group_id_list = self.get_available_bank_group_id_list()
        else:
            bank_group_id_list = [bank_groud_id]

        for bank_group_id in bank_group_id_list:
            tensor_info['bank_group_id'] = bank_group_id
            self.bank_group_mem_alloc(tensor_info, bank_group_id, block_id)
            if bank_group_id not in self.available_bank_group_id_list:
                raise KeyError
            self.available_bank_group_id_list.remove(bank_group_id)

            life_cycle_len = tensor_info['life_cycle'][1] - block_id
            life_cycle_len = life_cycle_len + 1
            if life_cycle_len == 1:
                pass
            else:
                for _block_id in range(life_cycle_len):
                    _block_id = _block_id + block_id
                    if _block_id > block_id:
                        _tensor_info = tensor.get_tensor_info(_block_id, self.memory_type)
                        if _tensor_info['dtype'] == tensor_info['dtype']:
                            _tensor_info['bank_group_id'] = tensor_info['bank_group_id']
                            _tensor_info['addr'] = tensor_info['addr']
                            _tensor_info['len'] = tensor_info['tensor_size']
            break

    def get_available_bank_group_id_list(self):
        return self.available_bank_group_id_list

    def get_bank_group_used_status_dict(self, block_id):

        for bank_group_id in range(self.bank_group_num):
            addr = 0
            length = self.bank_group_size

            if len(self.usage_record_dict[bank_group_id]) == 0:
                continue
            else:
                for tensor_info in self.usage_record_dict[bank_group_id]:
                    blk_start_id, blk_end_id = tensor_info['life_cycle']
                    used_addr = tensor_info['addr']
                    used_len = tensor_info['len']
                    if block_id <= blk_end_id:
                        addr = used_addr + used_len
                        length = length - addr
                    elif block_id > blk_end_id:
                        continue

            self.bank_group_used_status_dict[bank_group_id][0] = addr
            self.bank_group_used_status_dict[bank_group_id][1] = length

    def set_bank_group_id_used(self, bank_group_id):
        if bank_group_id in self.available_bank_group_id_list:
            self.available_bank_group_id_list.remove(bank_group_id)
        else:
            error_info = "share memory bank group id:{} already used".format(bank_group_id)
            raise Exception(error_info)

    def reset(self):
        self.available_bank_group_id_list = [0, 1, 2]

    def visualize(self):
        bank_group_id_list = self.usage_record_dict.keys()
        rectangle_list = []
        rectangle_info = {}
        max_life_cycle = 0
        for bgi in bank_group_id_list:
            base_start = self.bank_group_size * bgi
            for tensor_info in self.usage_record_dict[bgi]:
                rectangle_info['tensor_id'] = tensor_info['tensor_id']
                rectangle_info["life_cycle"] = tensor_info["life_cycle"]
                max_life_cycle = tensor_info["life_cycle"][1] \
                    if tensor_info["life_cycle"][1] > max_life_cycle else max_life_cycle
                rectangle_info["x_start"] = base_start + tensor_info["addr"]
                rectangle_info["y_start"] = tensor_info["life_cycle"][0]
                rectangle_info["x_len"] = tensor_info["len"]
                rectangle_info["y_len"] = (tensor_info["life_cycle"][1] - tensor_info["life_cycle"][0] + 1)
                rectangle_info["color"] = FIGURE_CONFIG["TRAIN_ACTIVATION_COLOR"]
                rectangle_list.append(deepcopy(rectangle_info))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)

        for rec in rectangle_list:
            x_start, y_start, \
                x_len, y_len, color = (rec["x_start"],
                                       rec["y_start"],
                                       rec["x_len"],
                                       rec["y_len"],
                                       rec["color"])

            hatch = None
            rect = Rectangle(
                (x_start, y_start),
                x_len,
                y_len,
                color=color,
                hatch=hatch,
            )
            rect.set_edgecolor("black")
            ax.add_patch(rect)

            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0

            ax.annotate(rec['tensor_id'], (cx, cy), color='b', weight='bold',
                        fontsize=12, ha='center', va='center')

        plt.xlim([0, self.bank_group_size * self.bank_group_num])
        plt.ylim([0, max_life_cycle + 1])
        plt.show()
        # print("------debug-----")


class GlobalMemory(object):

    def __init__(self):
        self.memory_type = "dma"
        self.group_bank_num = 4
        self.bank_num = 4
        self.bank_group_num = 1
        self.bank_group_size = 4 * 512 * 1024
        self.bank_size = 512
        self.usage_record_dict = dict()
        self.usage_record_timestep_list = []
        self.available_bank_group_id_list = [0]
        self.used_bank_group_id_list = []
        self.bank_group_used_status_dict = dict()
        self.memory_init()

    def memory_init(self):
        for k in self.available_bank_group_id_list:
            addr = 0
            length = self.bank_group_size
            self.usage_record_dict[k] = []
            self.bank_group_used_status_dict[k] = [addr, length]

    def bank_group_mem_alloc(self, tensor_info, bank_group_id, block_id):
        tensor_size = tensor_info['dma_tensor_size']
        self.get_bank_group_used_status_dict(block_id)
        addr, length = self.bank_group_used_status_dict[bank_group_id]
        if tensor_size <= length:
            alloc_len = tensor_size
            alloc_addr = addr
            tensor_info['dma_addr'] = alloc_addr
            tensor_info['dma_len'] = alloc_len
            self.usage_record_dict[bank_group_id].append(tensor_info)

        else:
            error_info = "global memory bank group id:{} memory allocate error".format(bank_group_id)
            raise Exception(error_info)

    def allocate2(self, tensor, tensor_info, block_id, bank_groud_id=None):

        if tensor_info.get('dma_life_cycle', None) == None:
            tensor_info['dma_life_cycle'] = tensor.tensor_life_cycle_for_block_list
        if bank_groud_id == None:
            bank_group_id_list = self.get_available_bank_group_id_list()
        else:
            bank_group_id_list = [bank_groud_id]

        for bank_group_id in bank_group_id_list:
            tensor_info['dma_bank_group_id'] = bank_group_id
            self.bank_group_mem_alloc(tensor_info, bank_group_id, block_id)
            if bank_group_id not in self.available_bank_group_id_list:
                raise KeyError
            self.available_bank_group_id_list.remove(bank_group_id)

            life_cycle_len = tensor_info['dma_life_cycle'][1] - block_id
            life_cycle_len = life_cycle_len + 1
            if life_cycle_len == 1:
                pass
            else:
                for _block_id in range(life_cycle_len):
                    _block_id = _block_id + block_id
                    if _block_id > block_id:
                        _tensor_info = tensor.get_tensor_info(_block_id, self.memory_type)
                        if _tensor_info['dtype'] == tensor_info['dtype']:
                            _tensor_info['dma_bank_group_id'] = tensor_info['dma_bank_group_id']
                            _tensor_info['dma_addr'] = tensor_info['dma_addr']
                            _tensor_info['dma_len'] = tensor_info['dma_tensor_size']
            break

    def get_available_bank_group_id_list(self):
        return self.available_bank_group_id_list

    def get_bank_group_used_status_dict(self, block_id):

        for bank_group_id in range(self.bank_group_num):
            addr = 0
            length = self.bank_group_size

            if len(self.usage_record_dict[bank_group_id]) == 0:
                continue
            else:
                for tensor_info in self.usage_record_dict[bank_group_id]:
                    blk_start_id, blk_end_id = tensor_info['dma_life_cycle']
                    used_addr = tensor_info['dma_addr']
                    used_len = tensor_info['dma_len']
                    if block_id <= blk_end_id:
                        addr = used_addr + used_len
                        length = length - addr
                    elif block_id > blk_end_id:
                        continue

            self.bank_group_used_status_dict[bank_group_id][0] = addr
            self.bank_group_used_status_dict[bank_group_id][1] = length

    def set_bank_group_id_used(self, bank_group_id):
        if bank_group_id in self.available_bank_group_id_list:
            self.available_bank_group_id_list.remove(bank_group_id)
        else:
            error_info = "global memory bank group id:{} already used".format(bank_group_id)
            raise Exception(error_info)

    def reset(self):
        self.available_bank_group_id_list = [0]

    def visualize(self):
        bank_group_id_list = self.usage_record_dict.keys()
        rectangle_list = []
        rectangle_info = {}
        max_life_cycle = 0
        for bgi in bank_group_id_list:
            base_start = self.bank_group_size * bgi
            for tensor_info in self.usage_record_dict[bgi]:
                rectangle_info['tensor_id'] = tensor_info['tensor_id']
                rectangle_info["dma_life_cycle"] = tensor_info["dma_life_cycle"]
                max_life_cycle = tensor_info["dma_life_cycle"][1] \
                    if tensor_info["life_cycle"][1] > max_life_cycle else max_life_cycle
                rectangle_info["x_start"] = base_start + tensor_info["dma_addr"]
                rectangle_info["y_start"] = tensor_info["dma_life_cycle"][0]
                rectangle_info["x_len"] = tensor_info["dma_len"]
                rectangle_info["y_len"] = (tensor_info["dma_life_cycle"][1] - tensor_info["dma_life_cycle"][0] + 1)
                rectangle_info["color"] = FIGURE_CONFIG["TRAIN_ACTIVATION_COLOR"]
                rectangle_list.append(deepcopy(rectangle_info))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)

        for rec in rectangle_list:
            x_start, y_start, \
                x_len, y_len, color = (rec["x_start"],
                                       rec["y_start"],
                                       rec["x_len"],
                                       rec["y_len"],
                                       rec["color"])

            hatch = None
            rect = Rectangle(
                (x_start, y_start),
                x_len,
                y_len,
                color=color,
                hatch=hatch,
            )
            rect.set_edgecolor("black")
            ax.add_patch(rect)

            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0

            ax.annotate(rec['tensor_id'], (cx, cy), color='b', weight='bold',
                        fontsize=12, ha='center', va='center')

        plt.xlim([0, self.bank_group_size * self.bank_group_num])
        plt.ylim([0, max_life_cycle + 1])
        plt.show()
