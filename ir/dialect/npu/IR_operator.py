import math
from copy import deepcopy

from ir.dialect.npu.IR_tensor import *
from ir.dialect.top.IR_operator import *


class NpuConv2d(Conv2d):
    fmi_size = None
    fmo_size = None
    weight_size = None
    input_offset = None
    weights_offset = None
    output_offset = None
    output_multiplier = None
    output_shift = None
    quantized_activation_min = None
    quantized_activation_max = None
    conv_type = None

    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()
        self.conv_type = 0

    def block_shape_reverse_infrence(self, block_address_list):

        bh, bw, bc, sl = block_address_list.shape
        assert sl == 6

        assert (self.StrideH == self.StrideW)
        assert (self.KerH == self.KerW)

        k_s = self.KerH
        Stride = self.StrideH
        pad_top = self.pad_top
        pad_bottom = self.pad_bottom
        pad_left = self.pad_left
        pad_right = self.pad_right
        InputC = self.InputShape[0].C

        block_pad_list = np.zeros([bh, bw, bc, 4], dtype=np.int32)

        new_c = math.floor(InputC / bc)
        c_nums = [new_c for i in range(bc)]
        for i in range(int(InputC - bc * new_c)):
            c_nums[i] += 1

        tem_c = []
        tem = 0
        for i in range(bc):
            tem_c.append(tem)
            tem += c_nums[i]

        for i in range(bh):
            for j in range(bw):
                for k in range(bc):
                    if i == 0:
                        block_pad_list[i][j][k][0] = pad_top
                    if i == bh - 1:
                        block_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        block_pad_list[i][j][k][2] = pad_left
                    if j == bw - 1:
                        block_pad_list[i][j][k][3] = pad_right

        for i in range(bh):
            for j in range(bw):
                for k in range(bc):
                    block_address_list[i][j][k][0] = block_address_list[i][j][k][0] * Stride - pad_top
                    block_address_list[i][j][k][1] = (block_address_list[i][j][k][1] * Stride) - Stride + k_s
                    block_address_list[i][j][k][2] = block_address_list[i][j][k][2] * Stride - pad_left
                    block_address_list[i][j][k][3] = (block_address_list[i][j][k][3] * Stride) - Stride + k_s
                    block_address_list[i][j][k][4] = tem_c[k]
                    block_address_list[i][j][k][5] = c_nums[k]
                    if i == 0:
                        block_address_list[i][j][k][0] += pad_top
                        block_address_list[i][j][k][1] -= pad_top
                    if i == bh - 1:
                        if pad_bottom > 0:
                            block_address_list[i][j][k][1] -= pad_bottom
                        else:
                            block_pad_list[i][j][k][1] = 0
                    if j == 0:
                        block_address_list[i][j][k][2] += pad_left
                        block_address_list[i][j][k][3] -= pad_left
                    if j == bw - 1:
                        if pad_right > 0:
                            block_address_list[i][j][k][3] -= pad_right
                        else:
                            block_pad_list[i][j][k][3] = 0
        return block_address_list, block_pad_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):

        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list, block_pad_list = self.block_shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)
        npu_op_block_address_list['block_pad_list'] = deepcopy(block_pad_list)

        return block_address_list

    def tile_shape_reverse_infrence(self, tile_address_list, block_address_list):

        assert (self.StrideH == self.StrideW)
        assert (self.KerH == self.KerW)
        k_s = self.KerH
        Stride = self.StrideH

        block_pad_list = block_address_list['block_pad_list']
        pad_top, pad_bottom, pad_left, pad_right = block_pad_list
        th, tw, tc, sl = tile_address_list.shape

        InputC = block_address_list['input_block_address_list'][5]

        tile_pad_list = np.zeros([th, tw, tc, 4], dtype=np.int32)

        new_c = math.floor(InputC / tc)
        c_nums = [new_c for i in range(tc)]
        for i in range(int(tc - tc * new_c)):
            c_nums[i] += 1

        tem_c = []
        tem = 0
        for i in range(tc):
            tem_c.append(tem)
            tem += c_nums[i]

        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    if i == 0:
                        tile_pad_list[i][j][k][0] = pad_top
                    if i == th - 1:
                        tile_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        tile_pad_list[i][j][k][2] = pad_left
                    if j == tw - 1:
                        tile_pad_list[i][j][k][3] = pad_right

        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    tile_address_list[i][j][k][0] = tile_address_list[i][j][k][0] * Stride - pad_top
                    tile_address_list[i][j][k][1] = (tile_address_list[i][j][k][1] * Stride) - Stride + k_s
                    tile_address_list[i][j][k][2] = tile_address_list[i][j][k][2] * Stride - pad_left
                    tile_address_list[i][j][k][3] = (tile_address_list[i][j][k][3] * Stride) - Stride + k_s
                    tile_address_list[i][j][k][4] = tem_c[k]
                    tile_address_list[i][j][k][5] = c_nums[k]
                    if i == 0:
                        tile_address_list[i][j][k][0] += pad_top
                        tile_address_list[i][j][k][1] -= pad_top
                    if i == th - 1:
                        tile_address_list[i][j][k][1] -= pad_bottom
                    if j == 0:
                        tile_address_list[i][j][k][2] += pad_left
                        tile_address_list[i][j][k][3] -= pad_left
                    if j == tw - 1:
                        tile_address_list[i][j][k][3] -= pad_right

        return tile_address_list, tile_pad_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list, block_address_list):

        npu_op_flow_tile_address_list['output_tile_address_list'] = deepcopy(tile_address_list)
        tile_address_list, tile_pad_list = self.tile_shape_reverse_infrence(tile_address_list, block_address_list)
        npu_op_flow_tile_address_list['input_tile_address_list'] = deepcopy(tile_address_list)
        npu_op_flow_tile_address_list['tile_pad_list'] = deepcopy(tile_pad_list)

        return tile_address_list


class NpuActivation(Activation):

    def __init__(self):
        super().__init__()

    def shape_reverse_infrence(self, address_list):
        return address_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):
        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list = self.shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)

        return block_address_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list):
        npu_op_flow_tile_address_list['input_tile_address_list'] = deepcopy(tile_address_list)
        tile_address_list = self.shape_reverse_infrence(tile_address_list)
        npu_op_flow_tile_address_list['output_tile_address_list'] = deepcopy(tile_address_list)

        return tile_address_list


class NpuLeakyRelu(NpuActivation):
    input_offset = None
    output_offset = None
    identity_output_multiplier = None
    alpha_output_multiplier = None
    identity_output_shift = None
    alpha_output_shift = None

    def __init__(self):
        super().__init__()
        self.lut_dict = None


class NpuLogistic(NpuActivation):
    input_offset = None
    input_multiplier = None
    output_offset = None
    output_multiplier = None

    quantized_activation_min = None
    quantized_activation_max = None

    def __init__(self):
        super().__init__()
        self.lut_dict = None


class NpuPool(Pool):
    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()

    def block_shape_reverse_infrence(self, block_address_list):

        assert (self.StrideH == self.StrideW)
        assert (self.KerH == self.KerW)

        bh, bw, bc, sl = block_address_list.shape
        assert sl == 6

        k_s = self.KerH
        Stride = self.StrideH
        pad_top = self.pad_top
        pad_bottom = self.pad_bottom
        pad_left = self.pad_left
        pad_right = self.pad_right

        block_pad_list = np.zeros([bh, bw, bc, 4], dtype=np.int32)

        for i in range(bh):
            for j in range(bw):
                for k in range(bc):
                    if i == 0:
                        block_pad_list[i][j][k][0] = pad_top
                    if i == bh - 1:
                        block_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        block_pad_list[i][j][k][2] = pad_left
                    if j == bw - 1:
                        block_pad_list[i][j][k][3] = pad_right

        for i in range(bh):
            for j in range(bw):
                for k in range(bc):
                    block_address_list[i][j][k][0] = block_address_list[i][j][k][0] * Stride - pad_top
                    block_address_list[i][j][k][1] = (block_address_list[i][j][k][1] * Stride) - Stride + k_s
                    block_address_list[i][j][k][2] = block_address_list[i][j][k][2] * Stride - pad_left
                    block_address_list[i][j][k][3] = (block_address_list[i][j][k][3] * Stride) - Stride + k_s
                    if i == 0:
                        block_address_list[i][j][k][0] += pad_top
                        block_address_list[i][j][k][1] -= pad_top
                    if i == bh - 1:
                        block_address_list[i][j][k][1] -= pad_bottom
                    if j == 0:
                        block_address_list[i][j][k][2] += pad_left
                        block_address_list[i][j][k][3] -= pad_left
                    if j == bw - 1:
                        block_address_list[i][j][k][3] -= pad_right

        return block_address_list, block_pad_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):

        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list, block_pad_list = self.block_shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)
        npu_op_block_address_list['block_pad_list'] = deepcopy(block_pad_list)

        return block_address_list

    def tile_shape_reverse_infrence(self, tile_address_list, block_address_list):

        assert (self.StrideH == self.StrideW)
        assert (self.KerH == self.KerW)
        k_s = self.KerH
        Stride = self.StrideH
        InputC = self.InputShape[0].C
        th, tw, tc, sl = tile_address_list.shape
        block_pad_list = block_address_list["block_pad_list"]
        pad_top, pad_bottom, pad_left, pad_right = block_pad_list

        tile_pad_list = np.zeros([th, tw, tc, 4], dtype=np.int32)

        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    if i == 0:
                        tile_pad_list[i][j][k][0] = pad_top
                    if i == th - 1:
                        tile_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        tile_pad_list[i][j][k][2] = pad_left
                    if j == tw - 1:
                        tile_pad_list[i][j][k][3] = pad_right

        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    tile_address_list[i][j][k][0] = tile_address_list[i][j][k][0] * Stride - pad_top
                    tile_address_list[i][j][k][1] = (tile_address_list[i][j][k][1] * Stride) - Stride + k_s
                    tile_address_list[i][j][k][2] = tile_address_list[i][j][k][2] * Stride - pad_left
                    tile_address_list[i][j][k][3] = (tile_address_list[i][j][k][3] * Stride) - Stride + k_s
                    if i == 0:
                        tile_address_list[i][j][k][0] += pad_top
                        tile_address_list[i][j][k][1] -= pad_top
                    if i == th - 1:
                        tile_address_list[i][j][k][1] -= pad_bottom
                    if j == 0:
                        tile_address_list[i][j][k][2] += pad_left
                        tile_address_list[i][j][k][3] -= pad_left
                    if j == tw - 1:
                        tile_address_list[i][j][k][3] -= pad_right

        return tile_address_list, tile_pad_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list, block_address_list):

        npu_op_flow_tile_address_list['output_tile_address_list'] = deepcopy(tile_address_list)
        tile_address_list, tile_pad_list = self.tile_shape_reverse_infrence(tile_address_list, block_address_list)
        npu_op_flow_tile_address_list['input_tile_address_list'] = deepcopy(tile_address_list)
        npu_op_flow_tile_address_list['tile_pad_list'] = deepcopy(tile_pad_list)

        return tile_address_list


class NpuResize(Resize):
    ratio_w = None
    ratio_h = None

    def __init__(self):
        super().__init__()

    def block_shape_reverse_infrence(self, block_address_list):

        bh, bw, bc, sl = block_address_list.shape
        assert bh * bw * bc == 1

        block_address_list[0][0][0][1] = self.InputShape[0].H
        block_address_list[0][0][0][3] = self.InputShape[0].W

        return block_address_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):

        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list = self.block_shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)

        return block_address_list

    def tile_shape_reverse_infrence(self, tile_address_list):

        th, tw, tc, sl = tile_address_list.shape
        #
        tile_drop_line_list = np.zeros([th, tw, tc, 4], dtype=np.int32)
        scale = self.ScaleFactor

        if th > 1:
            assert scale % 2 == 0

        assert tw == 1

        tile_ih_list = []
        tile_oh_list = []
        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    tile_oh = tile_address_list[i][j][k][1]
                    tile_ih = int(tile_oh / scale)
                    tile_ih_list.append(tile_ih)

        if th > 1:
            for i in range(th):
                for j in range(tw):
                    for k in range(tc):
                        tile_oh = tile_address_list[i][j][k][1]
                        assert tile_oh % scale == 0

                        if i == 0:
                            tile_address_list[i][j][k][0] = 0
                            tile_address_list[i][j][k][1] = tile_ih + 1
                            tile_drop_line_list[i][j][k][0] = 0
                            tile_drop_line_list[i][j][k][1] = scale

                        elif 0 < i < th - 1:
                            tile_address_list[i][j][k][0] = sum(tile_ih_list[:i]) - 1
                            if self.Mode == 0:
                                tile_address_list[i][j][k][1] = tile_ih + 1
                            else:
                                tile_address_list[i][j][k][1] = tile_ih + 2
                            tile_drop_line_list[i][j][k][0] = scale
                            tile_drop_line_list[i][j][k][1] = scale

                        else:
                            tile_address_list[i][j][k][0] = sum(tile_ih_list) - tile_ih_list[i] - 1
                            tile_address_list[i][j][k][1] = tile_ih_list[i] + 1
                            tile_drop_line_list[i][j][k][0] = scale
                            tile_drop_line_list[i][j][k][1] = 0

                        tile_address_list[i][j][k][2] = 0
                        tile_address_list[i][j][k][3] = int(tile_address_list[i][j][k][3] / scale)
        else:
            for i in range(th):
                for j in range(tw):
                    for k in range(tc):
                        assert tile_oh % scale == 0
                        tile_address_list[i][j][k][0] = 0
                        tile_address_list[i][j][k][1] = int(tile_address_list[i][j][k][1] / scale)
                        tile_address_list[i][j][k][2] = 0
                        tile_address_list[i][j][k][3] = int(tile_address_list[i][j][k][3] / scale)

        return tile_address_list, tile_drop_line_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list):

        npu_op_flow_tile_address_list['output_tile_address_list'] \
            = deepcopy(tile_address_list)

        tile_address_list, tile_drop_line_list = self.tile_shape_reverse_infrence(tile_address_list)

        npu_op_flow_tile_address_list['input_tile_address_list'] \
            = deepcopy(tile_address_list)

        npu_op_flow_tile_address_list['tile_drop_line_list'] \
            = deepcopy(tile_drop_line_list)
        return tile_address_list


class NpuConcat(Concat):

    def __init__(self):
        super().__init__()
        self.main_input_tensor_id = None

    def block_shape_reverse_infrence(self, block_address_list):

        bh, bw, bc, sl = block_address_list.shape
        assert bh * bw * bc == 1

        block_address_list[0][0][0][4] = 0
        block_address_list[0][0][0][5] = self.InputShape[0].C

        block_address_list_0 = deepcopy(block_address_list)

        block_address_list[0][0][0][4] = 0
        block_address_list[0][0][0][5] = self.InputShape[1].C

        return block_address_list_0, block_address_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):

        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list_0, block_address_list = self.block_shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list_0)
        npu_op_block_address_list['input1_block_address_list'] = deepcopy(block_address_list)

        return block_address_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list):

        npu_op_flow_tile_address_list['output_tile_address_list'] \
            = deepcopy(tile_address_list)

        th, tw, tc, sl = tile_address_list.shape
        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    tile_address_list[i][j][k][4] = 0
                    tile_address_list[i][j][k][5] = self.InputShape[0].C
                    npu_op_flow_tile_address_list['input_tile_address_list'] \
                        = deepcopy(tile_address_list)

                    tile_address_list[i][j][k][4] = 0
                    tile_address_list[i][j][k][5] = self.InputShape[1].C
                    npu_op_flow_tile_address_list['input1_tile_address_list'] \
                        = deepcopy(tile_address_list)

        return tile_address_list


class NpuSplit(Split):

    def __init__(self):
        super().__init__()
        # TODO


class NpuElemWise(ElemWise):
    left_shift = None
    input_multiplier = None
    input_shift = None
    input_offset = None
    input1_multiplier = None
    input1_shift = None
    input1_offset = None
    output_multiplier = None
    output_shift = None
    output_offset = None
    quantized_activation_min = None
    quantized_activation_max = None

    def __init__(self):
        super().__init__()

    def shape_reverse_infrence(self, address_list):
        return address_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):
        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list = self.shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)

        return block_address_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list):
        npu_op_flow_tile_address_list['input_tile_address_list'] = deepcopy(tile_address_list)
        tile_address_list = self.shape_reverse_infrence(tile_address_list)
        npu_op_flow_tile_address_list['output_tile_address_list'] = deepcopy(tile_address_list)

        return tile_address_list


class NpuReshape(Reshape):
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuTranspose(Transpose):
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuSoftmax(NpuActivation):
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuMean(Mean):
    Type = "NpuMean"
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuPad(Pad):
    Device = "npu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step

    def block_shape_reverse_infrence(self, block_address_list):

        bh, bw, bc, sl = block_address_list.shape

        pad_top = self.pad_top
        pad_bottom = self.pad_bottom
        pad_left = self.pad_left
        pad_right = self.pad_right
        block_pad_list = np.zeros([bh, bw, bc, 4], dtype=np.int32)

        for i in range(bh):
            for j in range(bw):
                for k in range(bc):
                    if i == 0:
                        block_pad_list[i][j][k][0] = pad_top
                    if i == bh - 1:
                        block_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        block_pad_list[i][j][k][2] = pad_left
                    if j == bw - 1:
                        block_pad_list[i][j][k][3] = pad_right

                    if i == 0:
                        block_address_list[i][j][k][1] -= pad_top
                    if i == bh - 1:
                        block_address_list[i][j][k][1] -= pad_bottom
                    if j == 0:
                        block_address_list[i][j][k][3] -= pad_left
                    if j == bw - 1:
                        block_address_list[i][j][k][3] -= pad_right

        return block_address_list, block_pad_list

    def update_block_address_list(self, npu_op_block_address_list, block_address_list):

        npu_op_block_address_list['output_block_address_list'] = deepcopy(block_address_list)
        block_address_list, block_pad_list = self.block_shape_reverse_infrence(block_address_list)
        npu_op_block_address_list['input_block_address_list'] = deepcopy(block_address_list)
        npu_op_block_address_list['block_pad_list'] = deepcopy(block_pad_list)

        return block_address_list

    def tile_shape_reverse_infrence(self, tile_address_list):

        pad_top = self.pad_top
        pad_bottom = self.pad_bottom
        pad_left = self.pad_left
        pad_right = self.pad_right
        th, tw, tc, sl = tile_address_list.shape
        tile_pad_list = np.zeros([th, tw, tc, 4], dtype=np.int32)

        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    if i == 0:
                        tile_pad_list[i][j][k][0] = pad_top
                    if i == th - 1:
                        tile_pad_list[i][j][k][1] = pad_bottom
                    if j == 0:
                        tile_pad_list[i][j][k][2] = pad_left
                    if j == tw - 1:
                        tile_pad_list[i][j][k][3] = pad_right

                    if i == 0:
                        tile_address_list[i][j][k][1] -= pad_top
                    if i == th - 1:
                        tile_address_list[i][j][k][1] -= pad_bottom
                    if j == 0:
                        tile_address_list[i][j][k][3] -= pad_left
                    if j == tw - 1:
                        tile_address_list[i][j][k][3] -= pad_right

        return tile_address_list, tile_pad_list

    def update_tile_address_list(self, npu_op_flow_tile_address_list, tile_address_list):

        npu_op_flow_tile_address_list['output_tile_address_list'] = deepcopy(tile_address_list)
        npu_op_flow_tile_address_list['input_tile_address_list'] = deepcopy(tile_address_list)
        tile_address_list, tile_pad_list = self.tile_shape_reverse_infrence(tile_address_list)
        npu_op_flow_tile_address_list['tile_pad_list'] = deepcopy(tile_pad_list)
        return tile_address_list


class NpuFullConnected(FullConnected):
    fmi_size = None
    fmo_size = None
    weight_size = None
    input_offset = None
    weights_offset = None
    output_offset = None
    output_multiplier = None
    output_shift = None
    quantized_activation_min = None
    quantized_activation_max = None
    conv_type = None

    def __init__(self):
        super().__init__()


class NpuOp(OpBase):
    # NPU OP MODE
    NpuOpMode = None
    NpuShortCutMode = None

    # PAD OP
    NpuOpPad = False
    NpuOpPadOp = None

    # CONV OP
    NpuOpConv = True
    NpuOpConvOp = None

    NpuOpFc = False
    NpuOpFcOp = None

    # POST OP
    NpuOpActivate = False
    NpuOpActivateOp = None

    NpuOpElemWise = False
    NpuOpElemWiseOp = None

    NpuOpPool = False
    NpuOpPoolOp = None
    NpuOpResize = False
    NpuOpResizeOp = None
    NpuOpConcat = False
    NpuOpConcatOp = None

    NpuOpReshape = False
    NpuOpReshapeOp = None

    # POST OP OUT
    NpuOpActiOut = False
    NpuOpPoolOut = False
    NpuOpResizeOut = False
    NpuOpElwOut = False

    NpuOpShortCutOut = False
    NpuOpShortCutOp = None

    NpuOpFmiSize = None
    NpuOpFmoSize = None
    NpuOpFmo1Size = None

    # time
    TimeStep = None
    Device = "npu"
    NpuOpId = None

    def __init__(self):
        super().__init__()
        self.Type = "NpuOp"
        self.Name = "NpuOp"
        self.NpuOpFlow = []
        self.fmi_tensor = []
        self.fmo_tensor = []
        self.weight_tensor = []
        self.concat_input_tensor = []
        self.short_cut_out_tensor = []  # 允许网络中的某些层直接跳过一些层与后续层相连?
        self.elemwise_input_tensor = []
        self.output_tensor_for_cancat = []
        self.concat_output = False
        self.concat_output_shape = None
        self.fmi_from_global_memory = False

    def __repr__(self):
        return f"""
    ######################## NpuOp.{self.NpuOpId} #############################
    Device: {self.Device}
    NpuOpMode: {self.NpuOpMode}
    NpuShortCutMode: {self.NpuShortCutMode}
    NpuOpPad: {self.NpuOpPad}
    NpuOpConv: {self.NpuOpConv}
    NpuOpFc: {self.NpuOpFc}
    NpuOpActivate: {self.NpuOpActivate}
    NpuOpElemWise: {self.NpuOpElemWise}
    NpuOpPool: {self.NpuOpPool}
    NpuOpResize: {self.NpuOpResize}
    NpuOpConcat: {self.NpuOpConcat}
    NpuOpReshape: {self.NpuOpReshape}

    NpuOpActiOut: {self.NpuOpActiOut}
    NpuOpPoolOut: {self.NpuOpPoolOut}
    NpuOpResizeOut: {self.NpuOpResizeOut}
    NpuOpElwOut: {self.NpuOpElwOut}
    NpuOpShortCutOut: {self.NpuOpShortCutOut}
    ######################## NpuOp.{self.NpuOpId} #############################
    """

    def add_fmi_tensor(self, ir_tensor_id):
        self.fmi_tensor.append(ir_tensor_id)

    def add_fmo_tensor(self, ir_tensor_id):
        self.fmo_tensor.append(ir_tensor_id)

    def add_weight_tensor(self, ir_tensor_id):
        self.weight_tensor.append(ir_tensor_id)

    def add_concat_input_tensor(self, ir_tensor_id):
        self.concat_input_tensor.append(ir_tensor_id)

    def add_short_cut_out_tensor(self, ir_tensor_id):
        self.short_cut_out_tensor.append(ir_tensor_id)

    def add_elemwise_input_tensor(self, ir_tensor_id):
        self.elemwise_input_tensor.append(ir_tensor_id)

    def add_output_tensor_for_cancat(self, ir_tensor_id):
        self.output_tensor_for_cancat.append(ir_tensor_id)

    def set_time_step(self, time_step):
        self.TimeStep = time_step

    def set_fmi_size(self):
        self.NpuOpFmiSize = self.NpuOpFlow[0].get_fmi_size()

    def set_fmo_size(self):
        self.NpuOpFmoSize = self.NpuOpFlow[-1].get_fmo_size()

    def set_fmo1_size(self):
        if self.NpuOpShortCutOut:
            self.NpuOpFmo1Size = self.NpuOpShortCutOp.get_fmo_size()

    def init_tensor(self):

        self.add_fmi_tensor(self.NpuOpFlow[0].InTensors[0])
        self.add_fmo_tensor(self.NpuOpFlow[-1].OutTensors[0])

        for idx in self.NpuOpFlow[0].InTensors[1:3]:
            self.add_weight_tensor(idx)

        if self.NpuOpShortCutOut:
            self.add_short_cut_out_tensor(self.NpuOpShortCutOp.OutTensors[0])

        if self.NpuOpConcat:
            npu_flow_tensor_record = []
            for npu_op in self.NpuOpFlow:
                if not isinstance(npu_op, NpuConcat):
                    npu_flow_tensor_record.extend(npu_op.InTensors)
                    npu_flow_tensor_record.extend(npu_op.OutTensors)
                else:
                    break

            for in_tensor in self.NpuOpConcatOp.InTensors:
                if in_tensor not in npu_flow_tensor_record:
                    self.add_concat_input_tensor(in_tensor)

        if self.NpuOpElemWise:
            npu_flow_tensor_record = []
            for npu_op in self.NpuOpFlow:
                if not isinstance(npu_op, NpuElemWise):
                    npu_flow_tensor_record.extend(npu_op.InTensors)
                    npu_flow_tensor_record.extend(npu_op.OutTensors)
                else:
                    break

            for in_tensor in self.NpuOpElemWiseOp.InTensors:
                if in_tensor not in npu_flow_tensor_record:
                    self.add_elemwise_input_tensor(in_tensor)
                    elew_input_len = len(self.NpuOpElemWiseOp.InTensors)
                    assert elew_input_len == 2
                    other_inp_index = self.NpuOpElemWiseOp.InTensors.index(in_tensor)
                    if other_inp_index == 0:
                        self.NpuOpElemWiseOp.input_index_for_tflite = 1
                    else:
                        self.NpuOpElemWiseOp.input_index_for_tflite = 0

    def set_short_cut_op(self):
        flow_len = len(self.NpuOpFlow)
        for i, p in enumerate(self.NpuOpFlow):
            if i < flow_len - 1:
                if len(p.PostTopOpId) > 1:  # 之后有多个op，存在捷径
                    self.NpuOpShortCutOut = True
                    self.NpuOpShortCutOp = p
                    if isinstance(p, NpuActivation):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ACTIVATION_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuConv2d):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.INPUT_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuPool):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.POOL_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuResize):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.RESIZE_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuElemWise):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ELEW_SHORT_CUT_OUTPUT

    def init_all(self):
        self.set_short_cut_op()
        self.set_fmi_size()
        self.set_fmo_size()
        self.set_fmo1_size()
        self.init_tensor()


class VpuPostOpSetMode(object):
    NONE = 0

    ACTIVATION = 1
    ELW = 2
    RESIZE = 3
    POOL = 3

    ACTIVATION_ELW = 4
    ACTIVATION_RESIZE = 5
    ACTIVATION_POOL = 5

    ELW_ACTIVATION = 6
    ELW_RESIZE = 7
    ELW_POOL = 7

    POOL_ELW = 8
    POOL_ACTIVATION = 9

    RESIZE_ELW = 8
    RESIZE_ACTIVATION = 9

    ACTIVATION_ELW_POOL = 10
    ACTIVATION_ELW_RESIZE = 10

    ACTIVATION_POOL_ELW = 11
    ACTIVATION_RESIZE_ELW = 11

    ELW_ACTIVATION_POOL = 12
    ELW_ACTIVATION_RESIZE = 12

    ELW_POOL_ACTIVATION = 13
    ELW_RESIZE_ACTIVATION = 13

    POOL_ELW_ACTIVATION = 14
    RESIZE_ELW_ACTIVATION = 14

    POOL_ACTIVATION_ELW = 15
    RESIZE_ACTIVATION_ELW = 15


class VpuAdditionOutputSelection(object):
    INPUT_SHORT_CUT_OUTPUT = 0
    ACTIVATION_SHORT_CUT_OUTPUT = 1
    RESIZE_SHORT_CUT_OUTPUT = 2
    POOL_SHORT_CUT_OUTPUT = 2
    ELEW_SHORT_CUT_OUTPUT = 3


class block_param(object):
    NpuOp = None
    group_block_id = None  # [h0,h1], 0 meaning h0
    block_split_mode = None  # h2w1c1
    tile_split_mode = None
    bias_add = False
    int32_out = False
    npu_psum_add = False
    int8_out = True
    layer_group_flag = False
    input_block = False
    output_block = False
    npu_op_group_id = None
    npu_op_id = None
    npu_op_block_id = None
    short_cut_out = False
    concat_input = False
    elemwise_input = False
    backend = None
    dma_read = False
    dma_write = False

    def __init__(self):
        self.weight_mapping_dict = dict(weight_format=None)

        self.npu_op_flow_block_address_list = None
        self.npu_op_flow_tile_address_list = None
        self.tile_output_vpu_shape_list = None

        self.dma_read_param_dict = dict(tensor_info=None)
        self.dma_write_param_dict = dict(tensor_info=None)
        self.shm_read_param_dict = dict(tensor_info=None)  # 共享内存
        self.shm_psum_read_param_dict = dict(tensor_info=None)
        self.shm_elemwise_read_param_dict = dict(tensor_info=None)
        self.concat_shm_read_param_dict = dict(tensor_info=None)
        self.short_cut_shm_write_param_dict = dict(tensor_info=None)
        self.shm_write_param_dict = dict(tensor_info=None)

    def set_npu_op(self, npu_op):
        self.NpuOp = npu_op

    def get_input_tensor_info(self) -> dict:
        input_tensor_info = self.npu_op_flow_block_address_list[0]['input_tensor_info']
        h, w, c = self.NpuOp.InputShape[0].H, self.NpuOp.InputShape[0].W, self.NpuOp.InputShape[0].C
        input_tensor_info['origin_shape'] = [h, w, c]
        input_tensor_info['concat_input'] = False
        input_tensor_info['concat_output_shape'] = None
        if 'input_block_address_list' in self.npu_op_flow_block_address_list[0].keys():
            input_tensor_info['block_address_list'] = self.npu_op_flow_block_address_list[0]['input_block_address_list']
        if 'block_pad_list' in self.npu_op_flow_block_address_list[0].keys():
            input_tensor_info['block_pad_list'] = self.npu_op_flow_block_address_list[0]['block_pad_list']
        return input_tensor_info

    def get_output_tensor_info(self):
        if self.int32_out is True:
            output_tensor_info = self.npu_op_flow_block_address_list[-1]['output_tensor_info']
            output_tensor_info['block_address_list'] = self.npu_op_flow_block_address_list[0][
                'output_block_address_list']
            h, w, c = output_tensor_info['block_address_list'][1], \
                output_tensor_info['block_address_list'][3], \
                output_tensor_info['block_address_list'][5]
            output_tensor_info['origin_shape'] = [h, w, c]

        else:
            output_tensor_info = self.npu_op_flow_block_address_list[-1]['output_tensor_info']
            h, w, c = self.NpuOp.OutputShape[0].H, self.NpuOp.OutputShape[0].W, self.NpuOp.OutputShape[0].C
            output_tensor_info['origin_shape'] = [h, w, c]
            if self.concat_input:
                output_tensor_info['concat_input'] = True
                output_tensor_info['concat_output_shape'] = self.NpuOp.concat_output_shape
                output_tensor_info['origin_shape'] = self.NpuOp.concat_output_shape[1:]

            output_tensor_info['block_address_list'] = \
                self.npu_op_flow_block_address_list[-1]['output_block_address_list']

        return output_tensor_info

    # TODO short_cut
    def get_short_cut_out_tensor_info(self):
        assert self.short_cut_out is True
        for index, npu_op in enumerate(self.NpuOp.NpuOpFlow):
            if id(npu_op) == id(self.NpuOp.NpuOpShortCutOp):
                break
        # TODO 需要吗?
        else:
            return None
        short_cut_out_tensor_info = dict(tensor_id=None,
                                         group_block_id=None,
                                         tensor_split=False,
                                         block_address_list=None,
                                         tensor_size=None,
                                         life_cycle=None,
                                         bank_group_id=None,
                                         addr=None,
                                         len=None,
                                         origin_shape=None)

        short_cut_out_tensor_info['tensor_id'] = npu_op.OutTensors[0]
        short_cut_out_tensor_info['group_block_id'] = self.group_block_id
        if self.block_split_mode.split_num > 1:
            short_cut_out_tensor_info['tensor_split'] = True

        short_cut_out_tensor_info['block_address_list'] = \
            self.npu_op_flow_block_address_list[index]['output_block_address_list']

        h, w, c = npu_op.OutputShape[0].H, npu_op.OutputShape[0].W, npu_op.OutputShape[0].C
        short_cut_out_tensor_info['origin_shape'] = [h, w, c]
        block_tensor_size = h * w * c
        if len(self.NpuOp.output_tensor_for_cancat) == 1:
            if short_cut_out_tensor_info['tensor_id'] == self.NpuOp.output_tensor_for_cancat[0]:
                concat_in_tensor_list = self.NpuOp.concat_in_tensor_list
                assert len(concat_in_tensor_list) == 2
                tensor_id = short_cut_out_tensor_info['tensor_id']
                tensor_concat_order = concat_in_tensor_list.index(tensor_id)
                if tensor_concat_order == 1:
                    short_cut_out_tensor_info['block_address_list'][4] = \
                        self.NpuOp.concat_output_shape[-1] - short_cut_out_tensor_info['block_address_list'][5]

                short_cut_out_tensor_info['concat_input'] = True
                short_cut_out_tensor_info['concat_output_shape'] = self.NpuOp.concat_output_shape
                short_cut_out_tensor_info['origin_shape'] = self.NpuOp.concat_output_shape[1:].tolist()
                short_cut_out_tensor_info['tensor_size'] = np.prod(self.NpuOp.concat_output_shape)
        else:
            short_cut_out_tensor_info['concat_input'] = False
            short_cut_out_tensor_info['concat_output_shape'] = None
            short_cut_out_tensor_info['tensor_size'] = block_tensor_size

        return short_cut_out_tensor_info

    def get_output_tensor_size(self, tensor_size_flag="origin"):
        output_tensor_info = self.get_output_tensor_info()
        if tensor_size_flag == "origin":
            h_l, w_l, c_l = output_tensor_info['origin_shape']
        elif tensor_size_flag == "block":
            _, h_l, _, w_l, _, c_l = output_tensor_info['block_address_list']
        else:
            raise Exception(NotImplementedError)
        block_tensor_size = h_l * w_l * c_l
        block_tensor_size = block_tensor_size * 4 if self.int32_out else block_tensor_size
        return block_tensor_size

    def get_input_tensor_size(self, tensor_size_flag="origin"):
        input_tensor_info = self.get_input_tensor_info()
        if tensor_size_flag == "origin":
            h_l, w_l, c_l = input_tensor_info['origin_shape']
        elif tensor_size_flag == "block":
            _, h_l, _, w_l, _, c_l = input_tensor_info['block_address_list']
        else:
            raise Exception(NotImplementedError)
        block_tensor_size = h_l * w_l * c_l
        block_tensor_size = block_tensor_size * 4 if self.npu_psum_add else block_tensor_size
        return block_tensor_size

    def get_input_tensor_id(self):
        input_tensor_info = self.npu_op_flow_block_address_list[0]['input_tensor_info']
        return input_tensor_info['tensor_id']

    def get_output_tensor_id(self):
        output_tensor_info = self.npu_op_flow_block_address_list[-1]['output_tensor_info']
        return output_tensor_info['tensor_id']

    def get_short_cut_out_tensor_id(self):
        assert self.short_cut_out is True
        short_cut_out_tensor_id = self.NpuOp.NpuOpShortCutOp.OutTensors[0]
        return short_cut_out_tensor_id

    def get_concat_input_tensor_id(self):
        assert self.concat_input is True
        assert len(self.NpuOp.concat_input_tensor) == 1
        concat_input_tensor_id = self.NpuOp.concat_input_tensor[0]
        return concat_input_tensor_id

    def get_elemwise_input_tensor_id(self):
        assert self.elemwise_input is True
        assert len(self.NpuOp.elemwise_input_tensor) == 1
        elemwise_input_tensor_id = self.NpuOp.elemwise_input_tensor[0]
        return elemwise_input_tensor_id

    def get_npu_conv_op(self):
        assert self.NpuOp is not None
        return self.NpuOp.NpuOpConvOp

    def weight_mapping_init(self):
        npu_conv_op = self.get_npu_conv_op()
        if npu_conv_op is not None:
            conv_type = npu_conv_op.conv_type
            kernel_n = npu_conv_op.OutputShape[0].C
            kernel_h = npu_conv_op.KerH
            kernel_w = npu_conv_op.KerW
            if self.block_split_mode.split_num > 1:
                # block被分片，当前分片的起始通道是  每个分片的通道数（input_c_bin） * 当前分片的编号（self.group_block_id[-1]）
                input_c_bin = npu_conv_op.InputShape[0].C / self.block_split_mode.c
                kernel_c_range = [input_c_bin * self.group_block_id[-1],
                                  input_c_bin * (self.group_block_id[-1] + 1)]
            else:
                kernel_c_range = [0, npu_conv_op.InputShape[0].C]

            first_layer_flag = npu_conv_op.FirstLayer
            output_tensor_info = self.get_output_tensor_info()

            _weight_mapping_dict = self.backend.get_weight_mapping_param(conv_type, kernel_n, kernel_h,
                                                                         kernel_w, kernel_c_range, output_tensor_info,
                                                                         first_layer_flag)
            self.tile_split_mode = deepcopy(_weight_mapping_dict['tile_split_mode'])
            self.weight_mapping_dict.update(_weight_mapping_dict)

        else:
            assert self.block_split_mode.h == 1
            assert self.block_split_mode.w == 1
            assert self.block_split_mode.c == 1

            ih, iw, ic = self.NpuOp.InputShape[0].H, self.NpuOp.InputShape[0].W, self.NpuOp.InputShape[0].C
            _weight_mapping_dict = self.backend.get_single_op_mapping_param(ih, iw, ic)
            self.tile_split_mode = deepcopy(_weight_mapping_dict['tile_split_mode'])
            self.weight_mapping_dict.update(_weight_mapping_dict)

    # 检查在进行卷积操作时，所需的行缓存（line buffer）大小是否超过了后端（backend）所能提供的最大行缓存大小
    def check_line_buffer(self):

        if self.NpuOp.NpuOpConv is True:
            kernel_h = self.NpuOp.NpuOpConvOp.KerH
            stride = self.NpuOp.NpuOpConvOp.StrideH

            input_block_address_list = self.npu_op_flow_block_address_list[0]['input_block_address_list']
            block_pad_list = self.npu_op_flow_block_address_list[0]['block_pad_list']
            all_line_buffer_size_kB = self.backend.max_line_buffer / self.tile_split_mode.split_num
            pad_h = block_pad_list[0] + block_pad_list[1]
            pad_w = block_pad_list[2] + block_pad_list[3]

            kernel_c = input_block_address_list[-1]
            block_w = input_block_address_list[3]
            # TODO 为什么？
            need_line_buffer_size_kB = kernel_c * (block_w + pad_w) * (kernel_h + pad_h + stride) / 1024

            if need_line_buffer_size_kB > all_line_buffer_size_kB:
                errer_info = "line_buffer_size overflow, need to split block in w axis"
                raise (Exception(errer_info))
        else:
            # todo check
            # kernel_c = input_block_address_list[-1]
            # block_w = input_block_address_list[3]
            # kernel_h = input_block_address_list[1]
            # need_line_buffer_size_kB = kernel_c * (block_w + pad_w) * (kernel_h + pad_h ) / 1024
            # if need_line_buffer_size_kB > all_line_buffer_size_kB:
            #     errer_info = "line_buffer_size overflow, need to split block in w axis"
            #     raise(Exception(errer_info))
            return

    # TODO tile_split memory 中的分割，block_split是sim中的分割?
    def tile_split(self):  # 数据块

        if not self.tile_split_mode:
            self.backend.get_tile_split_mode(self)
        th, tw, tc = self.tile_split_mode.h, self.tile_split_mode.w, self.tile_split_mode.c
        tile_num = self.tile_split_mode.split_num

        output_tensor_info = self.get_output_tensor_info()
        output_block_addr_l = output_tensor_info['block_address_list']
        oh, ow, oc = output_block_addr_l[1], output_block_addr_l[3], output_block_addr_l[5]
        ih, iw, ic = self.get_input_tensor_info()['block_address_list'][1], \
            self.get_input_tensor_info()['block_address_list'][3], \
            self.get_input_tensor_info()['block_address_list'][5]

        NpuOpFlow_nums = len(self.NpuOp.NpuOpFlow)
        if self.int32_out:
            NpuOpFlow_nums = 1

        npu_op_flow_tile_address_list = []
        for n in range(NpuOpFlow_nums):
            data = dict(input_tile_address_list=None,
                        output_tile_address_list=None,
                        tile_pad_list=None)
            npu_op_flow_tile_address_list.append(data)

        if isinstance(self.NpuOp.NpuOpFlow[-1], NpuResize) \
                or (len(self.NpuOp.NpuOpFlow) >= 2
                    and isinstance(self.NpuOp.NpuOpFlow[-2], NpuResize)
                    and isinstance(self.NpuOp.NpuOpFlow[-1], NpuConcat)):

            assert tw == 1

            if isinstance(self.NpuOp.NpuOpFlow[-1], NpuResize):
                scale = self.NpuOp.NpuOpFlow[-1].ScaleFactor
            else:
                scale = self.NpuOp.NpuOpFlow[-2].ScaleFactor
            # assert (scale) % 1 == 0 ,'resize scaling factor must be an integer'
            temp_tile_oh = int(oh / scale / th)
            temp_tile_oh_list = [temp_tile_oh for _ in range(th)]
            tile_h_start_list = [0 for _ in range(th)]

            tile_address_list = np.zeros([th, tw, tc, 6], dtype=np.int32)

            for i in range(int(oh / scale - temp_tile_oh * th)):
                temp_tile_oh_list[i] += 1

            tile_oh_list = [scale * h for h in temp_tile_oh_list]

            for i in range(th):
                tile_h_start_list[i] += sum(tile_oh_list[:i])

            for i in range(th):
                for j in range(tw):
                    for k in range(tc):
                        tile_address_list[i][j][k][0] = tile_h_start_list[i]
                        tile_address_list[i][j][k][1] = tile_oh_list[i]

                        tile_address_list[i][j][k][2] = 0
                        tile_address_list[i][j][k][3] = ow

                        tile_address_list[i][j][k][4] = oc / tc * k
                        tile_address_list[i][j][k][5] = oc / tc * (k + 1)

        elif isinstance(self.NpuOp.NpuOpFlow[-1], NpuPool) \
                or isinstance(self.NpuOp.NpuOpFlow[-1], NpuActivation) \
                or isinstance(self.NpuOp.NpuOpFlow[-1], NpuElemWise) \
                or isinstance(self.NpuOp.NpuOpFlow[-1], NpuConcat) \
                or isinstance(self.NpuOp.NpuOpFlow[-1], NpuConv2d):

            new_h = math.floor(oh / th)
            h_nums = [new_h for i in range(th)]
            for i in range(int(oh - th * new_h)):
                h_nums[i] += 1

            new_w = math.floor(ow / tw)
            w_nums = [new_w for i in range(tw)]
            for i in range(int(ow - tw * new_w)):
                w_nums[i] += 1

            new_c = math.floor(ic / tc)
            c_nums = [new_c for i in range(tc)]
            for i in range(int(ic - tc * new_c)):
                c_nums[i] += 1

            tile_address_list = np.zeros([th, tw, tc, 6], dtype=np.int32)

            tem_h = []
            tem_w = []
            tem_c = []

            tem = 0
            for i in range(th):
                tem_h.append(tem)
                tem += h_nums[i]

            tem = 0
            for i in range(tw):
                tem_w.append(tem)
                tem += w_nums[i]

            tem = 0
            for i in range(tc):
                tem_c.append(tem)
                tem += c_nums[i]

            for i in range(th):
                for j in range(tw):
                    for k in range(tc):
                        tile_address_list[i][j][k][0] = tem_h[i]
                        tile_address_list[i][j][k][1] = h_nums[i]

                        tile_address_list[i][j][k][2] = tem_w[j]
                        tile_address_list[i][j][k][3] = w_nums[j]

                        if self.NpuOp.NpuOpConv:
                            tile_address_list[i][j][k][4] = 0
                            tile_address_list[i][j][k][5] = oc
                        else:
                            tile_address_list[i][j][k][4] = tem_c[k]
                            tile_address_list[i][j][k][5] = c_nums[k]

        else:
            raise NotImplementedError

        for f_j in range(NpuOpFlow_nums):
            flow_op_id = NpuOpFlow_nums - 1 - f_j
            op = self.NpuOp.NpuOpFlow[flow_op_id]
            op_block_address_list = self.npu_op_flow_block_address_list[flow_op_id]
            _npu_op_flow_tile_address_list = npu_op_flow_tile_address_list[flow_op_id]

            if isinstance(op, (NpuPool, NpuConv2d)):
                tile_address_list = op.update_tile_address_list(_npu_op_flow_tile_address_list,
                                                                tile_address_list,
                                                                op_block_address_list)

            if isinstance(op, NpuResize):
                assert tw == 1
                assert isinstance(self.NpuOp.NpuOpFlow[-1], NpuResize) \
                       or (isinstance(self.NpuOp.NpuOpFlow[-2], NpuResize)
                           and isinstance(self.NpuOp.NpuOpFlow[-1], NpuConcat))

                tile_address_list = op.update_tile_address_list(_npu_op_flow_tile_address_list,
                                                                tile_address_list)

            if isinstance(op, NpuConcat) \
                    or isinstance(op, NpuActivation) \
                    or isinstance(op, NpuElemWise) \
                    or isinstance(op, NpuPad):
                tile_address_list = op.update_tile_address_list(_npu_op_flow_tile_address_list,
                                                                tile_address_list)

        self.npu_op_flow_tile_address_list = npu_op_flow_tile_address_list

    def update_tile_output_vpu_shape_list(self):
        vpu_shape_list = []
        th, tw, tc = self.tile_split_mode.h, self.tile_split_mode.w, self.tile_split_mode.c
        vpu_shape_list = np.zeros([th, tw, tc, 6], dtype=np.int32)

        vpu_shape_output_list = self.npu_op_flow_tile_address_list[-1]['output_tile_address_list']
        for i in range(th):
            for j in range(tw):
                for k in range(tc):
                    vpu_shape_list[i][j][k] = vpu_shape_output_list[i][j][k]
        self.tile_output_vpu_shape_list = vpu_shape_list


class split_mode(object):
    h = None  # >=1 
    w = None  # >=1
    c = None  # >=1

    def __init__(self, h, w, c):
        self.h = h
        self.w = w
        self.c = c
        self.split_num = h * w * c


class tile_split_mode(split_mode):

    def __init__(self, h, w, c):
        super().__init__(h, w, c)
        if self.split_num not in [1, 2, 4]:
            raise ValueError("tile split num must in [1, 2, 4]")


class block_split_mode(split_mode):

    def __init__(self, h, w, c):
        super().__init__(h, w, c)


class npu_op_group(object):
    npu_op_list = None
    block_split_mode = None
    npu_op_group_id = None
    net_in_op_id_list = None
    net_out_op_id_list = None
    backend = None

    # npu_op_list：子图中的一组op
    def __init__(self, npu_op_list, block_split_mode, npu_op_group_id,
                 net_in_op_id_list=None, net_out_op_id_list=None, backend=None):
        self.net_in_op_id_list = net_in_op_id_list
        self.net_out_op_id_list = net_out_op_id_list
        self.npu_op_list = npu_op_list
        self.npu_op_group_id = npu_op_group_id
        self.block_split_mode = block_split_mode
        self.backend = backend
        self.block_list = []
        self.gen_block_list()

    def output_channel_padding(self, op):
        assert isinstance(op, block_param)
        npu_conv_op = op.get_npu_conv_op()
        if npu_conv_op is not None:
            weight = npu_conv_op.WeightValue
            k_n, k_h, k_w, k_c = weight.shape
            if k_n % 16 != 0:
                assert op.NpuOp.NpuOpMode == VpuPostOpSetMode.NONE
                assert op.output_block is True
                # n_k_n = math.ceil(k_n/16)*16
                # TODO 为什么从32开始，而不是16，
                if k_n < 32:
                    n_k_n = 32
                elif k_n < 64:
                    n_k_n = 64
                elif k_n < 128:
                    n_k_n = 128
                elif k_n < 256:
                    n_k_n = 256
                else:
                    n_k_n = math.ceil(k_n / 16) * 16  # 向上取整
                npu_conv_op.OutputC2cpu = deepcopy(npu_conv_op.OutputShape[0].C)
                npu_conv_op.OutputShape[0].C = n_k_n
                op.NpuOp.OutputShape[0].C = n_k_n
                return True
            else:
                npu_conv_op.OutputC2cpu = npu_conv_op.OutputShape[0].C

    def gen_block_list(self):  # gen通常是generate
        npu_op_block_address_list = self.block_split()
        for h_bld in range(self.block_split_mode.h):
            for w_bld in range(self.block_split_mode.w):
                for c_bld in range(self.block_split_mode.c):
                    for npu_op_idx, npu_op in enumerate(self.npu_op_list):
                        bp = block_param()
                        if len(self.npu_op_list) > 1:
                            bp.layer_group_flag = True
                        bp.backend = self.backend
                        # TODO TimeStep是排序？
                        bp.npu_op_id = npu_op.TimeStep
                        bp.npu_op_group_id = self.npu_op_group_id
                        bp.group_block_id = [h_bld, w_bld, c_bld]
                        bp.block_split_mode = self.block_split_mode
                        bp.set_npu_op(npu_op)  # 为bp设置npu op
                        bp.dma_read = True if npu_op.fmi_from_global_memory else False
                        bp.input_block = True if bp.NpuOp.NpuOpId in self.net_in_op_id_list else False  # 标记包含输入的block
                        bp.output_block = True if bp.NpuOp.NpuOpId in self.net_out_op_id_list else False  # 标记包含输出的block
                        block_resplit_flag = self.output_channel_padding(bp)  # 对卷积输出的通道数有所调整
                        if block_resplit_flag:
                            npu_op_block_address_list = self.block_split()
                        bp.npu_op_flow_block_address_list = []

                        if c_bld == self.block_split_mode.c - 1:
                            # 输出精度不应该是int32
                            bp.int32_out = False
                            if self.block_split_mode.c > 1:
                                bp.npu_psum_add = True  # TODO 参数聚合？
                                _npu_op = deepcopy(bp.NpuOp)

                                if _npu_op.NpuOpMode == VpuPostOpSetMode.ACTIVATION:
                                    _npu_op.NpuOpMode = VpuPostOpSetMode.ELW_ACTIVATION
                                    bp.NpuOp = _npu_op

                                elif bp.NpuOp.NpuOpMode == VpuPostOpSetMode.POOL:
                                    _npu_op.NpuOpMode = VpuPostOpSetMode.ELW_POOL
                                    bp.NpuOp = _npu_op

                                elif bp.NpuOp.NpuOpMode == VpuPostOpSetMode.NONE:
                                    _npu_op.NpuOpMode = VpuPostOpSetMode.ELW
                                    bp.NpuOp = _npu_op

                                else:
                                    raise NotImplementedError

                        else:
                            bp.int32_out = True
                            bp.int8_out = False

                            if bp.NpuOp.NpuOpMode:
                                _npu_op = deepcopy(bp.NpuOp)
                                _npu_op.NpuOpMode = VpuPostOpSetMode.NONE
                                bp.NpuOp = _npu_op

                            elif bp.NpuOp.NpuOpMode == VpuPostOpSetMode.NONE:
                                pass

                            else:
                                raise NotImplementedError

                        for npu_flow_op_id, npu_flow_op in enumerate(npu_op_block_address_list[npu_op_idx]):
                            data = dict()
                            tensor_info = dict(tensor_id=None, group_block_id=None, tensor_split=False)
                            tensor_info['group_block_id'] = bp.group_block_id

                            if self.block_split_mode.split_num > 1:
                                data["tensor_split"] = True

                            data['input_block_address_list'] = npu_flow_op['input_block_address_list'][h_bld][w_bld][
                                c_bld]

                            if npu_flow_op_id == 0:
                                tensor_info['tensor_id'] = npu_op.fmi_tensor[0]
                                data["input_tensor_info"] = deepcopy(tensor_info)

                            data['output_block_address_list'] = npu_flow_op['output_block_address_list'][h_bld][w_bld][
                                c_bld]

                            if npu_flow_op_id == len(npu_op_block_address_list[npu_op_idx]) - 1:
                                tensor_info['tensor_id'] = npu_op.fmo_tensor[0]
                                data["output_tensor_info"] = deepcopy(tensor_info)

                            if isinstance(npu_op.NpuOpFlow[npu_flow_op_id], NpuConcat):
                                tensor_info['tensor_id'] = npu_op.concat_input_tensor[0]
                                data["input1_tensor_info"] = deepcopy(tensor_info)
                                data['input1_block_address_list'] = \
                                    npu_flow_op['input1_block_address_list'][h_bld][w_bld][c_bld]

                            if not (isinstance(npu_op.NpuOpFlow[npu_flow_op_id], NpuActivation)
                                    or isinstance(npu_op.NpuOpFlow[npu_flow_op_id], NpuElemWise)
                                    or isinstance(npu_op.NpuOpFlow[npu_flow_op_id], NpuResize)
                                    or isinstance(npu_op.NpuOpFlow[npu_flow_op_id], NpuConcat)):
                                data['block_pad_list'] = npu_flow_op['block_pad_list'][h_bld][w_bld][c_bld]
                            bp.npu_op_flow_block_address_list.append(data)

                        if npu_op.NpuOpConcat:
                            bp.concat_input = True

                        if npu_op.NpuOpShortCutOut:
                            short_cut_out_tensor_id = npu_op.NpuOpShortCutOp.OutTensors[0]
                            if (short_cut_out_tensor_id !=
                                    bp.npu_op_flow_block_address_list[-1]["output_tensor_info"]['tensor_id']):
                                bp.short_cut_out = True

                        if npu_op.NpuOpElemWise:
                            bp.elemwise_input = True

                        bp.weight_mapping_init()
                        bp.tile_split()
                        bp.check_line_buffer()
                        bp.update_tile_output_vpu_shape_list()
                        self.block_list.append(bp)

    def block_split(self):

        npu_op_nums = len(self.npu_op_list)  # 子图中的op组
        npu_op_block_address_list = []
        for n in range(npu_op_nums):
            npu_op_block_address_list.append([])
            for j in range(len(self.npu_op_list[n].NpuOpFlow)):  # npu_op_block_address_list[块[块中每个op的字典信息]]
                data = dict(input_block_address_list=None,
                            output_block_address_list=None,
                            block_pad_list=None)
                npu_op_block_address_list[n].append(data)

        bh, bw, bc = self.block_split_mode.h, self.block_split_mode.w, self.block_split_mode.c

        oh, ow, oc = self.npu_op_list[-1].NpuOpFlow[-1].OutputShape[0].H, \
            self.npu_op_list[-1].NpuOpFlow[-1].OutputShape[0].W, \
            self.npu_op_list[-1].NpuOpFlow[-1].OutputShape[0].C

        if isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuPool) \
                or isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuActivation) \
                or isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuConcat) \
                or isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuElemWise) \
                or isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuConv2d) \
                or isinstance(self.npu_op_list[-1].NpuOpFlow[-1], NpuResize):

            new_h = math.floor(oh / bh)  # oh是原始数据高度，bh是块在h轴上的块数，因此oh / bh是每个块的数据高度
            h_nums = [new_h for _ in range(bh)]  # bh个new_h
            for i in range(int(oh - bh * new_h)):  # 不够一个block的那一部分h，范围[0, bh - 1]
                h_nums[i] += 1  # 分给前几个block

            new_w = math.floor(ow / bw)
            w_nums = [new_w for _ in range(bw)]
            for i in range(int(ow - bw * new_w)):
                w_nums[i] += 1

            block_address_list = np.zeros([bh, bw, bc, 6], dtype=np.int32)  # 6是6种参数，详情见后

            tem_h = []  # 每次合并后当此的h?
            tem_w = []
            tem_c = []

            tem = 0
            for i in range(bh):
                tem_h.append(tem)  # 当前block的高度
                tem += h_nums[i]

            tem = 0
            for i in range(bw):
                tem_w.append(tem)
                tem += w_nums[i]

            # tem = 0
            # for i in range(bc):
            #     tem_c.append(tem)
            #     tem += c_nums[i]

            for i in range(bh):
                for j in range(bw):
                    for k in range(bc):
                        block_address_list[i][j][k][0] = tem_h[i]  # 当前分块的h
                        block_address_list[i][j][k][1] = h_nums[i]  # 当前分块合并后的总h

                        block_address_list[i][j][k][2] = tem_w[j]
                        block_address_list[i][j][k][3] = w_nums[j]

                        block_address_list[i][j][k][4] = 0
                        block_address_list[i][j][k][5] = oc

        for n_i in range(npu_op_nums):  # npu_op_nums = len(self.npu_op_list)  # 子图中的op组
            npu_op_id = npu_op_nums - 1 - n_i  # 逆序，从后向前
            NpuOpFlow = self.npu_op_list[npu_op_id].NpuOpFlow
            NpuOpFlow_nums = len(NpuOpFlow)

            for f_j in range(NpuOpFlow_nums):
                flow_op_id = NpuOpFlow_nums - 1 - f_j
                op = NpuOpFlow[flow_op_id]
                _npu_op_block_address_list = npu_op_block_address_list[npu_op_id][flow_op_id]
                # TODO 返回block_address_list还是_npu_op_block_address_list？
                block_address_list = op.update_block_address_list(_npu_op_block_address_list, block_address_list)

        return npu_op_block_address_list
