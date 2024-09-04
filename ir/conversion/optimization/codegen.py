# from backend.ada200.ada200 import ada200
from ir.conversion.optimization.ir_transform import _register_ir_transformation_rule
from ir.codegen.vpuPreparingParam import *
from ir.codegen.getRegisterWithConv import getRegisterWithConv
from ir.codegen.getRegisterWithoutConv import getRegisterWithoutConv
from enum import Enum
from tool.utils import *
import copy
import cv2

import json
import datetime

PRE_REGISTER_NUMS = 67
CIM_REGISTER_NUMS = 13
VPU_REGISTER_NUMS = 300
FE_REGISTER_NUMS = 8
BE_REGISTER_NUMS = 14
WGT_REGISTER_NUMS = 3
REGISTER_NUMS = PRE_REGISTER_NUMS + CIM_REGISTER_NUMS + VPU_REGISTER_NUMS + FE_REGISTER_NUMS + BE_REGISTER_NUMS + WGT_REGISTER_NUMS
np.random.seed(20230510)


class TransformRule(Enum):
    CREATE_REGISTER = 1
    CREATE_INFO = 2
    CREATE_XLSX = 3
    CREATE_TOP_INFO = 4
    CALCULATING_THE_WEIGHT = 5
    CREATE_REGISTER_WITH_DMA = 6


def intToBin(number, index, feature=True):
    # index is the bit width of the data and number is the data to be converted. 
    # If feature is True, decimal to binary (complement code) is performed; 
    # if feature is False, binary to decimal is performed. 
    assert (isinstance(number, int)
            or isinstance(number, np.int32)
            or isinstance(number, np.int64)
            or isinstance(number, np.int8)), 'the type of number must be int'
    if feature is True:
        if number >= 0:

            b = bin(number)
            b = '0' * (index + 2 - len(b)) + b
        else:
            b = 2 ** index + number
            b = bin(b)
            b = '1' * (index + 2 - len(b)) + b
        b = b.replace("0b", '')
        b = b.replace('-', '')
        assert (len(b) == index), "out of bitnums"
        return b
    elif feature is False:
        i = int(str(number), 2)
        if i >= 2 ** (index - 1):
            i = -(2 ** index - i)
            return i
        else:
            return i


def bin_listTobin(bin_list):
    bin = ""
    for i in bin_list:
        bin += i
    assert (len(bin) == 32), f"value of bitnums {len(bin)}==32"
    return bin


def binTohex(binNums, bit_nums):
    nums = bit_nums / 4
    res = hex(int(binNums, 2))
    res = res.split("0x")[-1]
    if len(res) != nums:
        res = "0" * int(nums - len(res)) + res
    return res


def get_list(nums=22, val=0):
    l = []
    for _ in range(nums):
        l.append(val)
    return l


def get_all_res(npuop, path, nn_sb, _weight_,
                sub_block_register_list=None,
                input_output_dict=None,
                weight_base_addr=None,
                SubBlockWeightInfo=None,
                dma_read_list=None):
    if npuop.NpuOp.NpuOpConv:
        sub_block_nums = getRegisterWithConv(npuop, path, nn_sb, _weight_,
                                             sub_block_register_list,
                                             input_output_dict,
                                             weight_base_addr,
                                             SubBlockWeightInfo,
                                             dma_read_list)
    else:
        sub_block_nums = getRegisterWithoutConv(npuop, path, nn_sb, _weight_,
                                                sub_block_register_list,
                                                input_output_dict)
        # sub_block_nums = 1
    return sub_block_nums


def read_file(path, file_list):
    with open(path, 'r') as f:
        line = f.readline()
        file_list.append(line)
        while line:
            line = f.readline()
            if line != '':
                file_list.append(line)


def read_files(path):
    file_list = []
    read_file(f'{path}/pre_param', file_list)
    read_file(f'{path}/cim_cluster_param', file_list)
    read_file(f'{path}/vpu_param', file_list)
    return file_list


def compare_file(file_list_src, file_list_dst):
    file_list_res = []

    for i in range(REGISTER_NUMS):
        if file_list_src[i] != file_list_dst[i]:
            file_list_res.append(file_list_dst[i])
            file_list_src[i] = file_list_dst[i]

    return file_list_res


def transfer_register_format(res_list):
    res_list_ = []
    for i in res_list:
        res_list_tem = []
        for j in i:
            res_list_tem.append(f"{j.split(' ')[0][4:]}\n")
            res_list_tem.append(f"{j.split(' ')[1]}")
        res_list_.append(res_list_tem)
    return res_list_


def get_block_head(file_list, npu_graph, sub_block_id, all_head_list, dma_read_list):
    SubBlockWeightInfo = npu_graph.SubBlockWeightInfo
    head_list = ['ffffffff']
    # block ctrl reg0
    register = [intToBin(sub_block_id, 16),
                intToBin(len(file_list), 16)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    # block ctrl reg1
    if sub_block_id == len(SubBlockWeightInfo) - 1:
        load_wgt_en = 0
        wgt_st_addr = 0
    else:
        if SubBlockWeightInfo[sub_block_id][1:] == SubBlockWeightInfo[sub_block_id + 1][1:]:
            load_wgt_en = 0
            wgt_st_addr = 0
        else:
            load_wgt_en = 1
            wgt_st_addr = SubBlockWeightInfo[sub_block_id + 1][1]
    register = [intToBin(load_wgt_en, 1),
                intToBin(wgt_st_addr, 31)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    # block ctrl reg2
    register = []
    if load_wgt_en:
        register.append(intToBin(SubBlockWeightInfo[sub_block_id + 1][2], 32))
    else:
        register.append(intToBin(0, 32))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    # block ctrl reg3
    if sub_block_id == len(SubBlockWeightInfo) - 1 and SubBlockWeightInfo[sub_block_id][0] is True:
        fmo_output_en = 1
    elif (SubBlockWeightInfo[sub_block_id][0] != SubBlockWeightInfo[sub_block_id + 1][0] and
          SubBlockWeightInfo[sub_block_id][0] is True):
        fmo_output_en = 1
    else:
        fmo_output_en = 0

    register = [intToBin(fmo_output_en, 1),
                intToBin(6144, 31)]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)

    # block ctrl reg4
    register = [intToBin(SubBlockWeightInfo[sub_block_id][4][1] * 8, 16),
                intToBin(int(SubBlockWeightInfo[sub_block_id][4][0] *
                             SubBlockWeightInfo[sub_block_id][4][2] / 2 / 8), 16)]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    # all_head_list.append(head_list)

    # block ctrl reg5
    flag = 0
    flag_id = 0
    fmi_st_addr = 0
    for i in range(len(dma_read_list)):
        if dma_read_list[i][1] == sub_block_id + 1:
            flag_id = i
            flag = 1
            if len(dma_read_list[:flag_id]) > 0:
                hw_list = []
                for d in dma_read_list[:flag_id]:
                    fmi_shape = d[-1]
                    h, w = fmi_shape[1], fmi_shape[3]
                    hw = h * w
                    if hw % 2 != 0:
                        hw += 1
                    hw_list.append(hw)
                fmi_st_addr += sum(hw_list)
    register = [intToBin(flag, 1),
                intToBin(fmi_st_addr, 31)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    # all_head_list.append(head_list)

    # block ctrl reg6
    register = []
    if flag:
        w = int(dma_read_list[flag_id][3][3])
        h = int(dma_read_list[flag_id][3][1])
        hw = h * w
        if hw % 2 == 0:
            pass
        else:
            hw += 1

        h = int(hw / 2)
        w = 2

        register.append(intToBin(w, 16))
        register.append(intToBin(h, 16))
    else:
        register.append(intToBin(int(0), 16))
        register.append(intToBin(int(0), 16))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    head_list.append(register)
    all_head_list.append(head_list)
    pass


def get_top_head(npu_graph):
    top_head = []
    SubBlockWeightInfo = npu_graph.SubBlockWeightInfo
    # (frame header)
    register = [intToBin(65520, 16),
                intToBin(len(SubBlockWeightInfo), 16)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    top_head.append(register)
    # (block ctrl reg0)
    register = [intToBin(1, 1),
                intToBin(0, 31)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    top_head.append(register)
    # (block ctrl reg1)
    register = []
    if len(npu_graph.dma_read_list) > 0:
        h = npu_graph.dma_read_list[0][3][1]
        w = npu_graph.dma_read_list[0][3][3]

        # input padding as even
        hw = h * w
        if hw % 2 == 0:
            pass
        else:
            hw += 1
        h = int(hw / 2)
        w = 2

    else:
        h = 0
        w = 0
    register.append(intToBin(w, 16))
    register.append(intToBin(h, 16))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    top_head.append(register)
    # (frame ctrl reg2)
    register = [intToBin(1, 1),
                intToBin(0, 31)
                ]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    top_head.append(register)
    # (frame ctrl reg3)
    register = [intToBin(SubBlockWeightInfo[0][2], 32)]
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    top_head.append(register)

    return top_head


def get_WGT_register(file_list, npu_graph):
    # ADDR_FE_BASE+16’h00
    register = []
    # wgt_endian
    register.append(n_zeros_str(27))
    register.append(intToBin(0, 1))
    # wgt_clk_inv_en
    register.append(n_zeros_str(2))
    register.append(intToBin(0, 1))
    # wgt_clr
    register.append(intToBin(0, 1))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c040 {register}\n")

    # ADDR_FE_BASE+16’h04
    register = []
    # wgt_shm_base_addr
    register.append(n_zeros_str(13))
    register.append(intToBin(65536, 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c044 {register}\n")

    # ADDR_FE_BASE+16’h08
    register = []
    # Status rigster
    register.append(intToBin(0, 32))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c048 {register}\n")
    pass


def get_FE_register(file_list, npu_graph, sub_block_id, dma_read_list):
    flag = 0
    flag_id = 0
    for i in range(len(dma_read_list)):
        if dma_read_list[i][1] == sub_block_id + 1:
            flag_id = i
            flag = 1

    if flag:
        w = int(dma_read_list[flag_id][3][3])
        h = int(dma_read_list[flag_id][3][1])

    else:
        _block_list = npu_graph.AllOps[0].block_list[0]
        _block_address_list = _block_list.npu_op_flow_block_address_list[0]
        input_block_address_list = _block_address_list["input_block_address_list"]
        h = input_block_address_list[1]
        w = input_block_address_list[3]

    if h == w == 128:
        padding_h = 128
        padding_w = 128

    else:
        hw = h * w
        if hw % 2 == 0:
            pass
        else:
            hw += 1

        padding_h = int(hw / 2)
        padding_w = 2

    # ADDR_FE_BASE+16’h00
    register = []
    # fe_shm_2x_pix_en
    register.append(n_zeros_str(3))
    register.append(intToBin(0, 1))
    # shm_ts
    register.append(n_zeros_str(3))
    register.append(intToBin(0, 1))
    # fe_bar_size
    register.append(intToBin(51, 8))
    # fe_soft_reset
    register.append(n_zeros_str(3))
    register.append(intToBin(1, 1))
    # fe_cmos_pclk_inv_en
    register.append(n_zeros_str(3))
    register.append(intToBin(0, 1))
    # fe_dat_mux
    register.append(n_zeros_str(1))
    register.append(intToBin(0, 3))
    # fe_mode
    register.append(n_zeros_str(2))
    register.append(intToBin(0, 2))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005800 {register}\n")

    # ADDR_FE_BASE+16’h04
    register = []
    # fe_rand_feedback_bit
    register.append(intToBin(14, 8))
    # fe_rand_seed
    register.append(intToBin(7947555, 24))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005804 {register}\n")

    # ADDR_FE_BASE+16’h08
    register = []
    # Shm_wdat64_byte7
    register.append(n_zeros_str(1))
    register.append(intToBin(7, 3))
    # Shm_wdat64_byte6
    register.append(n_zeros_str(1))
    register.append(intToBin(2, 3))
    # Shm_wdat64_byte5
    register.append(n_zeros_str(1))
    register.append(intToBin(5, 3))
    # Shm_wdat64_byte4
    register.append(n_zeros_str(1))
    register.append(intToBin(4, 3))
    # Shm_wdat64_byte3
    register.append(n_zeros_str(1))
    register.append(intToBin(7, 3))
    # Shm_wdat64_byte2
    register.append(n_zeros_str(1))
    register.append(intToBin(1, 3))
    # Shm_wdat64_byte1
    register.append(n_zeros_str(1))
    register.append(intToBin(0, 3))
    # Shm_wdat64_byte0
    register.append(n_zeros_str(1))
    register.append(intToBin(3, 3))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005808 {register}\n")

    # ADDR_FE_BASE+16’h0c
    register = []
    # block0_base_waddr
    register.append(n_zeros_str(13))
    register.append(intToBin(81920, 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000580c {register}\n")

    # ADDR_FE_BASE+16’h10
    register = []
    # block1_base_waddr
    register.append(n_zeros_str(13))
    register.append(intToBin(98304, 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005810 {register}\n")

    # ADDR_FE_BASE+16’h14
    register = []
    # Block0_height
    height = padding_h
    register.append(n_zeros_str(3))
    register.append(intToBin(height, 13))
    # Block0_width
    width = padding_w
    register.append(n_zeros_str(3))
    register.append(intToBin(width, 13))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005814 {register}\n")

    # ADDR_FE_BASE+16’h18
    register = []
    # Block0_height
    height = padding_h
    register.append(n_zeros_str(3))
    register.append(intToBin(height, 13))
    # Block0_width
    width = padding_w
    register.append(n_zeros_str(3))
    register.append(intToBin(width, 13))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h00005818 {register}\n")

    # ADDR_FE_BASE+16’h1c
    register = []
    # Status register
    register.append(n_zeros_str(32))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000581c {register}\n")


def get_BE_register(file_list, npu_graph, sub_block_id):
    SubBlockWeightInfo = npu_graph.SubBlockWeightInfo
    shm_base_addr = SubBlockWeightInfo[sub_block_id][3]
    shape = SubBlockWeightInfo[sub_block_id][4]
    # ADDR_FE_BASE+16’h00
    register = []
    # Status register
    register.append(n_zeros_str(3))
    register.append(intToBin(4, 3))
    # vpu_output_en
    if (sub_block_id == len(SubBlockWeightInfo) - 1
            and SubBlockWeightInfo[sub_block_id][0] == True):
        fmo_output_en = 1
    elif (SubBlockWeightInfo[sub_block_id][0] != SubBlockWeightInfo[sub_block_id + 1][0]
          and SubBlockWeightInfo[sub_block_id][0] == True):
        fmo_output_en = 1
    else:
        fmo_output_en = 0
    register.append(intToBin(fmo_output_en, 1))
    # soft_rst_n
    register.append(intToBin(1, 1))
    # de_polar
    register.append(n_zeros_str(1))
    register.append(intToBin(0, 1))
    # hsync_polar
    register.append(intToBin(0, 1))
    # vsync_polar
    register.append(intToBin(0, 1))
    # pixel_clk_pahse
    register.append(n_zeros_str(3))
    register.append(intToBin(1, 5))
    # pixel_clk_div
    register.append(n_zeros_str(3))
    register.append(intToBin(8, 5))
    # pixel_clk_en
    register.append(intToBin(1, 1))
    # pixel_clk_en
    register.append(intToBin(0, 3))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c000 {register}\n")

    # ADDR_BE_BASE+16’h04
    register = []
    # be_shm_base_addr0
    register.append(n_zeros_str(13))
    register.append(intToBin(shm_base_addr[0], 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c004 {register}\n")

    # ADDR_BE_BASE+16’h08
    register = []
    # be_shm_fmo_size0
    register.append(n_zeros_str(13))
    register.append(intToBin(int(shape[0] * shape[1] * shape[2] / 2 / 16), 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c008 {register}\n")

    # ADDR_BE_BASE+16’h20
    register = []
    # be_shm_base_addr0
    register.append(n_zeros_str(13))
    register.append(intToBin(shm_base_addr[1], 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c020 {register}\n")

    # ADDR_BE_BASE+16’h24
    register = []
    # be_shm_fmo_size0
    register.append(n_zeros_str(13))
    register.append(intToBin(int(shape[0] * shape[1] * shape[2] / 2 / 16), 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c024 {register}\n")

    # ADDR_BE_BASE+16’h28
    register = []
    # be_shm_base_addr0
    register.append(n_zeros_str(13))
    register.append(
        intToBin(shm_base_addr[2], 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c028 {register}\n")

    # ADDR_BE_BASE+16’h2c
    register = []
    # be_shm_fmo_size0
    register.append(n_zeros_str(13))
    register.append(intToBin(int(shape[0] * shape[1] * shape[2] / 2 / 16), 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c02c {register}\n")

    # ADDR_BE_BASE+16’h30
    register = []
    # be_shm_base_addr0
    register.append(n_zeros_str(13))
    register.append(intToBin(shm_base_addr[3], 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c030 {register}\n")

    # ADDR_BE_BASE+16’h34
    register = []
    # be_shm_fmo_size0
    register.append(n_zeros_str(13))
    register.append(intToBin(int(shape[0] * shape[1] * shape[2] / 2 / 16), 19))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c034 {register}\n")

    # ADDR_BE_BASE+16’h0c
    register = []
    # disp_all_height
    register.append(n_zeros_str(3))
    register.append(intToBin(int(shape[0] * shape[2] / 2 / 8) + 2, 13))
    # disp_all_width
    register.append(intToBin(shape[1] * 8 + 2, 16))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c00c {register}\n")

    # ADDR_BE_BASE+16’h10
    register = []
    # disp_fmo_height
    register.append(n_zeros_str(3))
    register.append(intToBin(int(shape[0] * shape[2] / 2 / 8), 13))
    # disp_fmo_width
    register.append(intToBin(shape[1] * 8, 16))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c010 {register}\n")

    # ADDR_BE_BASE+16’h14
    register = []
    # Disp_hsync_width
    register.append(intToBin(1, 16))
    # Disp_hsync_1st
    register.append(intToBin(0, 16))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c014 {register}\n")

    # ADDR_BE_BASE+16’h18
    register = []
    # Disp_vsync_height
    register.append(n_zeros_str(3))
    register.append(intToBin(1, 13))
    # Disp_hsync_1st
    register.append(n_zeros_str(3))
    register.append(intToBin(0, 13))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c018 {register}\n")

    # ADDR_BE_BASE+16’h1c
    register = []
    # status
    register.append(intToBin(0, 32))
    register = bin_listTobin(register)
    register = binTohex(register, 32)
    file_list.append(f"32'h0000c01c {register}\n")
    pass


def make_weight(top_info_path, npu_graph):
    weight_path = f'{top_info_path}/weight'
    weight = npu_graph.WeightTensors
    f = open(weight_path, 'w')
    for i in weight:
        for j in i:
            f.write(j + '\n')
    f.close()
    pass


def make_image_to_memory(top_info_path, image_size=[128, 128, 3], image_path=''):
    top_info_path = f'{top_info_path}/image'
    if (image_path == ''):
        image = (np.random.rand(image_size[0] * image_size[1] * image_size[2]).reshape(image_size) * 255 - 128).astype(
            np.int8)
    else:
        try:
            image = cv2.imread(image_path)
        except:
            assert (0), "image is not exist"

    f = open(top_info_path, '+w')
    string = ''
    n = 0
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            for z in range(image_size[2]):
                number = intToBin(image[i][j][z], 8, feature=True)
                number = binTohex(number, 8)
                string = number + string
                n += 1
                if n == 3:
                    f.write(string + '\n')
                    n = 0
                    string = ''
    f.close()
    pass


def gather_register(res_list, top_head, head_list, finally_path):
    f = open(finally_path, 'w')
    for i in range(len(top_head)):
        f.write(top_head[i] + '\n')
    for i in range(len(res_list)):
        for j in range(len(head_list[i])):
            f.write(head_list[i][j] + '\n')
        for j in range(len(res_list[i])):
            f.write(res_list[i][j])
    f.close()
    pass


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (int(round(dw - 0.1)), int(round(dh - 0.1)))


def make_image(image_path, Scale, ZeroPoint, size=[416, 416, 3]):
    im = cv2.imread(image_path)
    shape = im.shape
    im, ratio, (dw, dh) = letterbox(im, size, stride=32, auto=False)
    im = im.transpose((2, 0, 1))[::-1].transpose((1, 2, 0))
    im = np.ascontiguousarray(im)

    # im = im[..., ::-1]  # conver RGB to BGR
    x = im.astype(np.float32)
    x /= 255
    x = (x / Scale + ZeroPoint).astype(np.int8)
    return x


def fill_zero(block_tensor):
    h, w, c = block_tensor.shape
    tem_tensor = np.zeros(shape=[h, w, 8]).astype(np.int8)
    tem_tensor[:, :, 2] = block_tensor[:, :, 2]
    tem_tensor[:, :, 1] = block_tensor[:, :, 1]
    tem_tensor[:, :, 0] = block_tensor[:, :, 0]
    return tem_tensor


def Array2Txt_hwc_hex_bank(x, in_bit=8, out_bit=64, path_="", bank_start=0):
    n = 0
    path = path_
    for i in range(len(x)):
        f = open(path, "w")
        string = ""
        for j in range(len(x[i])):
            number = intToBin(x[i][j], in_bit, feature=True)
            number = binTohex(number, int(out_bit / in_bit))
            string = number + string
            n += 1
            if n == 8:
                string += "\n"
                f.write(string)
                string = ""
                n = 0
        f.close()


@_register_ir_transformation_rule(TransformRule.CREATE_REGISTER)
def create_register(npu_graph):
    assert npu_graph.codegen_path is not None
    path = npu_graph.codegen_path
    # path = "/data/tinynpu/test/model/yolov3_res"
    npu_op_block_list = []
    for op_group in npu_graph.AllOps:
        npu_op_block_list.extend(op_group.block_list)

    register_data_list = []
    sub_block_nums = {}
    del_dir(path)
    register_data_list = []
    sub_block_nums_list = []
    input_output_dict = {"input": [], "output": []}

    SubBlockWeightInfo = []
    dma_read = False
    dma_read_list = []
    for i in range(len(npu_op_block_list)):
        i = i

        if not os.path.exists(f"{path}/"):
            os.makedirs(f"{path}/")
        sub_block_register_list = []

        sub_block_nums.setdefault(npu_op_block_list[i].npu_op_group_id, 0)
        if npu_op_block_list[i].dma_read:
            dma_read = True
        sub_block_nums[npu_op_block_list[i].npu_op_group_id] += get_all_res(npu_op_block_list[i], f"{path}/",
                                                                            sub_block_nums[
                                                                                npu_op_block_list[i].npu_op_group_id],
                                                                            npu_graph.get_weight_tensor(
                                                                                npu_op_block_list[i].npu_op_id),
                                                                            sub_block_register_list, input_output_dict,
                                                                            npu_graph.get_weight_base_addr(
                                                                                npu_op_block_list[i].npu_op_id),
                                                                            SubBlockWeightInfo, dma_read_list)

        register_data_list.append(sub_block_register_list)

    input_output_dict["input"] = list(set(input_output_dict["input"]))
    input_output_dict["output"] = list(set(input_output_dict["output"]))
    npu_graph.SubBlockWeightInfo = SubBlockWeightInfo
    npu_graph.input_output_dict = input_output_dict
    npu_graph.sub_block_nums = sub_block_nums
    npu_graph.register_data_list = register_data_list
    npu_graph.dma_read = dma_read
    npu_graph.dma_read_list = dma_read_list
    pass


@_register_ir_transformation_rule(TransformRule.CREATE_INFO)
def create_info(npu_graph):
    assert npu_graph.codegen_path is not None
    path = npu_graph.codegen_path
    model_name = npu_graph.model_name
    sub_block_nums = npu_graph.sub_block_nums
    total_layer_num = len(sub_block_nums)
    input_output_dict = npu_graph.input_output_dict
    dma_read = npu_graph.dma_read
    npu_op_block_list = []
    for op_group in npu_graph.AllOps:
        npu_op_block_list.extend(op_group.block_list)
    if dma_read:
        text = f"{model_name} , {total_layer_num}, {len(input_output_dict['input'])}, {len(input_output_dict['output'])}, {1}\n"
    else:
        text = f"{model_name} , {total_layer_num}, {len(input_output_dict['input'])}, {len(input_output_dict['output'])}\n"
    for i in range(total_layer_num):
        text += f"{sub_block_nums[i]}, "
    text += "\n"
    for i in input_output_dict['input']:
        text += f"{i}, "
    text += "\n"
    for i in input_output_dict['output']:
        text += f"{i}, "
    text += "\n"
    for i in range(total_layer_num):
        text += f"1, "
    text = text[:-2]

    f = open(f"{path}/{model_name}.txt", "w")
    f.write(text)
    f.close()

    XM_info = {}
    XM_info["model_name"] = model_name
    XM_info["total_layer_num"] = total_layer_num
    XM_info["input_tensor_num"] = len(input_output_dict['input'])
    XM_info["output_tensor_num"] = len(input_output_dict['output'])
    XM_info["subblock_num"] = sub_block_nums
    XM_info["input_tensor_id"] = input_output_dict['input']
    XM_info["output_tensor_id"] = input_output_dict['output']
    XM_info["layer_enable_arrs"] = [1] * total_layer_num
    NetOutNpuOpId = npu_graph.NetOutNpuOpId
    origin_shape_list = []
    real_shape_list = []
    for i in range(len(input_output_dict['output'])):
        op = npu_graph.get_npu_op(NetOutNpuOpId[i])
        origin_shape_list.append([int(op.OutputH), int(op.OutputW), int(op.OutputC)])
        real_shape_list.append([int(op.OutputH), int(op.OutputW), int(op.NpuOpConvOp.OutputC2cpu)])

    XM_info["origin_shape_list"] = origin_shape_list
    XM_info["real_shape_list"] = real_shape_list
    dma_read_list = npu_graph.dma_read_list
    if len(dma_read_list) > 0:
        dma_read_shape_list = []
        for i in range(len(dma_read_list)):
            dma_read_shape = dma_read_list[i][3]
            dma_read_shape_list.append(dma_read_shape.tolist())
        XM_info["dma_read_shape_list"] = dma_read_shape_list
        XM_info["dma_read_fmi"] = [int(dma_read_list[0][2][0]), int(dma_read_list[0][2][1]),
                                   int(dma_read_list[0][2][2])]
        with open(f"{path}/{model_name}.json", 'w') as f:
            json.dump(XM_info, f)
        pass


@_register_ir_transformation_rule(TransformRule.CREATE_XLSX)
def create_info(npu_graph):
    path_base = npu_graph.codegen_path
    group_list = npu_graph.AllOps
    block_list = group_to_block(group_list)
    block_param_list = get_block_param_list(block_list)
    block_param_df = graph_to_df(block_param_list)
    model_name = npu_graph.model_name
    block_param_df.to_excel(f"{path_base}/{model_name}.xlsx", index=False)


@_register_ir_transformation_rule(TransformRule.CREATE_TOP_INFO)
def create_info(npu_graph):
    path_base = npu_graph.codegen_path
    top_info_path = f'{path_base}_top_info'
    if not os.path.exists(top_info_path):
        os.makedirs(top_info_path)
    make_weight(top_info_path, npu_graph)
    make_image_to_memory(top_info_path, image_size=[128, 128, 3], image_path='')

    sub_block_nums = npu_graph.sub_block_nums
    total_layer_num = len(sub_block_nums)
    tem_file_list = [''] * REGISTER_NUMS
    res_list = []
    total_sub_block_nums = 0
    head_list = []
    for i in range(total_layer_num):
        for j in range(sub_block_nums[i]):
            model_path = f"{path_base}/layer_{i}/sub_block_{j}"
            file_list = read_files(model_path)

            get_FE_register(file_list, npu_graph, total_sub_block_nums, npu_graph.dma_read_list)
            get_BE_register(file_list, npu_graph, total_sub_block_nums)
            get_WGT_register(file_list, npu_graph)

            file_list_res = compare_file(tem_file_list, file_list)
            get_block_head(file_list_res, npu_graph, total_sub_block_nums, head_list, npu_graph.dma_read_list)
            res_list.append(file_list_res)
            # res_list.append(compare_file(tem_file_list, file_list))
            tem_file_list = copy.deepcopy(file_list)
            total_sub_block_nums += 1
    assert (total_sub_block_nums == len(res_list))
    res_list = transfer_register_format(res_list)
    top_head = get_top_head(npu_graph)
    finally_path = f'{top_info_path}/register'
    gather_register(res_list, top_head, head_list, finally_path)

    getZipDir(f"{top_info_path}/", f"{top_info_path}.zip")
    pass


@_register_ir_transformation_rule(TransformRule.CALCULATING_THE_WEIGHT)
def calculating_the_weight(npu_graph):
    WeightBaseAddress = [0]
    sum = 0
    for i in range(len(npu_graph.WeightTensors) - 1):
        WeightBaseAddress.append(sum + len(npu_graph.WeightTensors[i]))
        sum += len(npu_graph.WeightTensors[i])
    npu_graph.WeightBaseAddress = WeightBaseAddress
    pass


@_register_ir_transformation_rule(TransformRule.CREATE_REGISTER_WITH_DMA)
def calculating_the_weight(npu_graph):
    if npu_graph.dma_read:
        dma_read_list = npu_graph.dma_read_list
        codegen_path = npu_graph.codegen_path
        memory_config_path = f'{codegen_path}/memory_config.txt'
        memory_config = open(memory_config_path, 'w')
        Scale = npu_graph.get_tensor([npu_graph.NetInTensors]).Scale
        ZeroPoint = npu_graph.get_tensor([npu_graph.NetInTensors]).ZeroPoint
        image_path = npu_graph.image_path
        image = make_image(image_path, Scale, ZeroPoint, [416, 416, 3])

        for i in range(len(dma_read_list)):
            memory_path = f'{codegen_path}/{dma_read_list[i][1]}'
            memory_config.write(f'{dma_read_list[i][1]}\n')
            dma_read_shape = dma_read_list[i][3]
            dma_read_tem = image[dma_read_shape[0]:dma_read_shape[1] + dma_read_shape[0],
                           dma_read_shape[2]:dma_read_shape[3] + dma_read_shape[2],
                           dma_read_shape[4]:dma_read_shape[5] + dma_read_shape[4]]
            print([dma_read_shape[0], dma_read_shape[1] + dma_read_shape[0], dma_read_shape[2],
                   dma_read_shape[3] + dma_read_shape[2],
                   dma_read_shape[4], dma_read_shape[5] + dma_read_shape[4]])
            dma_read_tem = fill_zero(dma_read_tem)
            dma_read_tem = [dma_read_tem.reshape(-1).tolist()]
            Array2Txt_hwc_hex_bank(dma_read_tem, 8, 64, memory_path, 1)
        memory_config.close()
    path_base = npu_graph.codegen_path

    json_path = f'{os.path.dirname(path_base)}/model_version.json'
    try:
        with open(json_path, "r") as file:
            model_version = json.load(file)
    except:
        model_version = {npu_graph.model_name: npu_graph.model_name}
    try:
        formatted_time = model_version[npu_graph.model_name]
    except:
        formatted_time = npu_graph.model_name
    del_file(f"{path_base}_{formatted_time}.zip")

    delta = datetime.timedelta(hours=8, minutes=0)
    current_time = datetime.datetime.now() + delta

    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    model_version[npu_graph.model_name] = formatted_time
    with open(json_path, "w") as file:
        json.dump(model_version, file)
    getZipDir(f"{path_base}/", f"{path_base}_{formatted_time}.zip")

    pass


codegen_transform = [TransformRule.CALCULATING_THE_WEIGHT,
                     TransformRule.CREATE_REGISTER,
                     TransformRule.CREATE_INFO,
                     TransformRule.CREATE_XLSX,
                     TransformRule.CREATE_TOP_INFO,
                     TransformRule.CREATE_REGISTER_WITH_DMA
                     ]
