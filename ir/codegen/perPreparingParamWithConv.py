from ir.codegen.utils import *
import math


class ProcessRegister():
    num_tile = 0
    conv_type = 1
    fmi_raddr_b0 = 2
    fmi_raddr_b1 = 3
    fmi_raddr_b2 = 4
    fmi_raddr_b3 = 5
    fmi_c = 6
    fmi_w = 7
    fmi_h = 8
    fmi_coffset = 9
    fmi_woffset = 10
    stride = 11
    kernel_m = 12
    kernel_cmod = 13
    kernel_w = 14
    kernel_h = 15
    pad_up = 16
    pad_down = 17
    pad_left = 18
    pad_right = 19
    first_layer = 20
    pad_zero_point = 21


def get_pre_process_dict(para_buf):
    num_tile = para_buf[0]
    conv_type = para_buf[1]
    fmi_raddr_b0 = para_buf[2]
    fmi_raddr_b1 = para_buf[3]
    fmi_raddr_b2 = para_buf[4]
    fmi_raddr_b3 = para_buf[5]
    fmi_c = para_buf[6]
    fmi_w = para_buf[7]
    fmi_h = para_buf[8]
    fmi_coffset = para_buf[9]
    fmi_woffset = para_buf[10]
    stride = para_buf[11]
    kernel_m = para_buf[12]
    kernel_cmod = para_buf[13]
    kernel_w = para_buf[14]
    kernel_h = para_buf[15]
    pad_up = para_buf[16]
    pad_down = para_buf[17]
    pad_left = para_buf[18]
    pad_right = para_buf[19]
    first_layer = para_buf[20]
    pad_zero_point = para_buf[21]

    fmi_cmod = 1 if first_layer == 1 else int(fmi_c / 8)

    pre_process_dict = {}

    if (conv_type == 0):
        pre_process_dict["para_num_tile"] = num_tile
    else:
        pre_process_dict["para_num_tile"] = 0

    pre_process_dict["para_conv_type"] = conv_type
    pre_process_dict["para_mult_type"] = 0
    pre_process_dict["para_first_layer"] = first_layer
    pre_process_dict["para_fmi_raddr_b0"] = fmi_raddr_b0

    pre_process_dict["para_fmi_raddr_b1"] = fmi_raddr_b1
    pre_process_dict["para_fmi_raddr_b2"] = fmi_raddr_b2
    pre_process_dict["para_fmi_raddr_b3"] = fmi_raddr_b3
    pre_process_dict["para_fmi_w"] = fmi_w

    pre_process_dict["para_fmi_h"] = fmi_h
    pre_process_dict["para_fmi_coffset"] = fmi_coffset
    pre_process_dict["para_fmi_woffset"] = fmi_woffset

    if (pre_process_dict["para_conv_type"] == 0):
        pre_process_dict["para_stride_mux"] = stride
    else:
        pre_process_dict["para_stride_mux"] = stride * 7

    pre_process_dict["para_stride"] = stride

    if (pre_process_dict["para_conv_type"] == 0):
        pre_process_dict["para_kernel_w"] = kernel_w
    else:
        pre_process_dict["para_stride_mux"] = (
                                                      7 - 1) * stride + 0 + kernel_w - 1 + 1

    pre_process_dict["para_kernel_h"] = kernel_h
    pre_process_dict["para_kernel_cmod"] = 1 if first_layer == 1 else kernel_cmod

    kernel_hwcmod = \
        pre_process_dict["para_kernel_h"] * pre_process_dict["para_kernel_w"] * \
        pre_process_dict["para_kernel_cmod"]

    pre_process_dict["para_fmi_c"] = fmi_c
    pre_process_dict["para_fmi_scmod"] = int(
        pre_process_dict["para_stride_mux"] * fmi_cmod)

    fmi_wcmod = pre_process_dict["para_fmi_w"] * fmi_cmod

    if first_layer:
        pre_process_dict["para_fmi_wcmod"] = int(fmi_wcmod)
    else:
        if (fmi_wcmod % 4 == 0):
            pre_process_dict["para_fmi_wcmod"] = int(fmi_wcmod / 4)
        else:
            pre_process_dict["para_fmi_wcmod"] = int(fmi_wcmod / 4) + 1

    pre_process_dict["para_pad_up"] = pad_up
    pre_process_dict["para_pad_down"] = pad_down
    pre_process_dict["para_pad_left"] = pad_left

    if (pre_process_dict["para_conv_type"] == 0):
        pre_process_dict["para_pad_right"] = pad_right

    pre_process_dict["para_pad_fmi_w"] = pre_process_dict["para_pad_left"] \
                                         + pre_process_dict["para_pad_right"] + pre_process_dict["para_fmi_w"]

    pre_process_dict["para_pad_fmi_h"] = pre_process_dict["para_pad_up"] \
                                         + pre_process_dict["para_pad_down"] + pre_process_dict["para_fmi_h"]

    pre_process_dict["para_pad_fmi_wcmod"] = int(
        pre_process_dict["para_pad_fmi_w"] * fmi_cmod)
    pre_process_dict["para_pad_fmi_wscmod"] = \
        int(pre_process_dict["para_pad_fmi_w"] *
            pre_process_dict["para_stride"] * fmi_cmod)

    pre_process_dict["para_pad_zero_point"] = pad_zero_point

    pre_process_dict["para_fmo_w"] = \
        int((pre_process_dict["para_pad_fmi_w"] - pre_process_dict["para_kernel_w"] + 1 - 1)
            / pre_process_dict["para_stride_mux"] + 1)

    pre_process_dict["para_fmo_h"] = int((pre_process_dict["para_pad_fmi_h"] - \
                                          pre_process_dict["para_kernel_h"] + 1 - 1) / pre_process_dict[
                                             "para_stride"] + 1)

    pre_process_dict["para_num_win"] = 0
    if first_layer:
        if (kernel_m == 1):
            pre_process_dict["para_num_win"] = 2
        elif (kernel_m == 2):
            pre_process_dict["para_num_win"] = 1
        elif ((kernel_m == 3) or (kernel_m == 4)):
            pre_process_dict["para_num_win"] = 0
    else:
        if (pre_process_dict["para_conv_type"] == 1):
            pre_process_dict["para_num_win"] = 0
        elif (kernel_hwcmod <= 64 * 4 / 8):
            if (kernel_m == 1):
                pre_process_dict["para_num_win"] = 2
            elif (kernel_m == 2):
                pre_process_dict["para_num_win"] = 1
            elif ((kernel_m == 3) or (kernel_m == 4)):
                pre_process_dict["para_num_win"] = 0
        elif ((kernel_hwcmod > 64 * 4 / 8) and (kernel_hwcmod <= 64 * 8 / 8)):
            if (kernel_m == 1):
                pre_process_dict["para_num_win"] = 1
            elif (kernel_m == 2):
                pre_process_dict["para_num_win"] = 0
        elif (kernel_hwcmod > 64 * 8 / 8):
            pre_process_dict["para_num_win"] = 0

    pre_process_dict["para_num_psum"] = 0
    if first_layer:
        pre_process_dict["para_num_psum"] = 0
    else:
        if (pre_process_dict["para_conv_type"] == 1):
            pre_process_dict["para_num_psum"] = 0
        elif (kernel_hwcmod <= 64 * 4 / 8):
            pre_process_dict["para_num_psum"] = 0
        elif ((kernel_hwcmod > 64 * 4 / 8) and (kernel_hwcmod <= 64 * 8 / 8)):
            pre_process_dict["para_num_psum"] = 1
        elif (kernel_hwcmod > 64 * 8 / 8):
            pre_process_dict["para_num_psum"] = 2

    pre_process_dict["para_kernel_hwcmod"] = kernel_hwcmod

    pre_process_dict["para_num_slice"] = 0
    if first_layer:
        pre_process_dict["para_num_slice"] = kernel_hwcmod
    else:
        if (pre_process_dict["para_conv_type"] == 1):
            pre_process_dict["para_num_slice"] = 0
        elif (kernel_hwcmod <= 64 * 4 / 8):
            pre_process_dict["para_num_slice"] = kernel_hwcmod
        elif (kernel_hwcmod > 64 * 4 / 8 and kernel_hwcmod <= 64 * 8 / 8):
            if (kernel_hwcmod % 2 == 0):
                kernel_hwcmod_t2 = int(kernel_hwcmod / 2)
            else:
                kernel_hwcmod_t2 = int((kernel_hwcmod + 1) / 2)
            if (kernel_hwcmod % 8 == 0):
                pre_process_dict["para_num_slice"] = kernel_hwcmod_t2 * 2
            else:
                pre_process_dict["para_num_slice"] = math.ceil(
                    kernel_hwcmod_t2 / 8) * 8 * 2
        elif (kernel_hwcmod > 64 * 8 / 8):
            if (kernel_hwcmod % 4):
                kernel_hwcmod_t4 = int(kernel_hwcmod / 4)
            else:
                kernel_hwcmod_t4 = int(kernel_hwcmod / 4 + 1)

            if (kernel_hwcmod_t4 % 8 == 0):
                pre_process_dict["para_num_slice"] = kernel_hwcmod_t4 * 4
            else:
                pre_process_dict["para_num_slice"] = math.ceil(
                    kernel_hwcmod_t4 / 8) * 8 * 4

    return pre_process_dict


def pre_process_param(pre_process_dict, first_layer):
    register_dict = []
    # PRE_PARA0_COMM
    PRE_PARA0_COMM = []
    PRE_PARA0_COMM.append(n_zeros_str(3))
    if first_layer:
        PRE_PARA0_COMM.append(intToBin(0, 2))
    else:
        PRE_PARA0_COMM.append(intToBin(2, 2))
    para_num_tile = pre_process_dict["para_num_tile"]
    PRE_PARA0_COMM.append(intToBin(para_num_tile, 2))
    para_num_win = pre_process_dict["para_num_win"]
    PRE_PARA0_COMM.append(intToBin(para_num_win, 2))
    para_num_psum = pre_process_dict["para_num_psum"]
    PRE_PARA0_COMM.append(intToBin(para_num_psum, 2))
    para_num_slice = pre_process_dict["para_num_slice"]
    PRE_PARA0_COMM.append(intToBin(para_num_slice, 16))
    para_conv_type = pre_process_dict["para_conv_type"]
    PRE_PARA0_COMM.append(intToBin(para_conv_type, 3))
    para_mult_type = pre_process_dict["para_mult_type"]
    PRE_PARA0_COMM.append(intToBin(para_mult_type, 1))
    para_first_layer = pre_process_dict["para_first_layer"]
    PRE_PARA0_COMM.append(intToBin(para_first_layer, 1))
    PRE_PARA0_COMM = bin_listTobin(PRE_PARA0_COMM)
    PRE_PARA0_COMM = binTohex(PRE_PARA0_COMM, 32)
    register_dict.append(PRE_PARA0_COMM)
    # PRE_PARA0_FMI_GRP1
    PRE_PARA0_FMI_GRP1 = []
    PRE_PARA0_FMI_GRP1.append(n_zeros_str(13))
    para_fmi_raddr_b0 = pre_process_dict["para_fmi_raddr_b0"]
    PRE_PARA0_FMI_GRP1.append(intToBin(para_fmi_raddr_b0, 19))
    PRE_PARA0_FMI_GRP1 = bin_listTobin(PRE_PARA0_FMI_GRP1)
    PRE_PARA0_FMI_GRP1 = binTohex(PRE_PARA0_FMI_GRP1, 32)
    register_dict.append(PRE_PARA0_FMI_GRP1)
    # PRE_PARA0_FMI_GRP2
    PRE_PARA0_FMI_GRP2 = []
    PRE_PARA0_FMI_GRP2.append(n_zeros_str(13))
    para_fmi_raddr_b1 = pre_process_dict["para_fmi_raddr_b1"]
    PRE_PARA0_FMI_GRP2.append(intToBin(para_fmi_raddr_b1, 19))
    PRE_PARA0_FMI_GRP2 = bin_listTobin(PRE_PARA0_FMI_GRP2)
    PRE_PARA0_FMI_GRP2 = binTohex(PRE_PARA0_FMI_GRP2, 32)
    register_dict.append(PRE_PARA0_FMI_GRP2)
    # PRE_PARA0_FMI_GRP3
    PRE_PARA0_FMI_GRP3 = []
    PRE_PARA0_FMI_GRP3.append(n_zeros_str(13))
    para_fmi_raddr_b2 = pre_process_dict["para_fmi_raddr_b2"]
    PRE_PARA0_FMI_GRP3.append(intToBin(para_fmi_raddr_b2, 19))
    PRE_PARA0_FMI_GRP3 = bin_listTobin(PRE_PARA0_FMI_GRP3)
    PRE_PARA0_FMI_GRP3 = binTohex(PRE_PARA0_FMI_GRP3, 32)
    register_dict.append(PRE_PARA0_FMI_GRP3)
    # PRE_PARA0_FMI_GRP4
    PRE_PARA0_FMI_GRP4 = []
    PRE_PARA0_FMI_GRP4.append(n_zeros_str(13))
    para_fmi_raddr_b3 = pre_process_dict["para_fmi_raddr_b3"]
    PRE_PARA0_FMI_GRP4.append(intToBin(para_fmi_raddr_b3, 19))
    PRE_PARA0_FMI_GRP4 = bin_listTobin(PRE_PARA0_FMI_GRP4)
    PRE_PARA0_FMI_GRP4 = binTohex(PRE_PARA0_FMI_GRP4, 32)
    register_dict.append(PRE_PARA0_FMI_GRP4)
    # PRE_PARA0_FMI_GRP5
    PRE_PARA0_FMI_GRP5 = []
    para_fmi_c = pre_process_dict["para_fmi_c"]
    PRE_PARA0_FMI_GRP5.append(intToBin(para_fmi_c, 16))
    para_fmi_scmod = pre_process_dict["para_fmi_scmod"]
    PRE_PARA0_FMI_GRP5.append(intToBin(para_fmi_scmod, 16))
    PRE_PARA0_FMI_GRP5 = bin_listTobin(PRE_PARA0_FMI_GRP5)
    PRE_PARA0_FMI_GRP5 = binTohex(PRE_PARA0_FMI_GRP5, 32)
    register_dict.append(PRE_PARA0_FMI_GRP5)
    # PRE_PARA0_FMI_GRP6
    PRE_PARA0_FMI_GRP6 = []
    para_fmi_w = pre_process_dict["para_fmi_w"]
    PRE_PARA0_FMI_GRP6.append(intToBin(para_fmi_w, 16))
    para_fmi_h = pre_process_dict["para_fmi_h"]
    PRE_PARA0_FMI_GRP6.append(intToBin(para_fmi_h, 16))
    PRE_PARA0_FMI_GRP6 = bin_listTobin(PRE_PARA0_FMI_GRP6)
    PRE_PARA0_FMI_GRP6 = binTohex(PRE_PARA0_FMI_GRP6, 32)
    register_dict.append(PRE_PARA0_FMI_GRP6)
    # PRE_PARA0_FMI_GRP7
    PRE_PARA0_FMI_GRP7 = []
    para_fmi_coffset = pre_process_dict["para_fmi_coffset"]
    PRE_PARA0_FMI_GRP7.append(intToBin(para_fmi_coffset, 16))
    para_fmi_woffset = pre_process_dict["para_fmi_woffset"]
    PRE_PARA0_FMI_GRP7.append(intToBin(para_fmi_woffset, 16))
    PRE_PARA0_FMI_GRP7 = bin_listTobin(PRE_PARA0_FMI_GRP7)
    PRE_PARA0_FMI_GRP7 = binTohex(PRE_PARA0_FMI_GRP7, 32)
    register_dict.append(PRE_PARA0_FMI_GRP7)
    # PRE_PARA0_FMI_GRP8
    PRE_PARA0_FMI_GRP8 = []
    PRE_PARA0_FMI_GRP8.append(n_zeros_str(16))
    para_fmi_wcmod = pre_process_dict["para_fmi_wcmod"]
    PRE_PARA0_FMI_GRP8.append(intToBin(para_fmi_wcmod, 16))
    PRE_PARA0_FMI_GRP8 = bin_listTobin(PRE_PARA0_FMI_GRP8)
    PRE_PARA0_FMI_GRP8 = binTohex(PRE_PARA0_FMI_GRP8, 32)
    register_dict.append(PRE_PARA0_FMI_GRP8)
    # PRE_PARA0_KERNEL_GRP1
    PRE_PARA0_KERNEL_GRP1 = []
    PRE_PARA0_KERNEL_GRP1.append(n_zeros_str(20))
    para_stride = pre_process_dict["para_stride"]
    PRE_PARA0_KERNEL_GRP1.append(intToBin(para_stride, 4))
    para_kernel_w = pre_process_dict["para_kernel_w"]
    PRE_PARA0_KERNEL_GRP1.append(intToBin(para_kernel_w, 4))
    para_kernel_h = pre_process_dict["para_kernel_h"]
    PRE_PARA0_KERNEL_GRP1.append(intToBin(para_kernel_h, 4))
    PRE_PARA0_KERNEL_GRP1 = bin_listTobin(PRE_PARA0_KERNEL_GRP1)
    PRE_PARA0_KERNEL_GRP1 = binTohex(PRE_PARA0_KERNEL_GRP1, 32)
    register_dict.append(PRE_PARA0_KERNEL_GRP1)
    # PRE_PARA0_KERNEL_GRP2
    PRE_PARA0_KERNEL_GRP2 = []
    para_kernel_cmod = pre_process_dict["para_kernel_cmod"]
    PRE_PARA0_KERNEL_GRP2.append(intToBin(para_kernel_cmod, 16))
    para_kernel_hwcmod = pre_process_dict["para_kernel_hwcmod"]
    PRE_PARA0_KERNEL_GRP2.append(intToBin(para_kernel_hwcmod, 16))
    PRE_PARA0_KERNEL_GRP2 = bin_listTobin(PRE_PARA0_KERNEL_GRP2)
    PRE_PARA0_KERNEL_GRP2 = binTohex(PRE_PARA0_KERNEL_GRP2, 32)
    register_dict.append(PRE_PARA0_KERNEL_GRP2)
    # PRE_PARA0_PAD_GRP1
    PRE_PARA0_PAD_GRP1 = []
    PRE_PARA0_PAD_GRP1.append(n_zeros_str(8))
    para_pad_up = pre_process_dict["para_pad_up"]
    PRE_PARA0_PAD_GRP1.append(intToBin(para_pad_up, 4))
    para_pad_down = pre_process_dict["para_pad_down"]
    PRE_PARA0_PAD_GRP1.append(intToBin(para_pad_down, 4))
    para_pad_left = pre_process_dict["para_pad_left"]
    PRE_PARA0_PAD_GRP1.append(intToBin(para_pad_left, 4))
    para_pad_right = pre_process_dict["para_pad_right"]
    PRE_PARA0_PAD_GRP1.append(intToBin(para_pad_right, 4))
    para_pad_zero_point = pre_process_dict["para_pad_zero_point"]
    PRE_PARA0_PAD_GRP1.append(intToBin(para_pad_zero_point, 8))
    PRE_PARA0_PAD_GRP1 = bin_listTobin(PRE_PARA0_PAD_GRP1)
    PRE_PARA0_PAD_GRP1 = binTohex(PRE_PARA0_PAD_GRP1, 32)
    register_dict.append(PRE_PARA0_PAD_GRP1)
    # PRE_PARA0_PAD_GRP2
    PRE_PARA0_PAD_GRP2 = []
    para_pad_fmi_w = pre_process_dict["para_pad_fmi_w"]
    PRE_PARA0_PAD_GRP2.append(intToBin(para_pad_fmi_w, 16))
    para_pad_fmi_h = pre_process_dict["para_pad_fmi_h"]
    PRE_PARA0_PAD_GRP2.append(intToBin(para_pad_fmi_h, 16))
    PRE_PARA0_PAD_GRP2 = bin_listTobin(PRE_PARA0_PAD_GRP2)
    PRE_PARA0_PAD_GRP2 = binTohex(PRE_PARA0_PAD_GRP2, 32)
    register_dict.append(PRE_PARA0_PAD_GRP2)
    # PRE_PARA0_PAD_GRP3
    PRE_PARA0_PAD_GRP3 = []
    para_pad_fmi_wcmod = pre_process_dict["para_pad_fmi_wcmod"]
    PRE_PARA0_PAD_GRP3.append(intToBin(para_pad_fmi_wcmod, 16))
    para_pad_fmi_wscmod = pre_process_dict["para_pad_fmi_wscmod"]
    PRE_PARA0_PAD_GRP3.append(intToBin(para_pad_fmi_wscmod, 16))
    PRE_PARA0_PAD_GRP3 = bin_listTobin(PRE_PARA0_PAD_GRP3)
    PRE_PARA0_PAD_GRP3 = binTohex(PRE_PARA0_PAD_GRP3, 32)
    register_dict.append(PRE_PARA0_PAD_GRP3)
    # PRE_PARA0_FMO
    PRE_PARA0_FMO = []
    para_fmo_w = pre_process_dict["para_fmo_w"]
    PRE_PARA0_FMO.append(intToBin(para_fmo_w, 16))
    para_fmo_h = pre_process_dict["para_fmo_h"]
    PRE_PARA0_FMO.append(intToBin(para_fmo_h, 16))
    PRE_PARA0_FMO = bin_listTobin(PRE_PARA0_FMO)
    PRE_PARA0_FMO = binTohex(PRE_PARA0_FMO, 32)
    register_dict.append(PRE_PARA0_FMO)
    return register_dict
