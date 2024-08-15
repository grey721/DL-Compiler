from codegen.utils import *
from codegen.vpu_utils import *
from ir.graph.Graph_IR import *
# from compiler.backend.ada200.ada200 import ada200
from codegen.cimPreparingParam import *
from codegen.perPreparingParamWithoutConv import *
from codegen.vpuPreparingParam import *
from ir.dialect.npu.IR_operator import *
from enum import Enum


def getRegisterWithoutConv(npuop, path, nn_sb, _weight_, sub_block_register_list=None, input_output_dict=None):
    if npuop.input_block:
        input_output_dict["input"].append(npuop.npu_op_id)
    if npuop.output_block:
        input_output_dict["output"].append(npuop.npu_op_id)

    npu_op_flow = npuop.NpuOp.NpuOpFlow
    op_param = npu_op_flow[0]
    group_block_id = npuop.group_block_id
    npu_op_block_id = npuop.npu_op_block_id
    npu_op_group_id = npuop.npu_op_group_id

    inputH, inputW, inputC, OutputH, OutputW, OutputC = \
        op_param.InputH, op_param.InputW, op_param.InputC, \
            op_param.OutputH, op_param.OutputW, op_param.OutputC
    t_h, t_w, t_c = npuop.tile_split_mode.h, npuop.tile_split_mode.w, npuop.tile_split_mode.c

    if isinstance(op_param, NpuPad):
        pad_zero_point = op_param.pad_val[0]
    else:
        pad_zero_point = 0
    write_bank_group_id = npuop.shm_write_param_dict['tensor_info']['bank_group_id']
    tile_pad_list = npuop.npu_op_flow_tile_address_list[0]["tile_pad_list"]
    tile_size_list = npuop.npu_op_flow_tile_address_list[0]["input_tile_address_list"]
    tile_nums = npuop.tile_split_mode.split_num
    read_tile_memory_address_list = npuop.shm_read_param_dict['read_tile_memory_address_list']

    res_txt = []
    res_txt.append(["00000019", "00000000", "00000000", "00000000", "00000000", "00000000", "0000006d"])

    for i in range(4):
        FirstLayer = 1 if npu_op_group_id == 0 else 0
        if tile_nums == 2:
            if i == 1:
                pre_process_dict = get_pre_process_dict_without_conv(para_buf)
                process_param = pre_process_param(pre_process_dict, FirstLayer)
                res_txt.append(process_param)
                continue
            elif i == 3:
                pre_process_dict = get_pre_process_dict_without_conv(para_buf)
                process_param = pre_process_param(pre_process_dict, FirstLayer)
                res_txt.append(process_param)
                continue
        elif tile_nums == 1:
            if i != 0:
                pre_process_dict = get_pre_process_dict_without_conv(para_buf)
                process_param = pre_process_param(pre_process_dict, FirstLayer)
                res_txt.append(process_param)
                continue

        if tile_nums == 4:
            hh = int(i / (t_c * t_w))
            ww = int((i - hh * t_w * t_c) / t_c)
            cc = int((i - hh * t_w * t_c - ww * t_c))
        elif tile_nums == 2:
            if t_h == 2:
                hh = int(i / 2)
                ww = 0
                cc = 0
            elif t_w == 2:
                hh = 0
                ww = int(i / 2)
                cc = 0
            elif t_c == 2:
                hh = 0
                ww = 0
                cc = int(i / 2)
        elif tile_nums == 1:
            hh = 0
            ww = 0
            cc = 0
        tile_size = tile_size_list[hh][ww][cc]
        if tile_pad_list != None:
            tile_pad = tile_pad_list[hh][ww][cc]
        else:
            tile_pad = [0, 0, 0, 0]
            pass
        para_buf = get_list(nums=22, val=0)

        para_buf[ProcessRegister.num_tile] = int(tile_nums / 2)
        para_buf[ProcessRegister.conv_type] = 0  # deconv:1 conv:0

        para_buf[ProcessRegister.fmi_raddr_b0] = read_tile_memory_address_list[hh][ww][cc][0]
        para_buf[ProcessRegister.fmi_raddr_b1] = read_tile_memory_address_list[hh][ww][cc][1]
        para_buf[ProcessRegister.fmi_raddr_b2] = read_tile_memory_address_list[hh][ww][cc][2]
        para_buf[ProcessRegister.fmi_raddr_b3] = read_tile_memory_address_list[hh][ww][cc][3]

        para_buf[ProcessRegister.fmi_c] = int(tile_size[5])
        para_buf[ProcessRegister.fmi_w] = int(tile_size[3])
        para_buf[ProcessRegister.fmi_h] = int(tile_size[1])

        # if tile_dict[2] == 1:
        #     para_buf[9] = 1
        # elif tile_dict[2] == 2:
        #     para_buf[9] = int(tile_address[5]/8/4*2)
        # elif tile_dict[2] == 4:
        #     para_buf[9] = int(tile_address[5]/8/4*4)

        if npuop.npu_op_id == 0:
            para_buf[ProcessRegister.fmi_coffset] = math.ceil(8 / 8)

            para_buf[ProcessRegister.fmi_woffset] = int(8 * inputW / 8)
        else:
            para_buf[ProcessRegister.fmi_coffset] = math.ceil(inputC / 8 / 4)

            para_buf[ProcessRegister.fmi_woffset] = int(inputC * inputW / 8 / 4)

        para_buf[ProcessRegister.kernel_cmod] = int(tile_size[5] / 8)

        para_buf[ProcessRegister.pad_up] = tile_pad[0]
        para_buf[ProcessRegister.pad_down] = tile_pad[1]
        para_buf[ProcessRegister.pad_left] = tile_pad[2]
        para_buf[ProcessRegister.pad_right] = tile_pad[3]

        if npuop.npu_op_id == 0:
            para_buf[ProcessRegister.first_layer] = 1

        para_buf[ProcessRegister.pad_zero_point] = pad_zero_point

        pre_process_dict = get_pre_process_dict_without_conv(para_buf)
        if npuop.npu_op_id == 0:
            process_param = pre_process_param(pre_process_dict, 1)
        else:
            process_param = pre_process_param(pre_process_dict, 1)

        res_txt.append(process_param)

    pre_nn_sb = nn_sb
    sub_block_nums = 1
    for sub_block_num in range(sub_block_nums):
        if not os.path.exists(f'{path}/layer_{npuop.npu_op_id}/sub_block_{pre_nn_sb}'):
            os.makedirs(f'{path}/layer_{npuop.npu_op_id}/sub_block_{pre_nn_sb}')
        pre_path = f'{path}/layer_{npuop.npu_op_id}/sub_block_{pre_nn_sb}/pre_param'
        pre_nn_sb += 1
        pre_writeTxt(res_txt, pre_path, 1)

        if sub_block_register_list is not None:
            all_register_dict = dict()
            all_register_dict['npu_op_group_id'] = npu_op_group_id
            all_register_dict['npu_op_block_id'] = npu_op_block_id
            all_register_dict['group_block_id'] = group_block_id
            all_register_dict['nn_sb_id'] = pre_nn_sb
            all_register_dict["pre_process_dict"] = pre_process_dict
            sub_block_register_list.append(all_register_dict)

    cim_cluster_dict = get_list(nums=34, val=0)
    cim_nn_sb = nn_sb
    for sub_block_num in range(sub_block_nums):

        cim_cluster_dict[CimClusterRegister.para_clusters_bypass_mode] = 1

        cim_cluster_dict_ = get_cim_cluster_dict(cim_cluster_dict)
        register_dict = cim_cluster_param(cim_cluster_dict_)

        cim_cluster_path = f'{path}/layer_{npuop.npu_op_id}/sub_block_{cim_nn_sb}/cim_cluster_param'
        cim_cluster_writeTxt(register_dict, cim_cluster_path, 1)
        cim_nn_sb += 1
        if sub_block_register_list is not None:
            sub_block_register_list[sub_block_num]['cim_cluster_dict'] = cim_cluster_dict_

    vpu_nn_sb = nn_sb
    if 0:
        vpu_dict = get_list(nums=389, val=0)
        vpu_path = f'{path}/layer_{npuop.npu_op_id}/sub_block_{vpu_nn_sb}/vpu_param'
        vpu_nn_sb += 1
        vpu_dict = get_vpu_dict(vpu_dict)
        vpu_dict = vpu_param(vpu_dict)
        if npuop.NpuOp.NpuOpActivate:
            vpu_dict = add_lut_dict(vpu_dict, npuop.NpuOp.NpuOpActivateOp.lut_dict)
        else:
            vpu_dict = add_lut_dict(vpu_dict, np.zeros([256]).astype(np.int8))
            vpu_writeTxt(vpu_dict, vpu_addr, vpu_path, npuop.NpuOp.NpuOpActivate)
        vpu_writeTxt(vpu_dict, vpu_addr, vpu_path)
    else:
        shape_tem_base = npuop.shm_write_param_dict['tensor_info']['block_address_list']
        for sub_block_num in range(sub_block_nums):
            vpu_dict = get_list(nums=391, val=0)

            vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode] = 1

            if npuop.npu_psum_add or npuop.short_cut_out:
                vpu_dict[VpuRegister.vpu_SF_para_short_st_arb_enable] = 1
                vpu_dict[VpuRegister.vpu_SF_para_short_cut_buffer] = 1
            else:
                vpu_dict[VpuRegister.vpu_SF_para_short_st_arb_enable] = 0
                vpu_dict[VpuRegister.vpu_SF_para_short_cut_buffer] = 0

            if npuop.npu_op_id == 0:
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_0] = 1
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_1] = 0
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_2] = 0
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_3] = 0
                vpu_dict[VpuRegister.vpu_SF_para_fifo2line_buffer_enable] = 1
                vpu_dict[VpuRegister.vpu_SF_para_global_line_buffer_enable] = 1
                vpu_dict[VpuRegister.vpu_SF_para_input_fifo_enable] = 1
            else:
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_0] = 1
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_1] = 1
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_2] = 1
                vpu_dict[VpuRegister.vpu_SF_para_line_controller_3] = 1
                vpu_dict[VpuRegister.vpu_SF_para_fifo2line_buffer_enable] = 15
                vpu_dict[VpuRegister.vpu_SF_para_global_line_buffer_enable] = 15
                vpu_dict[VpuRegister.vpu_SF_para_input_fifo_enable] = 15

            vpu_dict[VpuRegister.vpu_SF_para_psum_enable] = 0

            if npuop.NpuOp.NpuOpShortCutOut and npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_mode] = 3
            elif npuop.NpuOp.NpuOpShortCutOut and not npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_mode] = 1
            elif not npuop.NpuOp.NpuOpShortCutOut and npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_mode] = 2
            if npuop.npu_psum_add:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_mode] = 2

            if npuop.npu_psum_add:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_enable] = 1
            elif npuop.short_cut_out:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_enable] = 15
            else:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_enable] = 0

            if npuop.npu_op_id == 0:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_enable] = 1
            else:
                vpu_dict[VpuRegister.vpu_SF_para_vpu_enable] = 15

            vpu_dict[VpuRegister.vpu_TOP_para_fmt_channel_type] = 2

            vpu_dict[VpuRegister.vpu_TOP_para_switch_mode] = npuop.NpuOp.NpuOpMode

            if npuop.NpuOp.NpuShortCutMode == None:
                vpu_dict[VpuRegister.vpu_TOP_para_sc_mode] = 0
            else:
                vpu_dict[VpuRegister.vpu_TOP_para_sc_mode] = npuop.NpuOp.NpuShortCutMode

            vpu_dict[VpuRegister.vpu_TOP_para_line_buffer_mode] = 0

            if npuop.NpuOp.NpuOpElemwise:
                if npuop.NpuOp.NpuOpElemwiseOp.ElwMode == 0:
                    vpu_dict[VpuRegister.vpu_WISE_para_mode] = 1
                elif npuop.NpuOp.NpuOpElemwiseOp.ElwMode == 2:
                    vpu_dict[VpuRegister.vpu_WISE_para_mode] = 2
                elif npuop.NpuOp.NpuOpElemwiseOp.ElwMode == 3:
                    vpu_dict[VpuRegister.vpu_WISE_para_mode] = 3

            if npuop.npu_psum_add and npuop.int32_out:
                vpu_dict[VpuRegister.vpu_WISE_para_mode] = 5
            elif npuop.npu_psum_add and not npuop.int32_out:
                vpu_dict[VpuRegister.vpu_WISE_para_mode] = 4

            if npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_WISE_para_quantize_mul] = npuop.NpuOp.NpuOpElemwiseOp.output_multiplier[0]
                vpu_dict[VpuRegister.vpu_WISE_para_quantize_shf] = npuop.NpuOp.NpuOpElemwiseOp.output_shift[0]
                vpu_dict[VpuRegister.vpu_WISE_para_quantize_off] = npuop.NpuOp.NpuOpElemwiseOp.output_offset[0]
                vpu_dict[VpuRegister.vpu_WISE_para_element_wise_dequantize_0_sclale_o] = \
                npuop.NpuOp.NpuOpElemwiseOp.input_multiplier[0]
                vpu_dict[VpuRegister.vpu_WISE_para_element_wise_dequantize_0_shifter_o] = \
                npuop.NpuOp.NpuOpElemwiseOp.input_shift[0]
                vpu_dict[VpuRegister.vpu_WISE_para_element_wise_dequantize_1_sclale_o] = \
                npuop.NpuOp.NpuOpElemwiseOp.input1_multiplier[0]
                vpu_dict[VpuRegister.vpu_WISE_para_element_wise_dequantize_1_shifter_o] = \
                npuop.NpuOp.NpuOpElemwiseOp.input1_shift[0]
                vpu_dict[VpuRegister.vpu_WISE_para_div_fix_param] = 0
                vpu_dict[VpuRegister.vpu_WISE_para_div_shifter] = 0
            if npuop.NpuOp.NpuOpResize:
                vpu_dict[VpuRegister.vpu_BASIC_para_i_pl_rs_sel] = 1
                vpu_dict[
                    VpuRegister.vpu_BASIC_para_i_resize_param_half_pixal_flag] = npuop.NpuOp.NpuOpResizeOp.HalfPixelCenters
                vpu_dict[
                    VpuRegister.vpu_BASIC_para_i_resize_param_bil_nn_sel_flag] = npuop.NpuOp.NpuOpResizeOp.AlignCorners
            if npuop.NpuOp.NpuOpPool:
                vpu_dict[VpuRegister.vpu_BASIC_para_pl_func_mode] = npuop.NpuOp.NpuOpPoolOp.PMode
                vpu_dict[VpuRegister.vpu_BASIC_para_pl_factor] = 2
                vpu_dict[VpuRegister.vpu_FILTER_para_i_pooling_filter_width] = npuop.NpuOp.NpuOpPoolOp.KerW
                vpu_dict[VpuRegister.vpu_FILTER_para_i_pooling_filter_height] = npuop.NpuOp.NpuOpPoolOp.KerH
                vpu_dict[VpuRegister.vpu_STRIDE_para_i_pooling_stride_width] = npuop.NpuOp.NpuOpPoolOp.StrideW
                vpu_dict[VpuRegister.vpu_STRIDE_para_i_pooling_stride_height] = npuop.NpuOp.NpuOpPoolOp.StrideH
                vpu_dict[VpuRegister.vpu_BASIC_para_i_pl_rs_sel] = 0

            vpu_dict[VpuRegister.vpu_TOP_para_read_line_buffer_mode] = 1
            vpu_dict[VpuRegister.vpu_BASIC_para_i_fmt_width] = 0

            vpu_dict[VpuRegister.vpu_INTERFACE_para_b0_ad] = \
            zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][0]
            vpu_dict[VpuRegister.vpu_INTERFACE_para_b1_ad] = \
            zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][1]
            vpu_dict[VpuRegister.vpu_INTERFACE_para_b2_ad] = \
            zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][2]
            vpu_dict[VpuRegister.vpu_INTERFACE_para_b3_ad] = \
            zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][3]

            if npuop.short_cut_out:
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b4_wt_addr] = \
                zhaofang_share_mem_dict[npuop.short_cut_shm_write_param_dict['tensor_info']['bank_group_id']][0]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b5_wt_addr] = \
                zhaofang_share_mem_dict[npuop.short_cut_shm_write_param_dict['tensor_info']['bank_group_id']][1]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b6_wt_addr] = \
                zhaofang_share_mem_dict[npuop.short_cut_shm_write_param_dict['tensor_info']['bank_group_id']][2]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b7_wt_addr] = \
                zhaofang_share_mem_dict[npuop.short_cut_shm_write_param_dict['tensor_info']['bank_group_id']][3]

            if npuop.npu_psum_add:
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b4_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_psum_read_param_dict['tensor_info']['bank_group_id']][0]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b5_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_psum_read_param_dict['tensor_info']['bank_group_id']][1]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b6_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_psum_read_param_dict['tensor_info']['bank_group_id']][2]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b7_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_psum_read_param_dict['tensor_info']['bank_group_id']][3]

            if npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b4_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']][0]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b5_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']][1]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b6_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']][2]
                vpu_dict[VpuRegister.vpu_INTERFACE_para_b7_rd_addr] = \
                zhaofang_share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']][3]
                vpu_dict[VpuRegister.vpu_WISE_para_quantize_min] = npuop.NpuOp.NpuOpElemwiseOp.quantized_activation_min
                vpu_dict[VpuRegister.vpu_WISE_para_quantize_max] = npuop.NpuOp.NpuOpElemwiseOp.quantized_activation_max

            vpu_dict[VpuRegister.vpu_SF_para_vpu_fifo_group_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_SF_para_global_line_buffer_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_SF_para_sc_buffer_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_SF_para_vpu_unit_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_SF_para_interface_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_SF_para_top_ctrl_sf_rst] = 1
            vpu_dict[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_sel] = 0
            vpu_dict[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_enable] = 0
            vpu_dict[VpuRegister.vpu_SF_para_bp_mode_enable] = 0

            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_stov] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_ema] = 7
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_emaw] = 3
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_emas] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_ret1n] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_rawl] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_rawlm] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_wabl] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_wablm] = 0

            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_stov] = 0
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_ema] = 7
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_emaw] = 3
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_emas] = 1
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_ret1n] = 1
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_rawl] = 0
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_rawlm] = 0
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_wabl] = 1
            vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_wablm] = 0

            vpu_dict[VpuRegister.vpu_LUT_para_lut_stov] = 0
            vpu_dict[VpuRegister.vpu_LUT_para_lut_ema] = 7
            vpu_dict[VpuRegister.vpu_LUT_para_lut_emaw] = 3
            vpu_dict[VpuRegister.vpu_LUT_para_lut_emas] = 1
            vpu_dict[VpuRegister.vpu_LUT_para_lut_ret1n] = 1
            vpu_dict[VpuRegister.vpu_LUT_para_lut_rawl] = 0
            vpu_dict[VpuRegister.vpu_LUT_para_lut_rawlm] = 0
            vpu_dict[VpuRegister.vpu_LUT_para_lut_wabl] = 1
            vpu_dict[VpuRegister.vpu_LUT_para_lut_wablm] = 0

            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_stov] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_ema] = 7
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_emaw] = 3
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_emas] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_ret1n] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawl] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawlm] = 0
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_wabl] = 1
            vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_wablm] = 0

            sub_block_output_shape = npuop.npu_op_flow_block_address_list[-1]['output_block_address_list']
            fmo_shape = npuop.shm_write_param_dict['tensor_info']['origin_shape']
            tile_address_list = npuop.npu_op_flow_tile_address_list[-1]['output_tile_address_list']
            group_block_id = npuop.group_block_id

            if npuop.int32_out:
                bs = 1 / 2 / 4
            else:
                bs = 1 / 8 / 4

            s_addr = int(npuop.shm_write_param_dict['tensor_info']['addr'] * bs)
            out_bank_group_addr = share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']]
            out_bank_group_addr = [i + s_addr for i in out_bank_group_addr]
            sub_vpu_addr = tile_address_list.reshape(-1, 6)

            if len(sub_vpu_addr) == 4:

                shape_tem = sub_vpu_addr[3]
                shape_tem = [shape_tem[0] + shape_tem_base[0], shape_tem[1], shape_tem[2] + shape_tem_base[2],
                             shape_tem[3], shape_tem[4] + shape_tem_base[4], shape_tem[5]]
                ss_add = int(
                    (shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] + shape_tem[4]) * bs)
                shape_tem_addr = [i + ss_add for i in out_bank_group_addr]

                if npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 4:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[3]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 2:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[1]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 1:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[0]

                vpu_dict[VpuRegister.vpu_3_para_clus_3_line_buffer_w_max] = int(cluster_input_shape[3])
                vpu_dict[VpuRegister.vpu_3_para_clus_3_line_buffer_h_max] = int(cluster_input_shape[1])

                vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_0] = shape_tem_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_1] = shape_tem_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_2] = shape_tem_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_3] = shape_tem_addr[3] & 16383

                if npuop.NpuOp.NpuOpPool:
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h] = npuop.NpuOp.NpuOpPoolOp.KerH
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h_stride] = npuop.NpuOp.NpuOpPoolOp.StrideH

                elif npuop.NpuOp.NpuOpResize:
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h] = 2
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h_stride] = 1

                else:
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h] = 1
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h_stride] = 1

                vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l1_step] = 1
                vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l2_step] = 1
                vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l3_step] = 1

                fmo_c = fmo_shape[2]
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5]
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

                vpu_dict = vpu_jump(VpuRegister.vpu_3_para_clus_3_output_l1_condition,
                                    VpuRegister.vpu_3_para_clus_3_output_l2_condition, \
                                    VpuRegister.vpu_3_para_clus_3_output_l3_condition,
                                    VpuRegister.vpu_3_para_clus_3_output_l1_addr_step, \
                                    VpuRegister.vpu_3_para_clus_3_output_l2_addr_step,
                                    VpuRegister.vpu_3_para_clus_3_output_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

                if npuop.NpuOp.NpuOpShortCutOut == True:
                    short_cut_shm_write_param_dict = npuop.short_cut_shm_write_param_dict['tensor_info']
                    bank_group_id = short_cut_shm_write_param_dict['bank_group_id']
                    addr = int(short_cut_shm_write_param_dict['addr'] / 8 / 4)
                    output_shape = short_cut_shm_write_param_dict['concat_output_shape'][1:]
                    block_address_list = short_cut_shm_write_param_dict['block_address_list']

                    sc_shape = [block_address_list[0], block_address_list[1], block_address_list[2],
                                block_address_list[3], shape_tem[4], shape_tem[5]]
                    sc_add = int((sc_shape[0] * output_shape[1] * output_shape[2] + sc_shape[2] * output_shape[2] +
                                  sc_shape[4]) / 8 / 4)
                    sc_bank_group_addr = share_mem_dict[bank_group_id]
                    sc_bank_group_addr = [i + sc_add + addr for i in sc_bank_group_addr]

                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_0] = sc_bank_group_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_1] = sc_bank_group_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_2] = sc_bank_group_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_3] = sc_bank_group_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l2_step] = 1
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l3_step] = 1
                    fmo_c = output_shape[2]
                    fmo_w = output_shape[1]
                    fmo_h = output_shape[0]
                    cluster_fmo_c = sc_shape[5]
                    cluster_fmo_w = sc_shape[3]
                    cluster_fmo_h = sc_shape[1]

                    vpu_dict = vpu_jump(VpuRegister.vpu_3_para_clus_3_scw_l1_condition,
                                        VpuRegister.vpu_3_para_clus_3_scw_l2_condition, \
                                        VpuRegister.vpu_3_para_clus_3_scw_l3_condition,
                                        VpuRegister.vpu_3_para_clus_3_scw_l1_addr_step, \
                                        VpuRegister.vpu_3_para_clus_3_scw_l2_addr_step,
                                        VpuRegister.vpu_3_para_clus_3_scw_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

                if npuop.NpuOp.NpuOpResize:
                    resize_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list,
                                                 Resize)
                    resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)[-1]
                    resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)[-1]
                    tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)[-1]

                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_width_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_w)
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_height_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_h)
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_width] = resize_input_shape[3]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_height] = resize_input_shape[1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_output_width] = resize_output_shape[3]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_output_height] = resize_output_shape[1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_width] = tile_drop_line_list[0]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_height] = tile_drop_line_list[1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_width] = tile_drop_line_list[2]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_height] = tile_drop_line_list[3]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_mode] = 15

                if npuop.NpuOp.NpuOpPool:
                    pool_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Pool)
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[-1]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[-1]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[-1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_width] = pool_input_shape[3]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_height] = pool_input_shape[1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_width] = pool_output_shape[3]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_height] = pool_output_shape[1]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_width] = pool_tile_pad_list[2]
                    vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_height] = pool_tile_pad_list[0]

                if npuop.NpuOp.NpuOpElemwise:
                    elemwise_s_addr = int(npuop.shm_read_param_dict['tensor_info']['addr'] / 8 / 4)
                    elemwise_bank_group_addr = share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']]
                    elemwise_ss_add = int((shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] +
                                           shape_tem[4]) / 8 / 4)
                    elemwise_bank_group_addr = [i + elemwise_s_addr for i in elemwise_bank_group_addr]
                    elemwise_shape_tem_addr = [i + elemwise_ss_add for i in elemwise_bank_group_addr]
                    fmo_c = fmo_shape[2]
                    fmo_w = fmo_shape[1]
                    fmo_h = fmo_shape[0]
                    cluster_fmo_c = shape_tem[5]
                    cluster_fmo_w = shape_tem[3]
                    cluster_fmo_h = shape_tem[1]

                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_0] = elemwise_shape_tem_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_1] = elemwise_shape_tem_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_2] = elemwise_shape_tem_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_3] = elemwise_shape_tem_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l3_step] = 1
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l2_step] = 1

                    vpu_dict = vpu_jump(VpuRegister.vpu_3_para_clus_3_scr_l1_condition,
                                        VpuRegister.vpu_3_para_clus_3_scr_l2_condition, \
                                        VpuRegister.vpu_3_para_clus_3_scr_l3_condition,
                                        VpuRegister.vpu_3_para_clus_3_scr_l1_addr_step, \
                                        VpuRegister.vpu_3_para_clus_3_scr_l2_addr_step,
                                        VpuRegister.vpu_3_para_clus_3_scr_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

            if len(sub_vpu_addr) == 1:
                pass
            else:
                if len(sub_vpu_addr) == 4:
                    shape_tem = sub_vpu_addr[1]
                elif len(sub_vpu_addr) == 2:
                    shape_tem = sub_vpu_addr[1]

                if npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 4:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[1]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 2:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[0]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 1:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[0]

                vpu_dict[VpuRegister.vpu_1_para_clus_1_line_buffer_w_max] = int(cluster_input_shape[3])
                vpu_dict[VpuRegister.vpu_1_para_clus_1_line_buffer_h_max] = int(cluster_input_shape[1])

                shape_tem = [shape_tem[0] + shape_tem_base[0], shape_tem[1], shape_tem[2] + shape_tem_base[2],
                             shape_tem[3], shape_tem[4] + shape_tem_base[4], shape_tem[5]]
                ss_add = int(
                    (shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] + shape_tem[4]) * bs)
                shape_tem_addr = [i + ss_add for i in out_bank_group_addr]

                vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_0] = shape_tem_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_1] = shape_tem_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_2] = shape_tem_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_3] = shape_tem_addr[3] & 16383

                if npuop.NpuOp.NpuOpPool:
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h] = npuop.NpuOp.NpuOpPoolOp.KerH
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h_stride] = npuop.NpuOp.NpuOpPoolOp.StrideH
                elif npuop.NpuOp.NpuOpResize:
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h] = 2
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h_stride] = 1
                else:
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h] = 1
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h_stride] = 1

                vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l1_step] = 1
                vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l2_step] = 1
                vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l3_step] = 1

                fmo_c = fmo_shape[2]
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5]
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

                vpu_dict = vpu_jump(VpuRegister.vpu_1_para_clus_1_output_l1_condition,
                                    VpuRegister.vpu_1_para_clus_1_output_l2_condition, \
                                    VpuRegister.vpu_1_para_clus_1_output_l3_condition,
                                    VpuRegister.vpu_1_para_clus_1_output_l1_addr_step, \
                                    VpuRegister.vpu_1_para_clus_1_output_l2_addr_step,
                                    VpuRegister.vpu_1_para_clus_1_output_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

                if npuop.NpuOp.NpuOpShortCutOut == True:
                    short_cut_shm_write_param_dict = npuop.short_cut_shm_write_param_dict['tensor_info']
                    bank_group_id = short_cut_shm_write_param_dict['bank_group_id']
                    addr = int(short_cut_shm_write_param_dict['addr'] / 8 / 4)
                    output_shape = short_cut_shm_write_param_dict['concat_output_shape'][1:]
                    block_address_list = short_cut_shm_write_param_dict['block_address_list']

                    sc_shape = [block_address_list[0], block_address_list[1], block_address_list[2],
                                block_address_list[3], shape_tem[4], shape_tem[5]]
                    sc_add = int((sc_shape[0] * output_shape[1] * output_shape[2] + sc_shape[2] * output_shape[2] +
                                  sc_shape[4]) / 8 / 4)
                    sc_bank_group_addr = share_mem_dict[bank_group_id]
                    sc_bank_group_addr = [i + sc_add + addr for i in sc_bank_group_addr]

                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_0] = sc_bank_group_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_1] = sc_bank_group_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_2] = sc_bank_group_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_3] = sc_bank_group_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l2_step] = 1
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l3_step] = 1
                    fmo_c = output_shape[2]
                    fmo_w = output_shape[1]
                    fmo_h = output_shape[0]
                    cluster_fmo_c = sc_shape[5]
                    cluster_fmo_w = sc_shape[3]
                    cluster_fmo_h = sc_shape[1]

                    vpu_dict = vpu_jump(VpuRegister.vpu_1_para_clus_1_scw_l1_condition,
                                        VpuRegister.vpu_1_para_clus_1_scw_l2_condition, \
                                        VpuRegister.vpu_1_para_clus_1_scw_l3_condition,
                                        VpuRegister.vpu_1_para_clus_1_scw_l1_addr_step, \
                                        VpuRegister.vpu_1_para_clus_1_scw_l2_addr_step,
                                        VpuRegister.vpu_1_para_clus_1_scw_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

                if npuop.NpuOp.NpuOpResize:
                    resize_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list,
                                                 Resize)
                    resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)
                    resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)
                    tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)
                    if len(resize_input_shape) == 4:
                        resize_input_shape = resize_input_shape[1]
                        resize_output_shape = resize_output_shape[1]
                        tile_drop_line_list = tile_drop_line_list[1]
                    elif len(resize_input_shape) == 2:
                        resize_input_shape = resize_input_shape[0]
                        resize_output_shape = resize_output_shape[0]
                        tile_drop_line_list = tile_drop_line_list[0]
                    elif len(resize_input_shape) == 1:
                        resize_input_shape = resize_input_shape[0]
                        resize_output_shape = resize_output_shape[0]
                        tile_drop_line_list = tile_drop_line_list[0]

                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_width_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_w)
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_height_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_h)
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_width] = resize_input_shape[3]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_height] = resize_input_shape[1]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_output_width] = resize_output_shape[3]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_output_height] = resize_output_shape[1]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_width] = tile_drop_line_list[0]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_height] = tile_drop_line_list[1]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_width] = tile_drop_line_list[2]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_height] = tile_drop_line_list[3]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_mode] = 15

                if npuop.NpuOp.NpuOpPool:
                    pool_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Pool)
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)
                    if len(pool_input_shape) == 4:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
                    elif len(pool_input_shape) == 2:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
                    elif len(pool_input_shape) == 1:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]

                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_width] = pool_input_shape[3]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_height] = pool_input_shape[1]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_width] = pool_output_shape[3]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_height] = pool_output_shape[1]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_width] = pool_tile_pad_list[2]
                    vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_height] = pool_tile_pad_list[0]

                if npuop.NpuOp.NpuOpElemwise:
                    elemwise_s_addr = int(npuop.shm_read_param_dict['tensor_info']['addr'] / 8 / 4)
                    elemwise_bank_group_addr = share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']]
                    elemwise_ss_add = int((shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] +
                                           shape_tem[4]) / 8 / 4)
                    elemwise_bank_group_addr = [i + elemwise_s_addr for i in elemwise_bank_group_addr]
                    elemwise_shape_tem_addr = [i + elemwise_ss_add for i in elemwise_bank_group_addr]
                    fmo_c = fmo_shape[2]
                    fmo_w = fmo_shape[1]
                    fmo_h = fmo_shape[0]
                    cluster_fmo_c = shape_tem[5]
                    cluster_fmo_w = shape_tem[3]
                    cluster_fmo_h = shape_tem[1]

                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_0] = elemwise_shape_tem_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_1] = elemwise_shape_tem_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_2] = elemwise_shape_tem_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_3] = elemwise_shape_tem_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l3_step] = 1
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l2_step] = 1

                    vpu_dict = vpu_jump(VpuRegister.vpu_1_para_clus_1_scr_l1_condition,
                                        VpuRegister.vpu_1_para_clus_1_scr_l2_condition, \
                                        VpuRegister.vpu_1_para_clus_1_scr_l3_condition,
                                        VpuRegister.vpu_1_para_clus_1_scr_l1_addr_step, \
                                        VpuRegister.vpu_1_para_clus_1_scr_l2_addr_step,
                                        VpuRegister.vpu_1_para_clus_1_scr_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

            if len(sub_vpu_addr) == 4:

                shape_tem = sub_vpu_addr[2]
                shape_tem = [shape_tem[0] + shape_tem_base[0], shape_tem[1], shape_tem[2] + shape_tem_base[2],
                             shape_tem[3], shape_tem[4] + shape_tem_base[4], shape_tem[5]]
                ss_add = int(
                    (shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] + shape_tem[4]) * bs)
                shape_tem_addr = [i + ss_add for i in out_bank_group_addr]

                if npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 4:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[2]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 2:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[1]
                elif npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6).shape[0] == 1:
                    cluster_input_shape = \
                    npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[0]

                vpu_dict[VpuRegister.vpu_2_para_clus_2_line_buffer_w_max] = int(cluster_input_shape[3])
                vpu_dict[VpuRegister.vpu_2_para_clus_2_line_buffer_h_max] = int(cluster_input_shape[1])

                vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_0] = shape_tem_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_1] = shape_tem_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_2] = shape_tem_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_3] = shape_tem_addr[3] & 16383

                if npuop.NpuOp.NpuOpPool:
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h] = npuop.NpuOp.NpuOpPoolOp.KerH
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h_stride] = npuop.NpuOp.NpuOpPoolOp.StrideH

                elif npuop.NpuOp.NpuOpResize:
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h] = 2
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h_stride] = 1

                else:
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h] = 1
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h_stride] = 1

                vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l1_step] = 1
                vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l2_step] = 1
                vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l3_step] = 1

                fmo_c = fmo_shape[2]
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5]
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

                vpu_dict = vpu_jump(VpuRegister.vpu_2_para_clus_2_output_l1_condition,
                                    VpuRegister.vpu_2_para_clus_2_output_l2_condition, \
                                    VpuRegister.vpu_2_para_clus_2_output_l3_condition,
                                    VpuRegister.vpu_2_para_clus_2_output_l1_addr_step, \
                                    VpuRegister.vpu_2_para_clus_2_output_l2_addr_step,
                                    VpuRegister.vpu_2_para_clus_2_output_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

                if npuop.NpuOp.NpuOpElemwise:
                    elemwise_s_addr = int(npuop.shm_read_param_dict['tensor_info']['addr'] / 8 / 4)
                    elemwise_bank_group_addr = share_mem_dict[npuop.shm_read_param_dict['tensor_info']['bank_group_id']]
                    elemwise_ss_add = int((shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] +
                                           shape_tem[4]) / 8 / 4)
                    elemwise_bank_group_addr = [i + elemwise_s_addr for i in elemwise_bank_group_addr]
                    elemwise_shape_tem_addr = [i + elemwise_ss_add for i in elemwise_bank_group_addr]
                    fmo_c = fmo_shape[2]
                    fmo_w = fmo_shape[1]
                    fmo_h = fmo_shape[0]
                    cluster_fmo_c = shape_tem[5]
                    cluster_fmo_w = shape_tem[3]
                    cluster_fmo_h = shape_tem[1]

                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_0] = elemwise_shape_tem_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_1] = elemwise_shape_tem_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_2] = elemwise_shape_tem_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_3] = elemwise_shape_tem_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l3_step] = 1
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l2_step] = 1

                    vpu_dict = vpu_jump(VpuRegister.vpu_2_para_clus_2_scr_l1_condition,
                                        VpuRegister.vpu_2_para_clus_2_scr_l2_condition, \
                                        VpuRegister.vpu_2_para_clus_2_scr_l3_condition,
                                        VpuRegister.vpu_2_para_clus_2_scr_l1_addr_step, \
                                        VpuRegister.vpu_2_para_clus_2_scr_l2_addr_step,
                                        VpuRegister.vpu_2_para_clus_2_scr_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

                if npuop.NpuOp.NpuOpShortCutOut == True:
                    short_cut_shm_write_param_dict = npuop.short_cut_shm_write_param_dict['tensor_info']
                    bank_group_id = short_cut_shm_write_param_dict['bank_group_id']
                    addr = int(short_cut_shm_write_param_dict['addr'] / 4 / 8)
                    output_shape = short_cut_shm_write_param_dict['concat_output_shape'][1:]
                    block_address_list = short_cut_shm_write_param_dict['block_address_list']

                    sc_shape = [block_address_list[0], block_address_list[1], block_address_list[2],
                                block_address_list[3], shape_tem[4], shape_tem[5]]
                    sc_add = int((sc_shape[0] * output_shape[1] * output_shape[2] + sc_shape[2] * output_shape[2] +
                                  sc_shape[4]) / 8 / 4)
                    sc_bank_group_addr = share_mem_dict[bank_group_id]
                    sc_bank_group_addr = [i + sc_add + addr for i in sc_bank_group_addr]

                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_0] = sc_bank_group_addr[0] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_1] = sc_bank_group_addr[1] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_2] = sc_bank_group_addr[2] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_3] = sc_bank_group_addr[3] & 16383
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l1_step] = 1
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l2_step] = 1
                    vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l3_step] = 1
                    fmo_c = output_shape[2]
                    fmo_w = output_shape[1]
                    fmo_h = output_shape[0]
                    cluster_fmo_c = sc_shape[5]
                    cluster_fmo_w = sc_shape[3]
                    cluster_fmo_h = sc_shape[1]

                    vpu_dict = vpu_jump(VpuRegister.vpu_2_para_clus_2_scw_l1_condition,
                                        VpuRegister.vpu_2_para_clus_2_scw_l2_condition, \
                                        VpuRegister.vpu_2_para_clus_2_scw_l3_condition,
                                        VpuRegister.vpu_2_para_clus_2_scw_l1_addr_step, \
                                        VpuRegister.vpu_2_para_clus_2_scw_l2_addr_step,
                                        VpuRegister.vpu_2_para_clus_2_scw_l3_addr_step, \
                                        VpuRegister.vpu_TOP_para_interface_write_mode, \
                                        fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                        tot_bank=4)

                if npuop.NpuOp.NpuOpResize:
                    resize_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list,
                                                 Resize)
                    resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)
                    resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)
                    tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)
                    if len(resize_input_shape) == 4:
                        resize_input_shape = resize_input_shape[2]
                        resize_output_shape = resize_output_shape[2]
                        tile_drop_line_list = tile_drop_line_list[2]
                    elif len(resize_input_shape) == 2:
                        resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)[1]
                        resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)[1]
                        tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)[1]
                    elif len(resize_input_shape) == 1:
                        resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)[0]
                        resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)[0]
                        tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)[0]

                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_width_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_w)
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_height_ratio] = int(
                        npuop.NpuOp.NpuOpResizeOp.ratio_h)
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_width] = resize_input_shape[3]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_height] = resize_input_shape[1]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_output_width] = resize_output_shape[3]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_output_height] = resize_output_shape[1]

                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_width] = tile_drop_line_list[0]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_height] = tile_drop_line_list[1]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_width] = tile_drop_line_list[2]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_height] = tile_drop_line_list[3]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_mode] = 15

                if npuop.NpuOp.NpuOpPool:
                    pool_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Pool)
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)
                    if len(pool_input_shape) == 4:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[2]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[2]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[2]
                    elif len(pool_input_shape) == 2:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
                    elif len(pool_input_shape) == 1:
                        pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                        pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                        pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]

                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_width] = pool_input_shape[3]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_height] = pool_input_shape[1]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_width] = pool_output_shape[3]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_height] = pool_output_shape[1]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_width] = pool_tile_pad_list[2]
                    vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_height] = pool_tile_pad_list[0]

            if len(sub_vpu_addr) == 4:
                shape_tem = sub_vpu_addr[0]
            elif len(sub_vpu_addr) == 2:
                shape_tem = sub_vpu_addr[0]
            elif len(sub_vpu_addr) == 1:
                shape_tem = sub_vpu_addr[0]

            cluster_input_shape = npuop.npu_op_flow_tile_address_list[0]['output_tile_address_list'].reshape(-1, 6)[0]
            vpu_dict[VpuRegister.vpu_0_para_clus_0_line_buffer_w_max] = int(cluster_input_shape[3])
            vpu_dict[VpuRegister.vpu_0_para_clus_0_line_buffer_h_max] = int(cluster_input_shape[1])

            shape_tem = [shape_tem[0] + shape_tem_base[0], shape_tem[1], shape_tem[2] + shape_tem_base[2], shape_tem[3],
                         shape_tem[4] + shape_tem_base[4], shape_tem[5]]
            ss_add = int((shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] + shape_tem[4]) * bs)
            shape_tem_addr = [i + ss_add for i in out_bank_group_addr]

            if npuop.npu_psum_add:
                vpu_dict[VpuRegister.vpu_TOP_para_sc_width] = int(shape_tem[1] * shape_tem[3])
            elif npuop.NpuOp.NpuOpElemwise:
                vpu_dict[VpuRegister.vpu_TOP_para_sc_width] = int(shape_tem[1] * shape_tem[3] * shape_tem[5] / 32)

            if npuop.npu_psum_add == True:
                p_sum_s_addr = int(npuop.shm_psum_read_param_dict['tensor_info']['addr'] / 8 / 4)
                p_sum_bank_group_addr = share_mem_dict[npuop.shm_psum_read_param_dict['tensor_info']['bank_group_id']]
                p_sum_ss_add = int(
                    (shape_tem[0] * fmo_shape[1] * fmo_shape[2] + shape_tem[2] * fmo_shape[2] + shape_tem[4]) / 2 / 4)
                p_sum_bank_group_addr = [i + p_sum_s_addr for i in p_sum_bank_group_addr]
                p_sum_shape_tem_addr = [i + p_sum_ss_add for i in p_sum_bank_group_addr]
            else:
                p_sum_shape_tem_addr = [0, 0, 0, 0]

            if npuop.NpuOp.NpuOpShortCutOut == True:
                short_cut_shm_write_param_dict = npuop.short_cut_shm_write_param_dict['tensor_info']
                bank_group_id = short_cut_shm_write_param_dict['bank_group_id']
                addr = int(short_cut_shm_write_param_dict['addr'] / 8 / 4)
                output_shape = short_cut_shm_write_param_dict['origin_shape']
                block_address_list = short_cut_shm_write_param_dict['block_address_list']

                sc_shape = [block_address_list[0], block_address_list[1], block_address_list[2], block_address_list[3],
                            shape_tem[4], shape_tem[5]]
                sc_add = int((sc_shape[0] * output_shape[1] * output_shape[2] + sc_shape[2] * output_shape[2] +
                              sc_shape[4]) / 8 / 4)
                sc_bank_group_addr = share_mem_dict[bank_group_id]
                sc_bank_group_addr = [i + sc_add + addr for i in sc_bank_group_addr]

                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_0] = sc_bank_group_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_1] = sc_bank_group_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_2] = sc_bank_group_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_3] = sc_bank_group_addr[3] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l1_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l2_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l3_step] = 1

                fmo_c = output_shape[2]
                fmo_w = output_shape[1]
                fmo_h = output_shape[0]
                cluster_fmo_c = sc_shape[5]
                cluster_fmo_w = sc_shape[3]
                cluster_fmo_h = sc_shape[1]

                vpu_dict = vpu_jump(VpuRegister.vpu_0_para_clus_0_scw_l1_condition,
                                    VpuRegister.vpu_0_para_clus_0_scw_l2_condition, \
                                    VpuRegister.vpu_0_para_clus_0_scw_l3_condition,
                                    VpuRegister.vpu_0_para_clus_0_scw_l1_addr_step, \
                                    VpuRegister.vpu_0_para_clus_0_scw_l2_addr_step,
                                    VpuRegister.vpu_0_para_clus_0_scw_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

            if npuop.int32_out == False and len(sub_vpu_addr) == 1 and npuop.npu_op_id != 0:
                if vpu_nn_sb % 2 == 0:
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b0_ad] = \
                    zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][0]
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b1_ad] = \
                    zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][1]
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b2_ad] = 0
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b3_ad] = 0
                else:
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b0_ad] = \
                    zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][2]
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b1_ad] = \
                    zhaofang_share_mem_dict[npuop.shm_write_param_dict['tensor_info']['bank_group_id']][3]
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b2_ad] = 0
                    vpu_dict[VpuRegister.vpu_INTERFACE_para_b3_ad] = 0

            vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_0] = shape_tem_addr[0] & 16383
            vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_1] = shape_tem_addr[1] & 16383
            vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_2] = shape_tem_addr[2] & 16383
            vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_3] = shape_tem_addr[3] & 16383

            if npuop.NpuOp.NpuOpPool:
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h] = npuop.NpuOp.NpuOpPoolOp.KerH
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h_stride] = npuop.NpuOp.NpuOpPoolOp.StrideH

            elif npuop.NpuOp.NpuOpResize:
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h] = 2
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h_stride] = 1

            else:
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h_stride] = 1

            vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l1_step] = 1
            vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l2_step] = 1
            vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l3_step] = 1

            if npuop.int32_out:
                fmo_c = fmo_shape[2] * 4
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5] * 4
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]
            else:
                fmo_c = fmo_shape[2]
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5]
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

            vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode] = 1

            vpu_dict = vpu_jump(VpuRegister.vpu_0_para_clus_0_output_l1_condition,
                                VpuRegister.vpu_0_para_clus_0_output_l2_condition, \
                                VpuRegister.vpu_0_para_clus_0_output_l3_condition,
                                VpuRegister.vpu_0_para_clus_0_output_l1_addr_step, \
                                VpuRegister.vpu_0_para_clus_0_output_l2_addr_step,
                                VpuRegister.vpu_0_para_clus_0_output_l3_addr_step, \
                                VpuRegister.vpu_TOP_para_interface_write_mode, \
                                fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict, tot_bank=4)

            if npuop.npu_psum_add:
                vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode] = 0
            elif npuop.int8_out and not npuop.NpuOp.NpuOpPool and not npuop.NpuOp.NpuOpResize:
                if npuop.int32_out == False and len(sub_vpu_addr) == 1 and npuop.npu_op_id != 0:
                    vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode] = 1
                else:
                    vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode] = 2

            vpu_dict = vpu_write_mode_gen(npuop, sub_vpu_addr, VpuRegister.vpu_TOP_para_interface_write_mode, fmo_c,
                                          cluster_fmo_c, vpu_dict, tot_bank=4)

            if npuop.npu_psum_add:
                fmo_c = fmo_shape[2] * 4
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5] * 4
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_0] = p_sum_shape_tem_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_1] = p_sum_shape_tem_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_2] = p_sum_shape_tem_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_3] = p_sum_shape_tem_addr[3] & 16383

                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l3_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l1_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l2_step] = 1

                vpu_dict = vpu_jump(VpuRegister.vpu_0_para_clus_0_scr_l1_condition,
                                    VpuRegister.vpu_0_para_clus_0_scr_l2_condition, \
                                    VpuRegister.vpu_0_para_clus_0_scr_l3_condition,
                                    VpuRegister.vpu_0_para_clus_0_scr_l1_addr_step, \
                                    VpuRegister.vpu_0_para_clus_0_scr_l2_addr_step,
                                    VpuRegister.vpu_0_para_clus_0_scr_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

            if npuop.NpuOp.NpuOpElemwise:
                fmo_c = fmo_shape[2]
                fmo_w = fmo_shape[1]
                fmo_h = fmo_shape[0]
                cluster_fmo_c = shape_tem[5]
                cluster_fmo_w = shape_tem[3]
                cluster_fmo_h = shape_tem[1]

                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_0] = elemwise_shape_tem_addr[0] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_1] = elemwise_shape_tem_addr[1] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_2] = elemwise_shape_tem_addr[2] & 16383
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_3] = elemwise_shape_tem_addr[3] & 16383

                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l3_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l1_step] = 1
                vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l2_step] = 1

                vpu_dict = vpu_jump(VpuRegister.vpu_0_para_clus_0_scr_l1_condition,
                                    VpuRegister.vpu_0_para_clus_0_scr_l2_condition, \
                                    VpuRegister.vpu_0_para_clus_0_scr_l3_condition,
                                    VpuRegister.vpu_0_para_clus_0_scr_l1_addr_step, \
                                    VpuRegister.vpu_0_para_clus_0_scr_l2_addr_step,
                                    VpuRegister.vpu_0_para_clus_0_scr_l3_addr_step, \
                                    VpuRegister.vpu_TOP_para_interface_write_mode, \
                                    fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict,
                                    tot_bank=4)

            if npuop.NpuOp.NpuOpResize:
                resize_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Resize)
                resize_input_shape = resize_op['input_tile_address_list'].reshape(-1, 6)[0]
                resize_output_shape = resize_op['output_tile_address_list'].reshape(-1, 6)[0]
                tile_drop_line_list = resize_op['tile_drop_line_list'].reshape(-1, 4)[0]

                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_width_ratio] = int(
                    npuop.NpuOp.NpuOpResizeOp.ratio_w)
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_height_ratio] = int(
                    npuop.NpuOp.NpuOpResizeOp.ratio_h)
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_width] = resize_input_shape[3]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_height] = resize_input_shape[1]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_output_width] = resize_output_shape[3]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_output_height] = resize_output_shape[1]

                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_width] = tile_drop_line_list[0]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_height] = tile_drop_line_list[1]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_width] = tile_drop_line_list[2]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_height] = tile_drop_line_list[3]

                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_mode] = 15

            else:
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_width_ratio] = 0
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_height_ratio] = 0

            if npuop.NpuOp.NpuOpPool:
                pool_op = get_op_tile_info(False, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Pool)
                pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_width] = pool_input_shape[3]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_height] = pool_input_shape[1]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_width] = pool_output_shape[3]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_height] = pool_output_shape[1]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_width] = pool_tile_pad_list[2]
                vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_height] = pool_tile_pad_list[0]

            vpu_path = f'{path}/layer_{npuop.npu_op_id}/sub_block_{vpu_nn_sb}/vpu_param.txt'
            f = open(vpu_path, 'w')
            for sss in vpu_dict:
                f.write(f'{sss}\n')
            f.close()

            vpu_path = f'{path}/layer_{npuop.npu_op_id}/sub_block_{vpu_nn_sb}/vpu_param'
            vpu_nn_sb += 1
            vpu_dict_ = get_vpu_dict(vpu_dict)
            vpu_dict = vpu_param(vpu_dict_)
            if npuop.NpuOp.NpuOpActivate:
                vpu_dict = add_lut_dict(vpu_dict, npuop.NpuOp.NpuOpActivateOp.lut_dict)
            else:
                vpu_dict = add_lut_dict(vpu_dict, np.zeros([256]).astype(np.int8))

            vpu_writeTxt(vpu_dict, vpu_path)

            if sub_block_register_list is not None:
                sub_block_register_list[sub_block_num]['vpu_dict'] = vpu_dict_

    return sub_block_nums
