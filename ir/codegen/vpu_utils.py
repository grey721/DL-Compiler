from ir.codegen.vpuPreparingParam import VpuRegister
from ir.dialect.npu.IR_operator import *
import math
from ir.codegen.utils import *


def vpu_write_mode_gen(npuop, sub_vpu_addr, vpu_TOP_para_interface_write_mode, fmo_c, cluster_fmo_c, vpu_dict,
                       tot_bank=4):
    if tot_bank == 4:
        if npuop.npu_psum_add:
            vpu_dict[vpu_TOP_para_interface_write_mode] = 0
        elif fmo_c == 16:
            vpu_dict[vpu_TOP_para_interface_write_mode] = 2
        elif fmo_c == 32 and cluster_fmo_c == 16:
            vpu_dict[vpu_TOP_para_interface_write_mode] = 3
        elif fmo_c == 64 and cluster_fmo_c == 16:
            vpu_dict[vpu_TOP_para_interface_write_mode] = 3
        elif fmo_c > 64 and cluster_fmo_c == 16:
            vpu_dict[vpu_TOP_para_interface_write_mode] = 3
        else:
            if npuop.int8_out and not npuop.NpuOp.NpuOpPool and not npuop.NpuOp.NpuOpResize:
                if npuop.int32_out == False and len(sub_vpu_addr) == 1 and npuop.npu_op_id != 0:
                    vpu_dict[vpu_TOP_para_interface_write_mode] = 1
                # else:
                #     vpu_dict[vpu_TOP_para_interface_write_mode] = 2
    return vpu_dict


def vpu_jump(l1_condition, l2_condition, l3_condition, l1_addr_step, l2_addr_step, l3_addr_step,
             vpu_TOP_para_interface_write_mode, \
             fmo_c, fmo_w, fmo_h, cluster_fmo_c, cluster_fmo_w, cluster_fmo_h, vpu_dict, tot_bank=4,
             memory_padding=False):
    if memory_padding:
        vpu_dict[VpuRegister.vpu_SF_para_odd_output_enable] = 1

    if tot_bank == 4:  # 32 Channels/ Addr
        if fmo_c == 16:  # 2 width pixal / Addr
            vpu_dict[vpu_TOP_para_interface_write_mode] = 2
            vpu_dict[l1_condition] = math.ceil(cluster_fmo_w / 2)
            vpu_dict[l2_condition] = cluster_fmo_h
            vpu_dict[l3_condition] = 1

            vpu_dict[l1_addr_step] = 1
            vpu_dict[l2_addr_step] = math.ceil(fmo_w * (fmo_c / 32))
            vpu_dict[l3_addr_step] = 0
        elif fmo_c == 32:
            if cluster_fmo_c == 16:
                vpu_dict[vpu_TOP_para_interface_write_mode] = 3
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1

                vpu_dict[l1_addr_step] = 1
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

            elif cluster_fmo_c == 32:

                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1

                vpu_dict[l1_addr_step] = 1
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

        elif fmo_c == 64:
            if cluster_fmo_c == 16:
                vpu_dict[vpu_TOP_para_interface_write_mode] = 3
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1
                vpu_dict[l1_addr_step] = 2
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

            elif cluster_fmo_c == 32:
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1
                vpu_dict[l1_addr_step] = 2
                vpu_dict[l2_addr_step] = fmo_w * 2
                vpu_dict[l3_addr_step] = 0

            elif cluster_fmo_c == 64:  # TB_Testing
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1
                vpu_dict[l1_addr_step] = 2
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

        elif fmo_c > 64:
            if cluster_fmo_c == 16:
                vpu_dict[vpu_TOP_para_interface_write_mode] = 3
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1

                vpu_dict[l1_addr_step] = int(int(fmo_c / 32))
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

            elif cluster_fmo_c == 32:  # TB_Testing
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1

                vpu_dict[l1_addr_step] = int(int(fmo_c / 32))
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0

            elif cluster_fmo_c == 64:  # TB_Testing
                vpu_dict[l1_condition] = cluster_fmo_w
                vpu_dict[l2_condition] = cluster_fmo_h
                vpu_dict[l3_condition] = 1

                vpu_dict[l1_addr_step] = int(int(fmo_c / 32))
                vpu_dict[l2_addr_step] = int(int(fmo_w * (fmo_c / 32)))
                vpu_dict[l3_addr_step] = 0
    return vpu_dict


def NpuOpPool_gen(npuop, vpu_dict, vpu_id, vpu_j_para_i_clus_j_pooling_param_input_width, \
                  vpu_j_para_i_clus_j_pooling_param_input_height, vpu_j_para_i_clus_j_pooling_param_output_width, \
                  vpu_j_para_i_clus_j_pooling_param_output_height, vpu_j_para_i_clus_j_pooling_padding_width,
                  vpu_j_para_i_clus_j_pooling_padding_height):
    if npuop.NpuOp.NpuOpPool and npuop.int32_out == False:

        pool_op = get_op_tile_info(npuop.int32_out, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list, Pool)
        weight_format = npuop.weight_mapping_dict["weight_format"]
        cluster_p_sum = npuop.weight_mapping_dict['cluster_psum']
        if vpu_id == 3:
            pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[-1]
            pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[-1]
            pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[-1]
        elif vpu_id == 2:
            pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)
            pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)
            if len(pool_input_shape) == 4:
                if weight_format[0][2] * weight_format[0][3] == 8 and cluster_p_sum != 2:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
                else:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[2]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[2]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[2]
            elif len(pool_input_shape) == 2:
                if weight_format[0][2] * weight_format[0][3] == 8 and cluster_p_sum != 2:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
                else:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
            elif len(pool_input_shape) == 1:
                pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
        elif vpu_id == 1:
            pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)
            pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)
            if len(pool_input_shape) == 4:
                if weight_format[0][2] * weight_format[0][3] == 8 and cluster_p_sum != 2:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[2]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[2]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[2]
                else:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
            elif len(pool_input_shape) == 2:
                if weight_format[0][2] * weight_format[0][3] == 8 and cluster_p_sum != 2:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[1]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[1]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[1]
                else:
                    pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                    pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                    pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
            elif len(pool_input_shape) == 1:
                pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
                pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
                pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]
        elif vpu_id == 0:
            pool_op = get_op_tile_info(npuop.int32_out, npuop.NpuOp.NpuOpFlow, npuop.npu_op_flow_tile_address_list,
                                       Pool)
            pool_input_shape = pool_op['input_tile_address_list'].reshape(-1, 6)[0]
            pool_output_shape = pool_op['output_tile_address_list'].reshape(-1, 6)[0]
            pool_tile_pad_list = pool_op['tile_pad_list'].reshape(-1, 4)[0]

        vpu_dict[vpu_j_para_i_clus_j_pooling_param_input_width] = pool_input_shape[3]
        vpu_dict[vpu_j_para_i_clus_j_pooling_param_input_height] = pool_input_shape[1]

        vpu_dict[vpu_j_para_i_clus_j_pooling_param_output_width] = pool_output_shape[3]
        vpu_dict[vpu_j_para_i_clus_j_pooling_param_output_height] = pool_output_shape[1]

        vpu_dict[vpu_j_para_i_clus_j_pooling_padding_width] = pool_tile_pad_list[2]
        vpu_dict[vpu_j_para_i_clus_j_pooling_padding_height] = pool_tile_pad_list[0]
