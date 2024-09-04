# from compiler.frontend.tflite.tflite_convertor import *
from ir.graph.Graph_IR import *
from ir.dialect.top.IR_operator import *
from ir.dialect.top.IR_tensor import *
import pandas as pd


def get_top_graph_op_list(top_graph):
    keyword = ["op", "input_h", "input_w", "input_c",
               "output_h", "output_w", "output_c", "ker_h", "ker_w",
               "stride_h", "stride_w", "padding", "pad_h", "pad_w",
               "group", "bias", "input_idx", "ouput_idx", "MAC",
               "fmi_data_size", "fmo_data_size", "kernel_data_size"]

    graph_op_list = []
    for op in top_graph.AllOps:
        base_dict = {k: '' for k in keyword}
        base_dict['op'] = op.Name
        base_dict['input_h'] = op.InputH
        base_dict['input_w'] = op.InputW
        base_dict['input_c'] = op.InputC
        base_dict['output_h'] = op.OutputH
        base_dict['output_w'] = op.OutputW
        base_dict['output_c'] = op.OutputC

        if isinstance(op, ConvBase) or isinstance(op, Pool):
            base_dict['ker_h'] = op.KerH
            base_dict['ker_w'] = op.KerW
            base_dict['stride_h'] = op.StrideH
            base_dict['stride_w'] = op.StrideW
            base_dict['padding'] = op.Padding
            base_dict['pad_h'] = op.PadH
            base_dict['pad_w'] = op.PadW

        if isinstance(op, ConvBase) or isinstance(op, FullConnected):
            base_dict['group'] = op.Group if isinstance(op, ConvBase) else None
            base_dict['bias'] = op.Bias
            base_dict['MAC'] = op.get_mac()
            base_dict['kernel_data_size'] = op.get_weight_size() / 1024

        base_dict['input_idx'] = op.InTensors
        base_dict['ouput_idx'] = op.OutTensors

        base_dict['fmi_data_size'] = op.get_fmi_size() / 1024
        base_dict['fmo_data_size'] = op.get_fmo_size() / 1024

        graph_op_list.append(base_dict)

    return graph_op_list


# def get_npu_graph_op_list(npu_graph):

#     keyword = ["op", "input_h", "input_w", "input_c", 
#                 "output_h", "output_w", "output_c", "ker_h", "ker_w",
#                 "fmi_tensor", "fmi1_tensor", "weight_tensor", 
#                 "fmo_tensor", "concat_input_tensor", "short_cut_out_tensor",
#                 "fmi_tensor_size", "fmi1_tensor_size", "fmo_tensor_size",
#                 "short_cut_out_tensor_size", "weight_tensor_size",]


def graph_to_df(graph_list):
    graph_df = pd.DataFrame(graph_list)
    return graph_df


def df_to_excel(graph_df, path):
    graph_df.to_excel(path)


def group_to_block(group_list):
    model_block_list = []
    for npu_group_id, npu_group in enumerate(group_list):
        block_num = npu_group.block_split_mode.split_num
        block_list = npu_group.block_list
        npu_op_list_len = len(npu_group.npu_op_list)
        for block_id in range(block_num):
            for i, npu_op in enumerate(npu_group.npu_op_list):
                npu_op_id = npu_op.TimeStep
                block = block_list[block_id * npu_op_list_len + i]
                block.npu_op_id = npu_op_id
                block.npu_group_id = npu_group_id
                model_block_list.append(block)

    return model_block_list


def get_block_param_list(block_list):
    block_param_list = []
    for blk_id, block in enumerate(block_list):
        block_param_dict = {}
        block_param_dict["npu_op_id"] = block.npu_op_id
        block_param_dict["npu_group_id"] = block.npu_group_id
        block_param_dict["npu_block_id"] = blk_id
        block_param_dict["top_op_id_list"] = [npu_op.TopOpId for npu_op in block.NpuOp.NpuOpFlow]

        group_h_s = block.block_split_mode.h
        group_w_s = block.block_split_mode.w
        group_c_s = block.block_split_mode.c
        block_param_dict["block_split_mode"] = "h:{}, w:{}, c:{}".format(group_h_s, group_w_s, group_c_s)
        block_address_list = block.get_input_tensor_info()['block_address_list']
        block_h = block_address_list[1]
        block_w = block_address_list[3]
        block_c = block_address_list[5]

        block_param_dict["block_h"] = block_h
        block_param_dict["block_w"] = block_w
        block_param_dict["block_c"] = block_c

        npu_conv_op = block.get_npu_conv_op()
        if npu_conv_op is not None:
            k_h = npu_conv_op.KerH
            k_w = npu_conv_op.KerW
            k_c = int(npu_conv_op.InputC / block.block_split_mode.c)
            k_n = npu_conv_op.OutputC
            k_s = npu_conv_op.StrideH
        else:
            k_h = None
            k_w = None
            k_c = None
            k_n = None
            k_s = None

        block_param_dict['kernel_h'] = k_h
        block_param_dict['kernel_w'] = k_w
        block_param_dict['kernel_c'] = k_c
        block_param_dict['kernel_n'] = k_n
        block_param_dict['stride'] = k_s

        block_param_dict['shm_read_bank_group_id'] = block.shm_read_param_dict["tensor_info"]['bank_group_id']
        block_param_dict['shm_read_npu_ir_tensor'] = block.shm_read_param_dict["tensor_info"]['tensor_id']
        block_param_dict['shm_read_addr'] = block.shm_read_param_dict["tensor_info"]["addr"]
        block_param_dict['shm_read_len'] = block.shm_read_param_dict["tensor_info"]["len"]
        if block.shm_read_param_dict.get("read_tile_memory_address_list", None) is not None:
            block_param_dict['shm_read_bank_addr_list'] = block.shm_read_param_dict['read_tile_memory_address_list']
        else:
            block_param_dict['shm_read_bank_addr_list'] = None

        block_param_dict['shm_write_bank_group_id'] = block.shm_write_param_dict["tensor_info"]['bank_group_id']
        block_param_dict['shm_write_npu_ir_tensor'] = block.shm_write_param_dict["tensor_info"]['tensor_id']
        block_param_dict['shm_write_addr'] = block.shm_write_param_dict["tensor_info"]["addr"]
        block_param_dict['shm_write_len'] = block.shm_write_param_dict["tensor_info"]["len"]
        if block.shm_write_param_dict.get("write_vpu_memory_address_list", None) is not None:
            block_param_dict['shm_write_bank_addr_list'] = block.shm_write_param_dict['write_vpu_memory_address_list']
        else:
            block_param_dict['shm_write_bank_addr_list'] = None

        if block.short_cut_shm_write_param_dict["tensor_info"] is not None:
            block_param_dict['short_cut_out_write_bank_group_id'] = block.short_cut_shm_write_param_dict["tensor_info"][
                'bank_group_id']
            block_param_dict['short_cut_out_write_npu_ir_tensor'] = block.short_cut_shm_write_param_dict["tensor_info"][
                'tensor_id']
            block_param_dict['short_cut_out_addr'] = block.short_cut_shm_write_param_dict["tensor_info"]["addr"]
            block_param_dict['short_cut_out_len'] = block.short_cut_shm_write_param_dict["tensor_info"]["len"]
            # block_param_dict['short_cut_out_write_bank_addr_list'] = block.short_cut_out_write_dict['bank_addr_list']
        else:
            block_param_dict['short_cut_out_write_bank_group_id'] = None
            block_param_dict['short_cut_out_write_npu_ir_tensor'] = None
            block_param_dict['short_cut_out_addr'] = None
            block_param_dict['short_cut_out_len'] = None

        if block.shm_psum_read_param_dict['tensor_info']:
            block_param_dict['npu_psum_read_bank_group_id'] = block.shm_psum_read_param_dict['tensor_info'][
                'bank_group_id']
            block_param_dict['npu_psum_read_npu_ir_tensor'] = block.shm_psum_read_param_dict['tensor_info']['tensor_id']
            block_param_dict['npu_psum_read_addr'] = block.shm_psum_read_param_dict['tensor_info']["addr"]
            block_param_dict['npu_psum_read_len'] = block.shm_psum_read_param_dict['tensor_info']["len"]
            # block_param_dict['npu_psum_read_bank_addr_list'] = block.npu_psum_read_dict['bank_addr_list']
        else:
            block_param_dict['npu_psum_read_bank_group_id'] = None
            block_param_dict['npu_psum_read_npu_ir_tensor'] = None
            block_param_dict['npu_psum_read_addr'] = None
            block_param_dict['npu_psum_read_len'] = None

        if block.weight_mapping_dict.get("cluster_psum", None) is not None:
            block_param_dict["cluster_psum"] = block.weight_mapping_dict['cluster_psum']
            block_param_dict["cluster_win_num"] = block.weight_mapping_dict['cluster_win_num']
            block_param_dict["cim_psum"] = block.weight_mapping_dict['cim_psum']
            block_param_dict["bit_map"] = block.weight_mapping_dict['bit_map']
            block_param_dict["tile_num"] = block.tile_split_mode.split_num
            block_param_dict['weight_format'] = block.weight_mapping_dict['weight_format']
        else:
            block_param_dict["cluster_psum"] = None
            block_param_dict["cluster_win_num"] = None
            block_param_dict["cim_psum"] = None
            block_param_dict["bit_map"] = None
            block_param_dict["tile_num"] = None
            block_param_dict['weight_format'] = None

        if block.tile_split_mode:
            tile_h_s = block.tile_split_mode.h
            tile_w_s = block.tile_split_mode.w
            tile_c_s = block.tile_split_mode.c
        else:
            tile_h_s = tile_w_s = tile_c_s = None
        block_param_dict["tile_split_mode"] = "h:{}, w:{}, c:{}".format(tile_h_s, tile_w_s, tile_c_s)

        block_param_dict["tile_nums"] = block.tile_split_mode.split_num
        if block_param_dict.get('kernel_c', None) is not None:
            block_param_dict["need_line_buffer_size_kB"] = block_param_dict['kernel_c'] * block_param_dict["block_w"] \
                                                           * (block_param_dict['kernel_h'] + block_param_dict[
                'stride']) / 1024 \
                if block_param_dict["block_w"] else None
        else:
            block_param_dict["need_line_buffer_size_kB"] = None
        block_param_dict["all_line_buffer_size_kB"] = 16 * 4 / block_param_dict["tile_nums"] if block_param_dict[
            "tile_nums"] else None

        block_param_dict["bias_add"] = block.bias_add
        block_param_dict["int32_out"] = block.int32_out
        block_param_dict["npu_psum_add"] = block.npu_psum_add
        block_param_dict["layer_group_flag"] = block.layer_group_flag
        block_param_dict['input_block'] = block.input_block
        block_param_dict['output_block'] = block.output_block

        block_param_list.append(block_param_dict)

    return block_param_list


def param_to_txt(model_name, block_param_list):
    """
    {layer_name},{total_layer_num},{total_input_layer_num},{total_output_layer_num}
    {layer_0_block_num},{layer_1_block_num}, .... ,{layer_n_block_num}
    {input_layer_index}
    {output_layer_index}
    """
    total_input_layer_num = 0
    total_output_layer_num = 0
    total_layer_num = len(block_param_list)
    for p in block_param_list:
        if p.input_block:
            total_input_layer_num += 1
        if p.output_block:
            total_output_layer_num += 1

    first_line = "{},{},{},{}".format(model_name, total_layer_num, total_input_layer_num, total_output_layer_num)