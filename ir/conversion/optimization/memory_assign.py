from ir.graph.Graph_IR import *
from ir.dialect.npu.IR_operator import *
from ir.conversion.ir_transform import _register_ir_transformation_rule
from enum import Enum
from backend.ada200.ada200 import ada200
import copy

SHARE_MEM_DICT = {0: [81920, 98304, 114688, 131072],
                  1: [147456, 163840, 180224, 196608],
                  2: [212992, 229376, 245760, 262144]}

backend = ada200()


class TransformRule(Enum):
    NOPE = 1
    TILE_MEMORY_ASSIGN = 2


@_register_ir_transformation_rule(TransformRule.TILE_MEMORY_ASSIGN)
def _tile_memory_assign(net: GraphIR):
    for group_op_id, group_op in enumerate(net.AllOps):
        for op in group_op.block_list:
            if isinstance(op, block_param):
                tile_pad_list = op.npu_op_flow_tile_address_list[0]["tile_pad_list"]
                tile_pad = tile_pad_list.reshape(-1, 4)[0]

                read_bank_group_id = op.shm_read_param_dict['tensor_info']['bank_group_id']
                write_bank_group_id = op.shm_write_param_dict['tensor_info']['bank_group_id']

                block_address_list = copy.deepcopy(op.shm_read_param_dict['tensor_info']['block_address_list'])
                if op.input_block:
                    block_address_list[-1] = 8

                tile_address_list = copy.deepcopy(op.npu_op_flow_tile_address_list[0]['input_tile_address_list'])
                if op.input_block:
                    tile_address_list[:, :, :, -1] = 8

                backend.tile_address_list_check(tile_address_list)

                backend.origin_shape_check(op.shm_read_param_dict['tensor_info']['origin_shape'])
                origin_shape = op.shm_read_param_dict['tensor_info']['origin_shape']

                npu_conv_op = op.get_npu_conv_op()

                # if npu_conv_op.StrideH==2 and npu_conv_op.KerH==1:
                #     a=1

                if npu_conv_op:
                    InputH, InputW, InputC = npu_conv_op.InputH, npu_conv_op.InputW, npu_conv_op.InputC
                else:
                    InputH, InputW, InputC = op.NpuOp.InputH, op.NpuOp.InputW, op.NpuOp.InputC

                block_h_s, block_w_s, block_c_s = block_address_list[0], block_address_list[2], block_address_list[4]
                if op.input_block:
                    origin_shape = [origin_shape[0], origin_shape[1], 8]
                    block_base_addr = int(op.shm_read_param_dict['tensor_info']['addr'] / 8 / 1)

                    block_base_addr += int((block_h_s * origin_shape[1] * origin_shape[2] + block_w_s * origin_shape[
                        2] + block_c_s) / 8 / 1)
                else:
                    block_base_addr = int(op.shm_read_param_dict['tensor_info']['addr'] / 8 / 4)
                    block_base_addr += int((block_h_s * origin_shape[1] * origin_shape[2] + block_w_s * origin_shape[
                        2] + block_c_s) / 8 / 4)
                t_h, t_w, t_c = op.tile_split_mode.h, op.tile_split_mode.w, op.tile_split_mode.c
                tile_memory_address_list = np.zeros([t_h, t_w, t_c, 1], dtype=np.int32)

                if op.input_block:
                    for i in range(t_h):
                        for j in range(t_w):
                            for k in range(t_c):
                                h_start = tile_address_list[i][j][k][0]
                                w_start = tile_address_list[i][j][k][2]
                                c_start = tile_address_list[i][j][k][4]
                                tile_base_addr = block_base_addr

                                tile_base_addr += int((h_start * origin_shape[1] * origin_shape[2] + w_start *
                                                       origin_shape[2] + c_start) / 8 / 1)
                                tile_memory_address_list[i][j][k] = tile_base_addr
                else:
                    for i in range(t_h):
                        for j in range(t_w):
                            for k in range(t_c):
                                h_start = tile_address_list[i][j][k][0]
                                w_start = tile_address_list[i][j][k][2]
                                c_start = tile_address_list[i][j][k][4]
                                tile_base_addr = block_base_addr

                                tile_base_addr += int((h_start * origin_shape[1] * origin_shape[2] + w_start *
                                                       origin_shape[2] + c_start) / 8 / 4)
                                tile_memory_address_list[i][j][k] = tile_base_addr

                read_tile_memory_address_list = np.zeros([t_h, t_w, t_c, 4], dtype=np.int32)
                for i in range(t_h):
                    for j in range(t_w):
                        for k in range(t_c):
                            bank_info = SHARE_MEM_DICT[read_bank_group_id]
                            read_tile_memory_address_list[i][j][k] = [
                                bank_info[o] + (tile_memory_address_list[i][j][k]).tolist()[0] for o in
                                range(len(bank_info))]

                op.shm_read_param_dict['read_tile_memory_address_list'] = read_tile_memory_address_list

                vpu_shape_list = op.tile_output_vpu_shape_list
                write_vpu_memory_address_list = np.zeros([t_h, t_w, t_c, 4], dtype=np.int32)
                vpu_memory_address_list = np.zeros([t_h, t_w, t_c, 1], dtype=np.int32)
                vpu_base_addr = op.shm_write_param_dict['tensor_info']['addr']

                for i in range(t_h):
                    for j in range(t_w):
                        for k in range(t_c):
                            vpu_shape = vpu_shape_list[i][j][k]
                            vpu_memory_address_list[i][j][k] = vpu_base_addr
                            vpu_base_addr += int(vpu_shape[1] * vpu_shape[3] * vpu_shape[5] / 8 / 4)

                for i in range(t_h):
                    for j in range(t_w):
                        for k in range(t_c):
                            bank_info = SHARE_MEM_DICT[write_bank_group_id]
                            write_vpu_memory_address_list = [bank_info[o] + vpu_memory_address_list[i][j][k] for o in
                                                             range(len(bank_info))]

                op.shm_write_param_dict['write_vpu_memory_address_list'] = write_vpu_memory_address_list


# memory_assign_pass
memory_assign_transform = [TransformRule.TILE_MEMORY_ASSIGN
                           ]
