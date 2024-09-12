from ir.graph.Graph_IR import *
from ir.dialect.top.IR_tensor import *
from ir.dialect.npu.IR_operator import *
from ir.dialect.npu.IR_memory import *
from ir.conversion.optimization.memory.base import *
from ir.conversion.ir_transform import _register_ir_transformation_rule


@_register_ir_transformation_rule(TransformRule.NPU_TENSOR_LIFE_CYCLE_REPORT)
def _npu_tensor_life_cycle_report(net: GraphIR):
    print("--------TransformRule.NPU_TENSOR_LIFE_CYCLE_REPORT---------")

    for group_op_id, group_op in enumerate(net.AllOps):

        print("---------------------------------------")
        print("group_op_id: ", group_op_id)

        for op_block in group_op.block_list:
            if isinstance(op_block, block_param):
                block_id = op_block.npu_op_block_id
                print("block_id: ", block_id)

                # shortcut_tensor
                if op_block.NpuOp.NpuOpShortCutOut:
                    assert len(op_block.NpuOp.short_cut_out_tensor) == 1
                    short_cut_tensor_id = op_block.NpuOp.short_cut_out_tensor[0]
                    short_cut_tensor = net.get_tensor(short_cut_tensor_id)
                    tensor_info = "short_cut_tensor_id: {}, life_cycle: {}" \
                        .format(short_cut_tensor_id,
                                short_cut_tensor.tensor_life_cycle_for_block_list)
                    print(tensor_info)

                # input_tensor
                input_tensor_info = op_block.get_input_tensor_info()
                input_tensor_id = input_tensor_info['tensor_id']
                group_block_id = input_tensor_info['group_block_id']
                assert input_tensor_id == op_block.NpuOp.fmi_tensor[0]
                input_tensor = net.get_tensor(input_tensor_id)
                tensor_info = "input_tensor_id: {}, life_cycle: {}" \
                    .format(input_tensor_id,
                            input_tensor.tensor_life_cycle_for_block_list)
                print(tensor_info)
                if op_block.block_split_mode.split_num > 1:
                    imd_tensor_life_cycle = input_tensor.get_intermediate_tensor_life_cycle(group_block_id)
                    tensor_info = "input_intermediate_tensor_id: {}, group_block_id: {}, life_cycle: {}" \
                        .format(input_tensor_id,
                                group_block_id,
                                imd_tensor_life_cycle)
                    print(tensor_info)

                # concat_input_tensor
                if len(op_block.NpuOp.concat_input_tensor) > 0:
                    concat_input_tensor_id = op_block.get_concat_input_tensor_id()
                    concat_input_tensor = net.get_tensor(concat_input_tensor_id)
                    tensor_info = "concat_input_tensor_id: {}, life_cycle: {}" \
                        .format(concat_input_tensor_id,
                                concat_input_tensor.tensor_life_cycle_for_block_list)
                    print(tensor_info)

                if len(op_block.NpuOp.elemwise_input_tensor) > 0:
                    elemwise_input_tensor_id = op_block.get_elemwise_input_tensor_id()
                    elemwise_input_tensor = net.get_tensor(elemwise_input_tensor_id)
                    tensor_info = "elemwise_input_tensor_id: {}, life_cycle: {}" \
                        .format(elemwise_input_tensor_id,
                                elemwise_input_tensor.tensor_life_cycle_for_block_list)
                    print(tensor_info)

                output_tensor_info = op_block.get_output_tensor_info()
                output_tensor_id = output_tensor_info['tensor_id']
                group_block_id = output_tensor_info['group_block_id']
                assert output_tensor_id == op_block.NpuOp.fmo_tensor[0]
                output_tensor = net.get_tensor(output_tensor_id)
                tensor_info = "output_tensor_id: {}, life_cycle: {}" \
                    .format(output_tensor_id,
                            output_tensor.tensor_life_cycle_for_block_list)
                print(tensor_info)
                if op_block.block_split_mode.split_num > 1:
                    imd_tensor_life_cycle = output_tensor.get_intermediate_tensor_life_cycle(group_block_id)
                    tensor_info = "output_intermediate_tensor_id: {}, group_block_id: {}, life_cycle: {}" \
                        .format(output_tensor_id,
                                group_block_id,
                                imd_tensor_life_cycle)
                    print(tensor_info)


@_register_ir_transformation_rule(TransformRule.CHECK_BLOCK_AND_TILE_SHAPE)
def _check_block_and_tile_shape(net: GraphIR):
    with Capturing() as block_and_tile_shape_loger:
        print("--------TransformRule.CHECK_BLOCK_AND_TILE_SHAPE---------")

        for group_op in net.AllOps:
            for op in group_op.block_list:
                if isinstance(op, block_param):
                    block_id = op.npu_op_block_id
                    print("block_id: ", block_id)
                    print("input memory allocate")
                    print("shm_read_param_dict: ", op.shm_read_param_dict)

                    if op.shm_psum_read_param_dict.get('tensor_info', None) is not None:
                        print("shm_psum_read_param_dict: ", op.shm_psum_read_param_dict)

                    if op.elemwise_input:
                        print("elemwise_input memory allocate")
                        print("elemwise_input: ", op.elemwise_input)
                        print("shm_elemwise_read_param_dict: ", op.shm_elemwise_read_param_dict)

                    if op.concat_input:
                        print("concat_input memory allocate")
                        print("concat_input: ", op.concat_input)
                        print("shm_write_param_dict: ", op.shm_write_param_dict)

                    else:
                        print("output memory allocate")
                        print("group_block_id:", op.group_block_id)
                        print("shm_write_param_dict: ", op.shm_write_param_dict)

                    if op.short_cut_out:
                        print("short_cut_out memory allocate")
                        print("short_cut_shm_write_param_dict: ", op.short_cut_shm_write_param_dict)

                    print("npu_op_flow_block_address_list:")
                    for bl in op.npu_op_flow_block_address_list:
                        if bl.get("input_tensor_info", None):
                            print("input_tensor_info:", bl['input_tensor_info'])

                        if bl.get("output_tensor_info", None):
                            print("output_tensor_info:", bl['output_tensor_info'])

                        print("input_block_address_list:", bl["input_block_address_list"])
                        print("output_block_address_list", bl["output_block_address_list"])
                        print("----------------------             -----------------------")
                    print("-------------            ----------------               ----------")

                    print("npu_op_flow_tile_address_list:")
                    for tl in op.npu_op_flow_tile_address_list:
                        print("input_tile_address_list:", tl["input_tile_address_list"])
                        print("output_tile_address_list:", tl["output_tile_address_list"])
                        print("tile_pad_list:", tl["tile_pad_list"])
                        tile_drop_line_list = tl.get("tile_drop_line_list", None)
                        if tile_drop_line_list is not None:
                            print("tile_drop_line_list:", tile_drop_line_list)
                        print("----------------------             -----------------------")
                    print("------------------------------------------------------------------")

    net.block_and_tile_shape_loger_list = [str(f) for f in block_and_tile_shape_loger]

    for f in net.block_and_tile_shape_loger_list:
        print(f)
