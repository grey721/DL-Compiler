from ir.graph.Graph_IR import *
from ir.dialect.top.IR_tensor import *
from ir.dialect.npu.IR_operator import *
from ir.dialect.npu.IR_memory import *
from ir.conversion.optimization.memory.base import *
from ir.conversion.ir_transform import _register_ir_transformation_rule


def _update_intermediate_tensor_life_cycle_for_block_list(input_tensor, op_block):

    block_id = op_block.npu_op_block_id
    group_block_id = op_block.group_block_id

    if not input_tensor.check_intermediate_tensor(op_block.group_block_id):
        tensor_life_cycle_list = [-1, -1]
        tensor_life_cycle_list[0] = block_id
        tensor_life_cycle_list[1] = block_id
        input_tensor.add_intermediate_tensor_life_cycle(group_block_id, tensor_life_cycle_list)
    else:
        tensor_life_cycle_list = input_tensor.get_intermediate_tensor_life_cycle(group_block_id)
        start_block_id = tensor_life_cycle_list[0]
        end_block_id = tensor_life_cycle_list[1]

        if start_block_id != -1 and end_block_id != -1:
            if block_id < start_block_id:
                tensor_life_cycle_list[0] = block_id
        
            if block_id > end_block_id:
                tensor_life_cycle_list[1] = block_id
        else:
            tensor_life_cycle_list[0] = block_id
            tensor_life_cycle_list[1] = block_id


def _update_tensor_life_cycle_for_block_list(input_tensor, op_block):

    if op_block.block_split_mode.split_num > 1:
        _update_intermediate_tensor_life_cycle_for_block_list(input_tensor, op_block)

    block_id = op_block.npu_op_block_id

    if not isinstance(input_tensor.tensor_life_cycle_for_block_list, list):
        tensor_life_cycle_for_block_list = [0, 0]
        tensor_life_cycle_for_block_list[0] = block_id
        tensor_life_cycle_for_block_list[1] = block_id
        input_tensor.tensor_life_cycle_for_block_list = tensor_life_cycle_for_block_list
    
    else:
        start_block_id = input_tensor.tensor_life_cycle_for_block_list[0]
        end_block_id = input_tensor.tensor_life_cycle_for_block_list[1]

        if start_block_id != -1 and end_block_id != -1:
            if block_id < start_block_id:
                input_tensor.tensor_life_cycle_for_block_list[0] = block_id
        
            if block_id > end_block_id:
                input_tensor.tensor_life_cycle_for_block_list[1] = block_id
        else:
            input_tensor.tensor_life_cycle_for_block_list[0] = block_id
            input_tensor.tensor_life_cycle_for_block_list[1] = block_id


def _update_tensor_life_cycle(net, tensor_id, op_block):

    tensor = net.AllTensors[tensor_id]
    if not isinstance(tensor, NpuIRTensor):
        tensor = NpuIRTensor(tensor)
        _update_tensor_life_cycle_for_block_list(tensor, op_block)
        net.replace_tensor(tensor, tensor_id)
    else:
        _update_tensor_life_cycle_for_block_list(tensor, op_block)


@_register_ir_transformation_rule(TransformRule.GEN_NPU_OP_TIME_STEP)
def _gen_npu_op_time_step(net: GraphIR):

    print("---------TransformRule.GEN_NPU_OP_TIME_STEP--------")

    for t, op in  enumerate(net.AllOps):
        op.set_time_step(t)


@_register_ir_transformation_rule(TransformRule.UPDATE_CONCAT_TENSOR_LIFE_CYCLE)
def _update_concat_tensor_life_cycle(net: GraphIR):

    print("--------TransformRule.UPDATE_CONCAT_TENSOR_LIFE_CYCLE---------")

    npu_op_group_list = []
    for _npu_op_group in net.AllOps:
        assert isinstance(_npu_op_group, npu_op_group)
        npu_op_group_list.append(_npu_op_group)

    npu_op_block_list = []
    for op_group in npu_op_group_list:
        npu_op_block_list.extend(op_group.block_list)


    for block_id, op_block in enumerate(npu_op_block_list):
        op_block.npu_op_block_id = block_id
        print("---------------------------------------—--------")
        print("block_id:", block_id)

        if len(op_block.NpuOp.concat_input_tensor) > 0:
            concat_input_tensor_id = op_block.get_concat_input_tensor_id()
            concat_input_tensor = net.get_tensor(concat_input_tensor_id)
            output_tensor_id = op_block.NpuOp.fmo_tensor[0]
            output_tensor = net.get_tensor(output_tensor_id)
            end_block_id = output_tensor.tensor_life_cycle_for_block_list[1]
            concat_input_tensor.tensor_life_cycle_for_block_list[1] = end_block_id
            tensor_info = "concat_input_tensor_id: {}, life_cycle: {}"\
                            .format(concat_input_tensor_id, 
                                    concat_input_tensor.tensor_life_cycle_for_block_list)
            print(tensor_info)


@_register_ir_transformation_rule(TransformRule.GEN_NPU_TENSOR_LIFE_CYCLE)
def _gen_npu_tensor_life_cycle(net: GraphIR):

    print("--------TransformRule.GEN_NPU_TENSOR_LIFE_CYCLE---------")

    npu_op_group_list = []
    for _npu_op_group in net.AllOps:
        assert isinstance(_npu_op_group, npu_op_group)
        npu_op_group_list.append(_npu_op_group)

    npu_op_block_list = []
    for op_group in npu_op_group_list:
        npu_op_block_list.extend(op_group.block_list)

    for block_id, op_block in enumerate(npu_op_block_list):
        op_block.npu_op_block_id = block_id
        print("---------------------------------------—--------")
        print("block_id:", block_id)

        ## shortcut_tensor
        if op_block.NpuOp.NpuOpShortCutOut:
            assert len(op_block.NpuOp.short_cut_out_tensor) == 1

            short_cut_tensor_id = op_block.NpuOp.short_cut_out_tensor[0]
            _update_tensor_life_cycle(net, short_cut_tensor_id, op_block)
            short_cut_tensor = net.get_tensor(short_cut_tensor_id)
            tensor_info = "short_cut_tensor_id: {}, life_cycle: {}"\
                                .format(short_cut_tensor_id, 
                                    short_cut_tensor.tensor_life_cycle_for_block_list)
            print(tensor_info)

        ## input_tensor
        input_tensor_info = op_block.get_input_tensor_info()
        input_tensor_id = input_tensor_info['tensor_id']
        group_block_id = input_tensor_info['group_block_id']
        assert input_tensor_id == op_block.NpuOp.fmi_tensor[0]

        _update_tensor_life_cycle(net, input_tensor_id, op_block)
        input_tensor = net.get_tensor(input_tensor_id)
        tensor_info = "input_tensor_id: {}, life_cycle: {}"\
                        .format(input_tensor_id, 
                                input_tensor.tensor_life_cycle_for_block_list)
        print(tensor_info)
        if op_block.block_split_mode.split_num > 1:
            imd_tensor_life_cycle = input_tensor.get_intermediate_tensor_life_cycle(group_block_id)
            tensor_info = "input_intermediate_tensor_id: {}, group_block_id: {}, life_cycle: {}"\
                            .format(input_tensor_id,
                                    group_block_id,
                                    imd_tensor_life_cycle)
            print(tensor_info)


        ## concat_input_tensor
        if len(op_block.NpuOp.concat_input_tensor) > 0:
            concat_input_tensor_id = op_block.get_concat_input_tensor_id()
            _update_tensor_life_cycle(net, concat_input_tensor_id, op_block)
            concat_input_tensor = net.get_tensor(concat_input_tensor_id)
            tensor_info = "concat_input_tensor_id: {}, life_cycle: {}"\
                            .format(concat_input_tensor_id, 
                                    concat_input_tensor.tensor_life_cycle_for_block_list)
            print(tensor_info)


        ## elemwise_input_tensor
        if len(op_block.NpuOp.elemwise_input_tensor) > 0:
            elemwise_input_tensor_id = op_block.get_elemwise_input_tensor_id()
            _update_tensor_life_cycle(net, elemwise_input_tensor_id, op_block)
            elemwise_input_tensor = net.get_tensor(elemwise_input_tensor_id)
            tensor_info = "elemwise_input_tensor_id: {}, life_cycle: {}"\
                            .format(elemwise_input_tensor_id, 
                                    elemwise_input_tensor.tensor_life_cycle_for_block_list)
            print(tensor_info)
            

        output_tensor_info = op_block.get_output_tensor_info()
        output_tensor_id = output_tensor_info['tensor_id']
        group_block_id = output_tensor_info['group_block_id']
        assert output_tensor_id == op_block.NpuOp.fmo_tensor[0]
        _update_tensor_life_cycle(net, output_tensor_id, op_block)
        output_tensor = net.get_tensor(output_tensor_id)
        tensor_info = "output_tensor_id: {}, life_cycle: {}"\
                        .format(output_tensor_id, 
                                output_tensor.tensor_life_cycle_for_block_list)
        print(tensor_info)
        if op_block.block_split_mode.split_num > 1:
            imd_tensor_life_cycle = output_tensor.get_intermediate_tensor_life_cycle(group_block_id)
            tensor_info = "output_intermediate_tensor_id: {}, group_block_id: {}, life_cycle: {}"\
                            .format(output_tensor_id,
                                        group_block_id,
                                        imd_tensor_life_cycle)
            print(tensor_info)
