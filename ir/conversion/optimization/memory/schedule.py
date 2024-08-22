from ir.graph.Graph_IR import *
from backend.ada200.ada200 import ada200
from ir.dialect.top.IR_tensor import *
from ir.dialect.npu.IR_operator import *
from ir.dialect.npu.IR_memory import *
from ir.conversion.optimization.memory.base import *
from ir.conversion.optimization.ir_transform import  _register_ir_transformation_rule


@_register_ir_transformation_rule(TransformRule.NPU_MEMORY_SCHEDULE)
def _npu_memory_schedule(net: GraphIR):

    print("--------TransformRule.NPU_MEMORY_SCHEDULE---------")
    back_end = ada200()
    for group_op_id, group_op in enumerate(net.AllOps):

        print("---------------------------------------")
        print("group_op_id: ", group_op_id)

        for op in group_op.block_list:

            if isinstance(op, block_param):

                block_id = op.npu_op_block_id
                group_block_id = op.group_block_id
                print("block_id: ", block_id)
                print("group_block_id: ", group_block_id)
                
                # if block_id == 23 and group_block_id == [0, 0, 1]:
                #     print("debug")

                input_tensor_id = op.get_input_tensor_id()
                input_tensor = net.get_tensor(input_tensor_id)
                input_tensor_info = op.get_input_tensor_info()
                if op.input_block:
                    input_tensor_info['bank_polling_num'] = 1
                    input_tensor_info['origin_shape'][-1] = 8
                else:
                    input_tensor_info['bank_polling_num'] = 4


                output_tensor_id = op.get_output_tensor_id()
                output_tensor = net.get_tensor(output_tensor_id)
                output_tensor_info = op.get_output_tensor_info()
                output_tensor_info['bank_polling_num'] = 4
                output_tensor_info['memory_padding'] = False

                if op.short_cut_out:
                    short_cut_out_id = op.get_short_cut_out_tensor_id()
                
                if op.concat_input:
                    concat_input_tensor_id = op.get_concat_input_tensor_id()

                if op.elemwise_input:
                    elemwise_input_tensor_id = op.get_elemwise_input_tensor_id()
                
                ## input group schedule
                if group_op.npu_op_group_id == 0:

                    assert op.short_cut_out == False
                    assert op.concat_input == False
                    assert op.elemwise_input == False

                    if not op.layer_group_flag:
                        
                        if op.group_block_id == [0,0,0]:
                            tensor_size_flag = "origin"
                            assert input_tensor_info['tensor_split'] == False
                            input_tensor_shape = input_tensor.Shape.get_shape_as_np()
                            n, h, w, c = input_tensor_shape
                            input_tensor_size = h*w*8
                            input_tensor_info['tensor_size'] = input_tensor_size
                            input_tensor_info['need_allocate'] = True
                            input_tensor_info['dtype'] = "int8"
                            input_tensor.update_tensor_info(input_tensor_info, block_id)

                            output_tensor_info['tensor_size'] = op.get_output_tensor_size(tensor_size_flag)
                            output_tensor_info['need_allocate'] = True
                            output_tensor_info['dtype'] = "int8"
                            output_tensor.update_tensor_info(output_tensor_info, block_id)

                        else:
                            tensor_size_flag = "block"
                            input_tensor_info['tensor_size'] = op.get_input_tensor_size(tensor_size_flag)
                            input_tensor_info['need_allocate'] = False
                            input_tensor_info['dtype'] = "int8"
                            input_tensor.update_tensor_info(input_tensor_info, block_id)

                            output_tensor_info['tensor_size'] = op.get_output_tensor_size(tensor_size_flag)
                            output_tensor_info['need_allocate'] = False
                            output_tensor_info['dtype'] = "int8"
                            output_tensor.update_tensor_info(output_tensor_info, block_id)

                    else:
                        # op.layer_group_flag == True
                        assert len(group_op.npu_op_list) == 2

                        if op.group_block_id == [0,0,0]:
                            if op.input_block:
                                assert input_tensor_info['tensor_split'] == False
                                if op.dma_read:
                                    input_tensor_info['tensor_size'] = int(op.get_input_tensor_size("block") / 3 * 8)
                                    input_tensor_info['need_allocate'] = True
                                    input_tensor_info['dma_origin_shape'] = input_tensor_info['origin_shape']
                                    input_tensor_info['dma_block_address_list'] = deepcopy(input_tensor_info['block_address_list'])
                                    input_tensor_info['dma_tensor_size'] = int(op.get_input_tensor_size("origin") / 3 * 8)
                                    input_tensor_info['dma_read'] = True
                                    input_tensor_info['dma_need_allocate'] = True
                                    input_tensor_info['dma_life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                                    input_tensor_info['dtype'] = "int8"
                                    life_cycle_start = op.npu_op_block_id
                                    life_cycle_end =  op.npu_op_block_id
                                    input_tensor_info['life_cycle'] = [life_cycle_start, life_cycle_end]
                                    input_tensor.update_tensor_info(input_tensor_info, block_id, "dma")
                                    input_tensor_info['origin_shape'][0] = input_tensor_info['block_address_list'][1]
                                    input_tensor_info['origin_shape'][1] = input_tensor_info['block_address_list'][3]
                                    input_tensor_info['origin_shape'][2] = input_tensor_info['block_address_list'][5]
                                    input_tensor_info['block_address_list'][0] = 0
                                    input_tensor_info['block_address_list'][2] = 0
                                    input_tensor_info['block_address_list'][4] = 0
                                    input_tensor.update_tensor_info(input_tensor_info, block_id, "shared_memory")

                                else:
                                    input_tensor_shape = input_tensor.Shape.get_shape_as_np()
                                    n, h, w, c = input_tensor_shape
                                    input_tensor_size = h*w*8
                                    input_tensor_info['tensor_size'] = input_tensor_size 
                                    input_tensor_info['need_allocate'] = True
                                    input_tensor_info['dtype'] = "int8"
                                    input_tensor_info['life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                                    input_tensor.update_tensor_info(input_tensor_info, block_id)

                                output_tensor_info['tensor_size'] = op.get_output_tensor_size("block")
                                _, hl, _, wl, _, cl = output_tensor_info['block_address_list']
                                output_tensor_info['block_address_list'][0] = 0
                                output_tensor_info['block_address_list'][2] = 0
                                output_tensor_info['block_address_list'][4] = 0
                                output_tensor_info['origin_shape'] = [hl, wl, cl]
                                output_tensor_info['need_allocate'] = True
                                output_tensor_info['memory_padding'] = back_end.vpu_memory_check(output_tensor_info)
                                output_tensor_info['dtype'] = "int8"
                                output_tensor_info['life_cycle'] = [block_id, block_id+1]
                                output_tensor.update_tensor_info(output_tensor_info, block_id)

                            else:
                                input_tensor_info['tensor_size'] = op.get_input_tensor_size("block")
                                _, hl, _, wl, _, cl = input_tensor_info['block_address_list']
                                input_tensor_info['block_address_list'][0] = 0
                                input_tensor_info['block_address_list'][2] = 0
                                input_tensor_info['block_address_list'][4] = 0 
                                input_tensor_info['origin_shape'] = [hl, wl, cl]
                                input_tensor_info['need_allocate'] = False
                                input_tensor_info['dtype'] = "int8"
                                input_tensor_info['life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                                input_tensor.update_tensor_info(input_tensor_info, block_id)

                                assert back_end.vpu_memory_check(output_tensor_info) == False
                                output_tensor_info['tensor_size'] = op.get_output_tensor_size("origin")
                                output_tensor_info['need_allocate'] = True
                                output_tensor_info['dtype'] = "int8"
                                output_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                output_tensor.update_tensor_info(output_tensor_info, block_id)
                        else:
                            if op.input_block:
                                if op.dma_read:
                                    input_tensor_info['tensor_size'] = int(op.get_input_tensor_size("block") / 3 * 8)
                                    input_tensor_info['need_allocate'] = True
                                    input_tensor_info['dma_origin_shape'] = input_tensor_info['origin_shape']
                                    input_tensor_info['dma_block_address_list'] = deepcopy(input_tensor_info['block_address_list'])
                                    input_tensor_info['dma_tensor_size'] = int(op.get_input_tensor_size("block") / 3 * 8)
                                    input_tensor_info['dma_read'] = True
                                    input_tensor_info['dma_need_allocate'] = False
                                    input_tensor_info['dma_life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                                    input_tensor_info['dtype'] = "int8"
                                    life_cycle_start = op.npu_op_block_id
                                    life_cycle_end =  op.npu_op_block_id
                                    input_tensor_info['life_cycle'] = [life_cycle_start, life_cycle_end]
                                    input_tensor.update_tensor_info(input_tensor_info, block_id, "dma")
                                    input_tensor_info['origin_shape'][0] = input_tensor_info['block_address_list'][1]
                                    input_tensor_info['origin_shape'][1] = input_tensor_info['block_address_list'][3]
                                    input_tensor_info['origin_shape'][2] = input_tensor_info['block_address_list'][5]
                                    input_tensor_info['block_address_list'][0] = 0
                                    input_tensor_info['block_address_list'][2] = 0
                                    input_tensor_info['block_address_list'][4] = 0
                                    input_tensor.update_tensor_info(input_tensor_info, block_id, "shared_memory")

                                else:
                                    input_tensor_info['tensor_size'] = op.get_input_tensor_size("block") / 3 * 8
                                    input_tensor_info['origin_shape'][-1] = 8
                                    input_tensor_info['need_allocate'] = False
                                    input_tensor_info['dtype'] = "int8"
                                    input_tensor_info['life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                                    input_tensor.update_tensor_info(input_tensor_info, block_id)

                                output_tensor_info['tensor_size'] = op.get_output_tensor_size("block")
                                _, hl, _, wl, _, cl = output_tensor_info['block_address_list']
                                output_tensor_info['block_address_list'][0] = 0
                                output_tensor_info['block_address_list'][2] = 0
                                output_tensor_info['block_address_list'][4] = 0
                                output_tensor_info['origin_shape'] = [hl, wl, cl]
                                output_tensor_info['need_allocate'] = True
                                output_tensor_info['memory_padding'] = back_end.vpu_memory_check(output_tensor_info)
                                output_tensor_info['dtype'] = "int8"
                                output_tensor_info['life_cycle'] = [block_id, block_id+1]
                                output_tensor.update_tensor_info(output_tensor_info, block_id)

                            else:
                                input_tensor_info['tensor_size'] = op.get_input_tensor_size("block")
                                _, hl, _, wl, _, cl = input_tensor_info['block_address_list']
                                input_tensor_info['block_address_list'][0] = 0
                                input_tensor_info['block_address_list'][2] = 0
                                input_tensor_info['block_address_list'][4] = 0
                                input_tensor_info['origin_shape'] = [hl, wl, cl]
                                input_tensor_info['need_allocate'] = False
                                input_tensor_info['dtype'] = "int8"
                                input_tensor.update_tensor_info(input_tensor_info, block_id)

                                assert back_end.vpu_memory_check(output_tensor_info) == False
                                output_tensor_info['tensor_size'] = op.get_output_tensor_size("block")
                                output_tensor_info['need_allocate'] = False
                                output_tensor_info['dtype'] = "int8"
                                output_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                output_tensor.update_tensor_info(output_tensor_info, block_id)

                else:
                    
                    assert back_end.vpu_memory_check(output_tensor_info) == False

                    ## other group schedule
                    if not op.layer_group_flag:
                        _input_tensor_info = input_tensor.get_tensor_info(block_id)
                        inherit_tensor_id = _input_tensor_info.get('inherit_tensor_id', None) 
                        if inherit_tensor_id is not None:
                            input_tensor_info['inherit_tensor_id'] = inherit_tensor_id
                        input_tensor_info['tensor_size'] = op.get_input_tensor_size("block")
                        input_tensor_info['need_allocate'] = False
                        input_tensor_info['dtype'] = "int8"
                        input_tensor_info['life_cycle'] = input_tensor.tensor_life_cycle_for_block_list
                        input_tensor.update_tensor_info(input_tensor_info, block_id)

                        if op.elemwise_input:
                            elemwise_input_tensor = net.get_tensor(elemwise_input_tensor_id)
                            elemwise_input_tensor_info = deepcopy(elemwise_input_tensor.get_tensor_info(block_id))
                            elemwise_input_tensor_info['need_allocate'] = False
                            elemwise_input_tensor_info['dtype'] = "int8"
                            elemwise_input_tensor_info['life_cycle'] = elemwise_input_tensor.tensor_life_cycle_for_block_list
                            elemwise_input_tensor.update_tensor_info(elemwise_input_tensor_info, block_id)

                        ## the first block in group
                        if op.group_block_id == [0,0,0]:

                            if op.concat_input:
                                concat_input_tensor = net.get_tensor(concat_input_tensor_id)
                                concat_input_tensor_info = deepcopy(concat_input_tensor.get_tensor_info(block_id))
                                concat_input_tensor_info['need_allocate'] = False

                                ## update_tensor_info
                                concat_input_tensor_info['inherit_tensor_id'] = concat_input_tensor_info['tensor_id']
                                concat_input_tensor_info['tensor_id'] = output_tensor_id
                                concat_input_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                concat_in_tensor_list = op.NpuOp.concat_in_tensor_list
                                output_tensor.update_concat_output_tensor_info(concat_input_tensor_info, 
                                                                                concat_in_tensor_list,
                                                                                    concat_input_tensor_id,
                                                                                        output_tensor_info,
                                                                                            block_id)

                            else:

                                output_tensor_info = deepcopy(output_tensor_info)
                                output_tensor_info['tensor_split'] = True if op.block_split_mode.split_num > 1 else False
                                output_tensor_info['dtype'] = "int8"

                                if op.int32_out or op.npu_psum_add and op.int32_out:
                                    output_tensor_info['life_cycle'] = [block_id, block_id+1]
                                    output_tensor_info['dtype'] = "int32"
                                    output_tensor_info['tensor_size'] = op.get_output_tensor_size("block")
                                else:
                                    output_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                    output_tensor_info['dtype'] = "int8"
                                    output_tensor_info['tensor_size'] = op.get_output_tensor_size("origin")

                                output_tensor_info['need_allocate'] = True
                                output_tensor.update_tensor_info(output_tensor_info, block_id)


                            if op.short_cut_out:
                                
                                short_cut_out_tensor = net.get_tensor(short_cut_out_id)
                                short_cut_out_tensor_info = deepcopy(op.get_short_cut_out_tensor_info())
                                short_cut_out_tensor_info['bank_polling_num'] = 4
                                short_cut_out_tensor_info['need_allocate'] = True
                                short_cut_out_tensor_info['dtype'] = "int8"
                                short_cut_out_tensor_info['life_cycle'] = short_cut_out_tensor.tensor_life_cycle_for_block_list
                                short_cut_out_tensor_info = deepcopy(short_cut_out_tensor_info)
                                short_cut_out_tensor.update_tensor_info(short_cut_out_tensor_info, block_id)


                        else:

                            if op.concat_input:
                                concat_input_tensor = net.get_tensor(concat_input_tensor_id)
                                concat_input_tensor_info = deepcopy(concat_input_tensor.get_tensor_info(block_id))
                                concat_input_tensor_info['need_allocate'] = False

                                ## update_tensor_info
                                concat_input_tensor_info['inherit_tensor_id'] = concat_input_tensor_info['tensor_id']
                                concat_input_tensor_info['tensor_id'] = output_tensor_id
                                concat_input_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                concat_in_tensor_list = op.NpuOp.concat_in_tensor_list
                                output_tensor.update_concat_output_tensor_info(concat_input_tensor_info, 
                                                                                concat_in_tensor_list,
                                                                                    concat_input_tensor_id,
                                                                                        output_tensor_info,
                                                                                            block_id)


                            else:
                                output_tensor_info = deepcopy(output_tensor_info)
                                output_tensor_info['tensor_size'] = op.get_output_tensor_size("block")
                                output_tensor_info['tensor_split'] = True if op.block_split_mode.split_num > 1 else False

                                if op.int32_out or op.npu_psum_add and op.int32_out:
                                    output_tensor_info['life_cycle'] = [block_id, block_id+1]
                                    output_tensor_info['dtype'] = "int32"
                                    output_tensor_info['need_allocate'] = True
                                
                                elif op.group_block_id[0] == op.group_block_id[1] == 0 \
                                        and op.group_block_id[-1] == op.block_split_mode.c - 1:
                                    output_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                    output_tensor_info['dtype'] = "int8"
                                    output_tensor_info['need_allocate'] = True

                                else:
                                    output_tensor_info['life_cycle'] = output_tensor.tensor_life_cycle_for_block_list
                                    output_tensor_info['dtype'] = "int8"
                                    output_tensor_info['need_allocate'] = False

                                output_tensor.update_tensor_info(output_tensor_info, block_id)


                            if op.short_cut_out:
                                short_cut_out_tensor = net.get_tensor(short_cut_out_id)
                                short_cut_out_tensor_info = deepcopy(op.get_short_cut_out_tensor_info())
                                short_cut_out_tensor_info['bank_polling_num'] = 4
                                short_cut_out_tensor_info['need_allocate'] = False
                                short_cut_out_tensor_info['life_cycle'] = short_cut_out_tensor_info.tensor_life_cycle_for_block_list
                                short_cut_out_tensor_info['dtype'] = "int8"
                                short_cut_out_tensor_info = deepcopy(short_cut_out_tensor_info)
                                short_cut_out_tensor.update_tensor_info(short_cut_out_tensor_info, block_id)


                    else:
                        raise Exception(NotImplementedError)