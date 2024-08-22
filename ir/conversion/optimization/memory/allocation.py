from ir.graph.Graph_IR import *
from ir.dialect.top.IR_tensor import *
from ir.dialect.npu.IR_operator import *
from ir.dialect.npu.IR_memory import *
from ir.conversion.optimization.memory.base import *
from ir.conversion.optimization.ir_transform import _register_ir_transformation_rule
from backend.ada200.ada200 import ada200
import traceback


@_register_ir_transformation_rule(TransformRule.NPU_MEMORY_ALLOCATION)
def _npu_memory_allocation(net: GraphIR):

    backend = ada200()

    with Capturing() as npu_memory_allocation_loger:

        print("--------TransformRule.NPU_MEMORY_ALLOCATION---------")

        _net = net
        status = False
        restart_block_id = None
        op_shm_record = {}
        glm_op_record = {}
        shm_op_record = {}
        net_record = {}
        shm_psum_read_param_dict_record = {}

        while not status:
                
            try:
                glm = GlobalMemory()
                shm = SharedMemory()

                if restart_block_id is not None:
                    net = net_record[restart_block_id]
                    shm = shm_op_record[restart_block_id]
                    glm = glm_op_record[restart_block_id]

                for group_op_id, group_op in enumerate(net.AllOps):

                    print("---------------------------------------")
                    print("group_op_id: ", group_op_id)

                    record = {}
                    shm_psum_read_param_dict = dict(bank_addr_list=None, tensor_info=None, dtype=None)

                    for op in group_op.block_list:
                                                    
                        if isinstance(op, block_param):

                            block_id = op.npu_op_block_id
                            print("block_id: ", block_id)

                            if restart_block_id:
                                if block_id < restart_block_id:
                                    continue
                            

                            shm_op_record[block_id] = deepcopy(shm)
                            glm_op_record[block_id] = deepcopy(glm)
                            net_record[block_id] = deepcopy(net)


                            record['group_op_id'] = group_op_id
                            # if group_op_id == 0:
                            #     print("debug")
                            record['block_id'] = block_id

                            ## input layer default setting 
                            if group_op_id == 0 \
                                and op.input_block == False \
                                    and op.layer_group_flag == True:
                                
                                used_bank_group_id = 0
                                shm.set_bank_group_id_used(used_bank_group_id)

                            
                            if op.group_block_id[-1] > 0:

                                if op.int32_out or op.npu_psum_add:
                                    if restart_block_id is not None:
                                        if block_id == group_op.block_list[0].npu_op_block_id:
                                            shm_psum_read_param_dict = shm_psum_read_param_dict_record[block_id]
                                        else:
                                            shm_psum_read_param_dict = shm_psum_read_param_dict_record[block_id-1]

                                    op.shm_psum_read_param_dict = deepcopy(shm_psum_read_param_dict)
                                    used_bank_group_id = op.shm_psum_read_param_dict['tensor_info']['bank_group_id']
                                    shm.set_bank_group_id_used(used_bank_group_id)
                                    shm_psum_read_param_dict = dict(bank_addr_list=None, tensor_info=None, dtype=None)
                                    print("shm_psum_read_param_dict: ", op.shm_psum_read_param_dict)
                                    record['psum_read'] = used_bank_group_id
                            

                            print("input memory allocate")
                            input_tensor_id = op.get_input_tensor_id()
                            input_tensor = net.get_tensor(input_tensor_id)
                            input_tensor_info = input_tensor.get_tensor_info(block_id)
                            inherit_tensor_id = input_tensor_info.get('inherit_tensor_id', None) 
                            if inherit_tensor_id is not None:
                                inherit_tensor = net.get_tensor(inherit_tensor_id)
                                inherit_tensor_info = inherit_tensor.get_tensor_info(block_id)
                                input_tensor_info['bank_group_id'] = inherit_tensor_info['bank_group_id']
                                input_tensor_info['addr'] = inherit_tensor_info['addr']
                                input_tensor_info['len'] = inherit_tensor_info['len']

                            if input_tensor_info['need_allocate']:
                                bank_group_id = shm.get_available_bank_group_id_list()[0]
                                shm.allocate2(input_tensor, input_tensor_info, block_id, bank_group_id)
                            else:
                                used_bank_group_id = input_tensor_info['bank_group_id']
                                shm.set_bank_group_id_used(used_bank_group_id)
                            op.shm_read_param_dict['tensor_info'] = input_tensor_info
                            print("shm_read_param_dict: ", op.shm_read_param_dict)
                            record['read'] = bank_group_id
                            record['avail'] = deepcopy(shm.get_available_bank_group_id_list())

                            if input_tensor_info.get("dma_read", None):
                                if input_tensor_info['dma_need_allocate']:
                                    glm.allocate2(input_tensor, input_tensor_info, block_id)
                                op.dma_read_param_dict['tensor_info'] = input_tensor_info
                                print("dma_read_param_dict: ", op.dma_read_param_dict)

                            if op.elemwise_input:
                                print("elemwise_input memory allocate")
                                elemwise_input_tensor_id = op.get_elemwise_input_tensor_id()
                                elemwise_input_tensor = net.get_tensor(elemwise_input_tensor_id)
                                elemwise_input_tensor_info = elemwise_input_tensor.get_tensor_info(block_id)
                                used_bank_group_id = elemwise_input_tensor_info['bank_group_id']
                                shm.set_bank_group_id_used(used_bank_group_id)
                                op.shm_elemwise_read_param_dict['tensor_info'] = elemwise_input_tensor_info
                                print("elemwise_input: ", op.elemwise_input)
                                print("shm_elemwise_read_param_dict: ", op.shm_elemwise_read_param_dict)
                                record['elem_read'] = elemwise_input_tensor_info['bank_group_id']
                                record['avail'] = deepcopy(shm.get_available_bank_group_id_list())


                            if op.concat_input:
                                print("concat_input memory allocate")
                                concat_input_tensor_id = op.get_concat_input_tensor_id()
                                concat_input_tensor = net.get_tensor(concat_input_tensor_id)
                                concat_input_tensor_info = concat_input_tensor.get_tensor_info(block_id)
                                used_bank_group_id = concat_input_tensor_info['bank_group_id']
                                shm.set_bank_group_id_used(used_bank_group_id)
                                output_tensor_id = op.get_output_tensor_id()
                                output_tensor = net.get_tensor(output_tensor_id)
                                output_tensor_info = output_tensor.get_tensor_info(block_id)
                                output_tensor_info['bank_group_id'] = concat_input_tensor_info['bank_group_id']
                                output_tensor_info["addr"] = concat_input_tensor_info["addr"]
                                output_tensor_info["len"] = concat_input_tensor_info["len"]
                                op.shm_write_param_dict['tensor_info'] = output_tensor_info
                                print("concat_input: ", op.concat_input)
                                print("shm_write_param_dict: ", op.shm_write_param_dict)
                                record['write'] = concat_input_tensor_info['bank_group_id']
                        
                            else:
                                print("output memory allocate")
                                output_tensor_id = op.get_output_tensor_id()
                                output_tensor = net.get_tensor(output_tensor_id)
                                output_tensor_info = output_tensor.get_tensor_info(block_id)

                                if output_tensor_info['need_allocate']:
                                    if restart_block_id is not None:
                                        if block_id == restart_block_id:
                                            last_time_used_bank_group_id = op_shm_record[block_id]["write"]
                                            op_shm_record[block_id]["avail"].remove(last_time_used_bank_group_id)
                                            if len(op_shm_record[block_id]["avail"]) > 0: 
                                                used_bank_group_id = op_shm_record[block_id]["avail"][0]
                                                shm.allocate2(output_tensor, output_tensor_info, block_id, used_bank_group_id)
                                            else:
                                                errer_info = "share memory bank group id is not avail"
                                                raise(Exception(errer_info))
                                        else:
                                            shm.allocate2(output_tensor, output_tensor_info, block_id)
                                    else:
                                        shm.allocate2(output_tensor, output_tensor_info, block_id)
                                else:
                                    used_bank_group_id = output_tensor_info['bank_group_id']
                                    shm.set_bank_group_id_used(used_bank_group_id)


                                op.shm_write_param_dict['tensor_info'] = output_tensor_info

                                if op.int32_out:
                                    shm_psum_read_param_dict['tensor_info'] = output_tensor_info
                                    shm_psum_read_param_dict_record[block_id] = shm_psum_read_param_dict

                                print("shm_write_param_dict: ", op.shm_write_param_dict)
                                record['write'] = output_tensor_info['bank_group_id']


                            if op.short_cut_out:
                                print("short_cut_out memory allocate")
                                short_cut_out_id = op.get_short_cut_out_tensor_id()
                                short_cut_out_tensor = net.get_tensor(short_cut_out_id)
                                short_cut_out_tensor_info = short_cut_out_tensor.get_tensor_info(block_id)

                                if short_cut_out_tensor_info['need_allocate']:
                                    shm.allocate2(short_cut_out_tensor, short_cut_out_tensor_info, block_id)

                                _short_cut_out_tensor_info = deepcopy(short_cut_out_tensor_info)
                                op.short_cut_shm_write_param_dict['tensor_info'] = _short_cut_out_tensor_info
                                print("short_cut_shm_write_param_dict: ", op.short_cut_shm_write_param_dict)
                                record['cut_write'] = short_cut_out_tensor_info['bank_group_id']

                            shm.reset()

                            print("---------------------------------------")


                        if restart_block_id:
                            if block_id >= restart_block_id:
                                op_shm_record[block_id] = deepcopy(record)
                        else:
                            op_shm_record[block_id] = deepcopy(record)     
                        

                status = True


            except Exception as err:

                if err.args[0] in backend.get_catch_error_info_list():
                    print(err)
                    restart_block_id = block_id - 1
                    status = False
                else:
                    print(err)
                    traceback.print_tb(err.__traceback__)
                    raise("stop")
                
        _net.SharedMemory = shm
        _net.update(net)