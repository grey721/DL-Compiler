from ir.graph.Graph_IR import *
from ir.dialect.npu.IR_operator import *
from backend.ada200.ada200 import ada200
from ir.conversion.optimization.memory.base import *
# from ir.conversion.optimization.memory.debug import *
# from ir.conversion.optimization.memory.schedule import *
# from ir.conversion.optimization.memory.allocation import *
# from ir.conversion.optimization.memory.life_cycle import *
from ir.conversion.ir_transform import _register_ir_transformation_rule


@_register_ir_transformation_rule(TransformRule.GEN_NPU_OP_GROUP)
def _gen_npu_op_group(net: GraphIR):
    print("---------OpTransformRule.GEN_NPU_OP_GROUP--------")

    backend = ada200()
    in_ops, out_ops = net.NetInputNpuOpId, net.NetOutNpuOpId
    npu_op_group_list = []
    fmo_size_record = []
    npu_op_group_id = 0
    skip_npu_op_id_list = []
    print("here", net.SubGraphs)
    for sub_graph in net.SubGraphs:
        group_op_list = []
        for npu_op_id in sub_graph:  # 子图为一个组，重复op排除
            if npu_op_id not in skip_npu_op_id_list:
                skip_npu_op_id_list.append(npu_op_id)
                npu_op = net.get_npu_op(npu_op_id)
                group_op_list.append(npu_op)
                fmo_size_record.append(npu_op.NpuOpFmoSize)

                if npu_op.NpuOpConvOp:
                    if npu_op.NpuOpConvOp.Name == "Conv_63":
                        print("here layer")
                        print(npu_op.NpuOpConvOp)

                # TODO 为什么输出特征图size大于后端最大尺寸反而跳过了，这块不懂
                if npu_op.NpuOpFmoSize > backend.fmo_max_size:
                    continue
                else:
                    max_fmo_size = max(fmo_size_record)
                    h_slice = 1
                    w_slice = 1
                    c_slice = 1

                    while max_fmo_size / h_slice > npu_op.NpuOpFmoSize:  # max_fmo_size 在h轴上切片，保证NpuOpFmoSize是最大的，为什么？
                        h_slice += 1

                    first_npu_op = group_op_list[0]
                    if first_npu_op.NpuOpConvOp.FirstLayer:  # 如果该组的第一个也是网络的第一个
                        # TODO ?为什么/3*8
                        firstLayer_fmisize = first_npu_op.NpuOpFmiSize / 3 * 8
                        if backend.first_layer_fmi_max_size < firstLayer_fmisize:
                            first_npu_op.fmi_from_global_memory = True

                        while backend.first_layer_fmi_max_size < firstLayer_fmisize / h_slice:
                            h_slice += 1

                    if npu_op.NpuOpConv:
                        kh = npu_op.NpuOpConvOp.KerH
                        kw = npu_op.NpuOpConvOp.KerW
                        assert kh == kw
                        ic = npu_op.NpuOpConvOp.InputShape[0].C

                        while ic / c_slice > backend.cluster_max_hwc / kw / kh:
                            c_slice = c_slice + 1

                    else:
                        # 最大一行支持 512 x 128 bits，channel 只能够识别16
                        ic = npu_op.InputShape[0].C
                        ih = npu_op.InputShape[0].H
                        iw = npu_op.InputShape[0].W

                        assert ic % 16 == 0
                        assert ic <= 64
                        assert ic * iw <= backend.vpu_max_buffer_size

                    w_slice_flag = False
                    while not w_slice_flag:

                        try:
                            blk_split_mode = block_split_mode(h_slice, w_slice, c_slice)
                            # npu_op_group会赋值n_k_n给OutputShape.c
                            npu_op_group_ins = npu_op_group(group_op_list,
                                                            blk_split_mode,
                                                            npu_op_group_id,
                                                            in_ops,
                                                            out_ops,
                                                            backend)
                            w_slice_flag = True

                        except Exception as err:
                            if err.args[0] in backend.get_catch_error_info_list():
                                print(err)
                                w_slice += 1
                            else:
                                print(err)
                                traceback.print_tb(err.__traceback__)
                                raise "stop"

                    npu_op_group_list.append(npu_op_group_ins)
                    print("npu_op_group_id:", npu_op_group_id)
                    print("block_split_mode:", h_slice, w_slice, c_slice)

                    npu_op_group_id += 1
                    group_op_list = []
                    fmo_size_record = []

                if npu_op.NpuOpConvOp:
                    if npu_op.NpuOpConvOp.Name == "Conv_63":
                        print("here layer")
                        print(npu_op.NpuOpConvOp)

    net.AllOps = deepcopy(npu_op_group_list)


# layer_group_pass
layer_group_transform = [# TransformRule.GEN_NPU_OP_TIME_STEP,
                         TransformRule.GEN_NPU_OP_GROUP,
                         # TransformRule.GEN_NPU_TENSOR_LIFE_CYCLE,
                         # TransformRule.UPDATE_CONCAT_TENSOR_LIFE_CYCLE,
                         # TransformRule.NPU_TENSOR_LIFE_CYCLE_REPORT,
                         # TransformRule.NPU_MEMORY_SCHEDULE,
                         # TransformRule.NPU_MEMORY_ALLOCATION,
                         # TransformRule.CHECK_BLOCK_AND_TILE_SHAPE
                         ]
