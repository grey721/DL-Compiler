from ir.graph.Graph_IR import *
from ir.dialect.top.IR_tensor import *
from ir.dialect.npu.IR_operator import *
from ir.conversion.optimization.ir_transform import _register_ir_transformation_rule, \
    _find_pre_npu_op, \
    _find_post_npu_op
# from backend.ada200.ada200 import ada200
from enum import Enum
import copy


class TransformRule(Enum):
    NOPE = 1
    SUBNET = 2  # 区分NPU OP 和 CPU OP到不同的列表
    ORDER_NPU_OPS = 3  # 排序，找前后
    REORDER_NPU_OPS = 4  # 获得子图，从输入到输出的任意一条路径？
    UPDATE_CONCAT_OPS = 5  # 为带有串联的OP和其前一个Op更新串联信息


@_register_ir_transformation_rule(TransformRule.SUBNET)
def _subnet(net: GraphIR):  # 区分NPU OP 和CPU OP
    print("----TransformRule.SUBNET----")

    npu_op_list = []
    cpu_op_list = []
    if len(net.NetOutTensors) == 1:  # 单输出网络
        break_op_id = None
        for op_id, op in enumerate(net.AllOps):
            if break_op_id is None:
                if isinstance(op, NpuOp):
                    npu_op_list.append(op)
                else:
                    # TODO 为什么直接全都算CPU的？
                    break_op_id = op_id
                    cpu_op_list.append(op)
            else:
                if op_id > break_op_id:
                    cpu_op_list.append(op)

        net.AllOps = npu_op_list
        net.AllCpuOps = cpu_op_list

        for op in net.AllOps:  # 网络的输入和输出Op
            if isinstance(op, NpuOp):
                pre_op_id = _find_pre_npu_op(net, op)
                post_op_id = _find_post_npu_op(net, op)
                if len(pre_op_id) == 0:
                    net.NetInputNpuOpId.append(op.NpuOpId)
                if len(post_op_id) == 0:
                    net.NetOutNpuOpId.append(op.NpuOpId)
    else:  # 多输出
        for op in net.AllOps:
            if isinstance(op, NpuOp):  # 网络的输入和输出Op
                pre_op_id = _find_pre_npu_op(net, op)
                post_op_id = _find_post_npu_op(net, op)
                if len(pre_op_id) == 0:
                    net.NetInputNpuOpId.append(op.NpuOpId)
                if len(post_op_id) == 0:
                    net.NetOutNpuOpId.append(op.NpuOpId)

        for op_id, op in enumerate(net.AllOps):

            if isinstance(op, NpuOp):
                npu_op_list.append(op)
            else:
                cpu_op_list.append(op)

        net.AllOps = npu_op_list
        net.AllCpuOps = cpu_op_list


@_register_ir_transformation_rule(TransformRule.REORDER_NPU_OPS)
def _reorder_npu_ops(net: GraphIR):
    print("----TransformRule.REORDER_NPU_OPS----")

    # for op in net.AllOps:
    #     if isinstance(op, NpuOp):
    #         pre_op_id = _find_pre_npu_op(net, op)
    #         post_op_id = _find_post_npu_op(net, op)
    #         if len(pre_op_id) == 0:
    #             net.NetInputNpuOpId.append(op.NpuOpId)
    #         if len(post_op_id) == 0:
    #             net.NetOutNpuOpId.append(op.NpuOpId)

    def dfs(sub_graph, op_id):  # 深度优先搜索
        sub_graph.append(op_id)
        net_op = net.get_npu_op(op_id)
        op_id_list = _find_pre_npu_op(net, net_op)
        if op_id_list:
            # TODO 可行吗？
            next_id = max(op_id_list)
            return dfs(sub_graph, next_id)
        # if len(op_id_list) == 1:
        #     op_id = op_id_list[0]
        #     print(op_id)
        #     return dfs(sub_graph, op_id)
        #
        # elif len(op_id_list) == 2:
        #     op_id = max(op_id_list)
        #     print(op_id)
        #     return dfs(sub_graph, op_id)
        #
        # elif len(op_id_list) > 2:
        #     raise NotImplementedError
        #
        # else:
        #     pass

    if len(net.NetOutNpuOpId) > 1:
        sub_graph_list = []
        for NpuOpId in net.NetOutNpuOpId:
            sub_graph = []
            dfs(sub_graph, NpuOpId)
            sub_graph_list.append(sub_graph)

        # sorted() 函数用于对可迭代对象进行排序，并返回一个新的列表，key是排序的依据，默认升序。
        # 排序首先基于元组的第一个元素（即len(x)），即子图中元素的数量。如果两个子图的元素数量相同，则比较元组的第二个元素（即子图对象本身x）。
        sorted_sub_graph_list = sorted(sub_graph_list, key=lambda x: (len(x), x))
        # sub_graph_list中的Op顺序是倒序，main_graph是倒序
        main_graph = sorted_sub_graph_list[-1]  # main_graph = 最长的子图

        net.SubGraphs.append(main_graph[::-1])
        if len(sorted_sub_graph_list) > 0:
            for sub_graph in sorted_sub_graph_list[:-1]:  # 除了最后一个元素（最长子图）
                op_record = []
                for op_id in sub_graph:
                    if op_id in main_graph:
                        break
                    else:
                        op_record.append(op_id)
                # sub_graph中的是倒序，反转调正
                net.SubGraphs.append(op_record[::-1])
                # 子图中所有的op
                op_record.extend(main_graph)
                # 当使用等号 = 赋值时，main_graph 将只是 op_record 的一个引用，这意味着对 main_graph 的任何修改都会反映在 op_record 上，
                # 反之亦然。使用 copy.deepcopy 可以创建 op_record 的一个完全独立的副本，这样对 main_graph 的修改不会影响到原始的 op_record。
                main_graph = copy.deepcopy(op_record)

        npu_sorted_id_list = []
        npu_op_list = []
        main_graph.reverse()
        for op_id in main_graph:
            npu_op = net.get_npu_op(op_id)
            npu_sorted_id_list.append(net.get_op_idx(npu_op))
            npu_op_list.append(npu_op)

        net.AllOps = npu_op_list
    else:
        op_record = []
        for op in net.AllOps:
            op_record.append(op.NpuOpId)
        net.SubGraphs.append(op_record)


@_register_ir_transformation_rule(TransformRule.ORDER_NPU_OPS)
def _order_npu_ops(net: GraphIR):
    print("----TransformRule.ORDER_NPU_OPS----")
    for op in net.AllOps:
        pre_op_id = _find_pre_npu_op(net, op)
        post_op_id = _find_post_npu_op(net, op)
        op.PreOpId = pre_op_id
        op.PostOpId = post_op_id
        print(op.NpuOpId, pre_op_id, post_op_id)


@_register_ir_transformation_rule(TransformRule.UPDATE_CONCAT_OPS)
def _update_concat_ops(net: GraphIR):
    print("----TransformRule.UPDATE_CONCAT_OPS----")
    for op in net.AllOps:
        outputs = deepcopy(op.fmo_tensor)
        outputs.extend(op.short_cut_out_tensor)
        if len(op.PostOpId) > 0:  # 还有下一个op
            for post_op_id in op.PostOpId:
                post_op = net.get_npu_op(post_op_id)
                if len(post_op.concat_input_tensor) > 0:  # 如果下一个NPU OP中有串联
                    for output in outputs:
                        if output == post_op.concat_input_tensor[0]:  # 如果当前Op的输出是下一个串联的输入
                            assert len(post_op.fmo_tensor) == 1
                            post_op_fmo_tensor = net.get_tensor(post_op.fmo_tensor[0])
                            concat_output_shape = post_op_fmo_tensor.Shape.get_shape_as_np()
                            concat_in_tensor_list = post_op.NpuOpConcatOp.InTensors
                            op.add_output_tensor_for_cancat(output)  # 将当前张量标记为下一个Op串联需要，为串联准备
                            op.concat_in_tensor_list = concat_in_tensor_list  # 记录下一个串联的输入列表
                            op.concat_output = True
                            op.concat_output_shape = concat_output_shape
                            post_op.concat_output_shape = concat_output_shape
                            post_op.concat_in_tensor_list = concat_in_tensor_list


subnet_transform = [TransformRule.SUBNET,
                    TransformRule.REORDER_NPU_OPS,
                    TransformRule.ORDER_NPU_OPS,
                    TransformRule.UPDATE_CONCAT_OPS
                    ]
