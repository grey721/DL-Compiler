from enum import Enum
from ir.utils.formula import *

from ir.conversion.ir_transform import _find_post_op, _find_pre_op, \
    _register_ir_transformation_rule, \
    _copy_opbase_input_info, \
    _copy_opbase_output_info, \
    _order_pre_op, \
    _order_post_op
from ir.graph.Graph_IR import *


class TransformRule(Enum):
    ORDER_TOP_OPS = 1
    ORDER_NPU_OPS = 2
    CONST_FOLDING = 3
    NPU_PAD = 4
    NPU_SISO_OP = 5

    SHORTCUT_CONV_ACTIVATION_ELW = 6
    NPU_CONCAT = 7


# TOOL
class OpType(Enum):
    pre = 0
    tpu = 1
    vpu = 2


op_type_mapping = {
    OpType.pre: (NpuPad,),
    OpType.tpu: (NpuConv2d,),
    OpType.vpu: (NpuActivation, NpuElemWise, NpuResize, NpuPool, NpuTranspose, NpuReshape)
}


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x > pivot]  # 大于枢轴的元素
    middle = [x for x in arr if x == pivot]  # 等于枢轴的元素
    right = [x for x in arr if x < pivot]  # 小于枢轴的元素
    return quick_sort(left) + middle + quick_sort(right)


def _check_concat_mode(concat_ops, tensor_record):
    """ concat 的每一个输入都在 tensor_record 中 """
    if len(concat_ops) == 0:
        mode = 0
    else:
        mode_flag_list = []
        concat_in_tensors = concat_ops[0].InTensors
        for tensor_id in tensor_record:
            if tensor_id in concat_in_tensors:
                mode_flag = 1
                mode_flag_list.append(mode_flag)
        if sum(mode_flag_list) == len(concat_in_tensors):  # concat 的每一个输入都在 tensor_record 中
            mode = 1
        else:
            mode = 0
    print("     mode: ", mode)
    return mode


def _check_op_state(net, op, tensor_record):
    """ op的输入怎么以及完备 """
    flag = False
    flag_list = []
    op_in_tensors = [t for t in op.InTensors if net.AllTensors[t].Type == TensorType.Intermediate]
    if isinstance(op, NpuOp):
        op_in_tensors.extend(op.elemwise_input_tensor)
        op_in_tensors.extend(op.concat_input_tensor)
    for tensor_id in op_in_tensors:
        if tensor_id in tensor_record:
            flag_list.append(1)
    if len(flag_list) == len(op_in_tensors):
        flag = True
    return flag


def _update_tensor_record(tensor_record, op):
    if isinstance(op, NpuOp):
        tensor_record.extend(op.fmi_tensor)  # 将一个列表中的所有元素添加到另一个列表的末尾
        tensor_record.extend(op.fmo_tensor)
        tensor_record.extend(op.short_cut_out_tensor)
    else:
        tensor_record.extend(op.InTensors)
        tensor_record.extend(op.OutTensors)


# MAIN
@_register_ir_transformation_rule(TransformRule.ORDER_TOP_OPS)
def _order_top_ops(net: GraphIR):  # 排序Top
    print("----start TransformRule.ORDER_TOP_OPS---")
    for op in net.AllOps:
        pre_op_id = _find_pre_op(net, op)
        post_op_id = _find_post_op(net, op)
        op.PreTopOpId = pre_op_id
        op.PostTopOpId = post_op_id
        print(net.get_op_idx(op), pre_op_id, post_op_id)


@_register_ir_transformation_rule(TransformRule.ORDER_NPU_OPS)
def _order_npu_ops(net: GraphIR):  # 排序NPU
    print("----start TransformRule.ORDER_NPU_OPS---")
    lens = len(net.AllOps)
    for op_idx, op in enumerate(net.AllOps):
        op.NpuOpId = op_idx
        pre_op_id = _order_pre_op(net, op)
        post_op_id = _order_post_op(net, op)
        op.PreOpId = pre_op_id
        op.PostOpId = post_op_id

        # 确保AllOps列表里的NpuOp的输入特征图是上一个Op的输出特征图
        if op_idx < lens - 1:
            next_op = net.AllOps[op_idx + 1]
            if len(next_op.InTensors) > 1 and isinstance(next_op, NpuOp):
                for t in next_op.InTensors:
                    if t in next_op.fmo_tensor:
                        net.AllOps[op_idx + 1].fmi_tensor = [t]
                        break

        # 给Concat的PreOpId排序
        if isinstance(op, NpuConcat):
            n_pre_op = []
            for t in op.InTensors:
                for pre_op_idx in op.PreOpId:
                    pre_t = net.AllOps[pre_op_idx].OutTensors
                    if t in pre_t:
                        n_pre_op.append(pre_op_idx)
                        continue
            if len(n_pre_op) == len(op.PreOpId):
                op.PreOpId = n_pre_op

        print(op.NpuOpId, op.Type, pre_op_id, post_op_id)


@_register_ir_transformation_rule(TransformRule.NPU_PAD)
def _post_fuse_pad(net: GraphIR):  # 融合Pad和Conv、Pool
    print("----start TransformRule NPU_PAD-----")
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, NpuPad):
            print(f"iter:{_op_id}", op.Type)
            post_op_ids = _order_post_op(net, op)
            flag = True
            for post_op_idx in post_op_ids:
                post_op = net.get_op(post_op_idx)
                if not isinstance(post_op, (NpuConv2d, NpuPool)):
                    flag = False
            print(flag)
            if flag:
                for post_op_idx in post_op_ids:
                    post_op = net.get_op(post_op_idx)
                    if post_op.pad_top < 0 or post_op.pad_bottom < 0 or \
                            post_op.pad_left < 0 or post_op.pad_right < 0:
                        print("Pool Pad < 0")
                    post_op.pad_top += op.pad_top
                    post_op.pad_bottom += op.pad_bottom
                    post_op.pad_left += op.pad_left
                    post_op.pad_right += op.pad_right
                net.delete_op(_op_id)


@_register_ir_transformation_rule(TransformRule.NPU_SISO_OP)
def _fuse_single_output(net: GraphIR):
    print("----start TransformRule.COMMON---")

    def fuse_op(op_idx, fuse_list):
        op = net.get_op(op_idx)
        if isinstance(op, op_type_mapping[OpType.tpu]):
            if fuse_list:
                return
            else:
                fuse_list.append(op)
        elif isinstance(op, op_type_mapping[OpType.vpu]):
            if fuse_list:
                net.delete_op(op_idx)
                fuse_list.append(op)
        else:
            # print("stop:", op.Type)
            return

        post_op_idx = _order_post_op(net, op)
        if len(post_op_idx) == 1:
            fuse_op(post_op_idx[0], fuse_list)

    for op_id, npu_op in enumerate(net.AllOps):
        if isinstance(npu_op, op_type_mapping[OpType.tpu]):
            op_list = []
            fuse_op(op_id, op_list)
            if len(op_list) > 1:
                net.delete_op(op_id)

                npu_op = NpuOp()
                npu_op.fuse_ops(op_list)
                net.insert_op(npu_op, op_id)

                print(f'iter:{op_id}', [x.Type for x in op_list])
            else:
                print(f'iter:{op_id}', npu_op.Type)
        else:
            print(f'iter:{op_id}', npu_op.Type)


@_register_ir_transformation_rule(TransformRule.CONST_FOLDING)
def _delete_fuse_constant(net: GraphIR):
    print("----start TransformRule DELETE_FUSE_CONST-----")
    value_class_operator = (Floor, Cast)
    shape_class_operator = (OpShape, Unsqueeze, Transpose, Reshape, Slice)
    conv_class_operator = (NpuConv2d, NpuPool)

    # 常量传播
    def _fuse_post(graph, _current_op, result):
        post_op_ids = _current_op.PostTopOpId  # _find_post_op(graph, _current_op)
        for post_idx in post_op_ids:
            post_op = graph.get_op(post_idx)
            # 在此限制可与常数融合的算子类型
            # 单输入单输出
            if isinstance(post_op, value_class_operator) or isinstance(post_op, shape_class_operator):
                if post_idx not in result:
                    for _out_tensor_id in post_op.OutTensors:
                        net.AllTensors[_out_tensor_id].Type = TensorType.Const
                    result.append(post_idx)
                _fuse_post(graph, post_op, result)
            else:
                flag = True
                for _in_tensor_id in post_op.InTensors:
                    if net.AllTensors[_in_tensor_id].Type != TensorType.Const:
                        flag = False
                if flag:
                    if post_idx not in result:
                        result.append(post_idx)
                    _fuse_post(graph, post_op, result)


    delete_list = []
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, (Constant, OpShape)):
            out_tensor_idx = op.OutTensors[0]
            if net.AllTensors[out_tensor_idx].Type != TensorType.Parameter:
                _fuse_post(net, op, delete_list)
            if _op_id not in delete_list:
                delete_list.append(_op_id)

    delete_list = quick_sort(delete_list)

    for idx in delete_list:
        print('Delete:', net.AllOps[idx].Name)
        net.delete_op(idx)

    # 常量折叠
    def handle(v, b, mode, _idx):
        if b == 0:
            b = Variable(f"B_{_idx}")
        if mode == -1:
            v += b
        elif mode == -2:
            v -= b
        elif mode == -3:
            v *= b
        elif mode == -5:
            v **= b
        return v

    record = []
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, ElemWise) and op.TopOpId not in record and op.Mode < 0 and len(op.PostTopOpId) == 1:
            print(op.Name, _op_id)
            current_op = op
            temp = [_op_id]
            x = Variable("X")
            x = handle(x, op.B, op.Mode, _op_id)
            while True:
                post_idx = _find_post_op(net, current_op)[0]

                post_op = net.get_op(post_idx)
                if isinstance(post_op, ElemWise) and post_op.Mode < 0 and len(post_op.PostTopOpId) == 1:
                    print(post_op.Name, post_idx)
                    x = handle(x, post_op.B, post_op.Mode, post_idx)
                    temp.append(post_idx)
                    record.append(post_op.TopOpId)
                else:
                    print("========", x.params)
                    break
                current_op = post_op


@_register_ir_transformation_rule(TransformRule.SHORTCUT_CONV_ACTIVATION_ELW)
def _post_fuse_conv_activation_elw(net: GraphIR):
    print("----start TransformRule NPU_CONV_ACTIVATION_ELW-----")
    i = 0
    tensor_record = []
    for op in net.AllOps:
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Type)
        i += 1
        if isinstance(op, NpuConv2d):
            flag = False
            post_conv_ids = _find_post_op(net, op)
            acti_ops = []
            elw_ops = []
            elw_op_ids = []
            post_elw_ops = []
            post_elw_ids = []
            post_acti_op_ids = []
            post_acti_elw_num = 0

            if len(post_conv_ids) == 2:
                for post_conv_op_id in post_conv_ids:
                    post_conv_op = net.get_op(post_conv_op_id)
                    if isinstance(post_conv_op, Activation):
                        _update_tensor_record(tensor_record, post_conv_op)
                        acti_ops.append(post_conv_op)

                    elif isinstance(post_conv_op, ElemWise):
                        _update_tensor_record(tensor_record, post_conv_op)
                        elw_op_ids.append(post_conv_op_id)
                        elw_ops.append(post_conv_op)

                        post_elw_op_ids = _find_post_op(net, post_conv_op)
                        for post_elw_op_idx in post_elw_op_ids:
                            post_elw_op = net.get_op(post_elw_op_idx)
                            post_elw_ops.append(post_elw_op)
                            post_elw_ids.append(post_elw_op_idx)

                if len(acti_ops) == 1 and len(elw_ops) == 1:  # 激活后面的op是这两个
                    post_acti_op_ids = _find_post_op(net, acti_ops[0])
                    if len(post_acti_op_ids) == 1 and post_acti_op_ids[0] in post_conv_ids:
                        post_acti_elw_num += 1
                        flag = True

            print("     Post Conv:", post_conv_ids)
            print("     post Anti:", post_acti_op_ids)

            print("     flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                npu_op = NpuOp()
                npu_op.fuse_ops([op, acti_ops[0], elw_ops[0]])
                npu_op.Type = "CSM"
                assert (op_id + 1) == post_conv_ids[0]
                assert (op_id + 2) == elw_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete acti
                net.delete_op(op_id)  # delete elw

                net.insert_op(npu_op, op_id)
                # print("op_id:", op.TopOpId)


@_register_ir_transformation_rule(TransformRule.NPU_CONCAT)
def _fuse_concat(net: GraphIR):
    print("----start TransformRule NPU_CONCAT-----")
    # 模型一般已经满足该顺序
    # # 排序，确保concat的输入均已出现
    # tensor_record = []
    # n_ops = [net.AllOps[0]]
    # op_record = [0]
    # temp_ids = []
    # i = -1
    # # TODO 仅单输入网络可用，且第一个必须是网络输入op
    # while True:
    #     i += 1
    #     current_op = n_ops[-1]
    #     _update_tensor_record(tensor_record, current_op)
    #     post_op_ids = _order_post_op(net, current_op)
    #     print(f"iter:{i}", n_ops[-1].Name)
    #
    #     if len(post_op_ids) > 1:
    #         for _id in post_op_ids[1:]:
    #             if _id not in temp_ids and _id not in op_record:
    #                 temp_ids.append(_id)
    #         temp_ids = quick_sort(temp_ids)
    #     elif len(post_op_ids) == 0 and len(temp_ids) == 0:
    #         # name_list = [x.Name for x in n_ops]
    #         # for op in net.AllOps:
    #         #     if op.Name not in name_list:
    #         #         print(op.Name)
    #         assert len(n_ops) == len(net.AllOps), f"n_ops:{len(n_ops)}, AllOps:{len(net.AllOps)}"
    #         break
    #
    #     post_op_id = post_op_ids[0]
    #     post_op = net.get_op(post_op_id)
    #     print("      Post Ops:", post_op_ids, post_op.Name)
    #     print("      Temp Ops:", temp_ids)
    #
    #     if isinstance(post_op, (NpuElemWise, NpuConcat, NpuOp)):
    #         temp_ids = [x for x in temp_ids if x not in op_record]
    #         if _check_op_state(net, post_op, tensor_record):
    #             print("      Post Op Ready")
    #             n_ops.append(post_op)
    #         elif temp_ids:
    #             print("      Load Temp Op")
    #             post_op_id = temp_ids.pop(-1)
    #             post_op = net.get_op(post_op_id)
    #             n_ops.append(post_op)
    #     else:
    #         n_ops.append(post_op)
    #     op_record.append(post_op_id)
    # net.AllOps = n_ops

    # 具体处理Concat
    for op in net.AllOps:
        if isinstance(op, NpuConcat):
            idx = net.get_op_idx(op)
            # 将concat融入前一个NpuOp
            pre_op = net.AllOps[idx - 1]
            if isinstance(pre_op, NpuOp):
                pre_op.fuse_ops(op)

                net.delete_op(idx)
            else:
                npu_op = NpuOp()
                npu_op.fuse_ops([pre_op, op])

                net.delete_op(idx-1)
                net.delete_op(idx-1)
                net.insert_op(npu_op, idx-1)
    print("Net Lens:", len(net.AllOps))


op_fuse_transform = [
    TransformRule.ORDER_TOP_OPS,
    TransformRule.CONST_FOLDING,
    TransformRule.NPU_PAD,
    TransformRule.NPU_SISO_OP,
    TransformRule.SHORTCUT_CONV_ACTIVATION_ELW,
    # TransformRule.NPU_CONCAT,
    TransformRule.ORDER_NPU_OPS
]
