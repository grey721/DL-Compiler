from ir.graph.Graph_IR import *
from ir.dialect.npu.IR_operator import *
from ir.conversion.ir_pass.ir_transform import _find_post_op, _find_pre_op, \
    _register_ir_transformation_rule, \
    _copy_opbase_input_info, \
    _copy_opbase_output_info, \
    _find_pre_npu_op, \
    _find_post_npu_op
from enum import Enum


class TransformRule(Enum):
    NOPE = 1  #

    # reoder ops
    ORDER_TOP_OPS = 2  # 排序
    ORDER_NPU_OPS = 5  # 排序
    # fuse ops
    FUSE_CONV_BN = 7  #
    FUSE_CONV_RELU = 3  #
    PADDING_FOR_CONV_AND_POOLING = 4  #
    # remove reshape
    REMOVE_RESHAPE_INPUT = 10  #
    REMOVE_RESHAPE_REALTED = 11  #
    REMOVE_DROPOUT = 12  #
    REMOVE_IDENTITY = 13  #
    REMOVE_PADDING = 14  #

    DEPTHWISE_CONV_RESHAPE_WEIGHT = 106  #

    # vpu post layer fuse all set
    NPU_CONV = 107   # conv + [[conv, None, reshape, transpose], [concat, elw] + [conv, None, reshape, transpose]]
    NPU_FC = 108    # FullConnect + [Conv / FullConnect / NPU / Pad], [Reshape / Transpose / Softmax]

    # A41
    NPU_CONV_ACTIVATION = 250
    NPU_CONV_RESIZE = 251
    NPU_CONV_POOL = 252
    NPU_CONV_ELW = 253

    # A42
    NPU_CONV_ACTIVATION_ELW = 201
    NPU_CONV_ACTIVATION_RESIZE = 202
    NPU_CONV_ACTIVATION_POOL = 203
    NPU_CONV_POOL_ELW = 204
    NPU_CONV_POOL_ACTIVATION = 205
    NPU_CONV_RESIZE_ELW = 206
    NPU_CONV_RESIZE_ACTIVATION = 207
    NPU_CONV_ELW_RESIZE = 208
    NPU_CONV_ELW_POOL = 209
    NPU_CONV_ELW_ACTIVATION = 200

    # A43
    NPU_CONV_ACTIVATION_ELW_POOL = 221
    NPU_CONV_ACTIVATION_ELW_RESIZE = 222
    NPU_CONV_ACTIVATION_POOL_ELW = 223
    NPU_CONV_ACTIVATION_RESIZE_ELW = 224
    NPU_CONV_ELW_ACTIVATION_POOL = 225
    NPU_CONV_ELW_ACTIVATION_RESIZE = 226
    NPU_CONV_ELW_POOL_ACTIVATION = 227
    NPU_CONV_ELW_RESIZE_ACTIVATION = 228
    NPU_CONV_POOL_ELW_ACTIVATION = 229
    NPU_CONV_POOL_ACTIVATION_ELW = 230
    NPU_CONV_RESIZE_ACTIVATION_ELW = 231
    NPU_CONV_RESIZE_ELW_ACTIVATION = 232

    NPU_PAD = 300
    NPU_PAD_CONV = 301  # 融合Pad和Conv
    NPU_PAD_POOL = 302  # TODO 融合Pad Pool Concat?

    NPU_RESHAPE_FC = 303  # 将单输入且输出为1*1的Reshape与相连的FullConnect融合
    NPU_RESHAPE_CONV = 304  # 将单输入且输出为1*1的Reshape与相连的Conv融合


def _check_post_op_only_conv_or_out(net, op):
    """post op is all: Conv2d / FullConnected / NpuOp / NpuPad"""
    flag = True
    post_op_ids = _find_post_op(net, op)
    for post_op_ids_idx in post_op_ids:
        post_op = net.get_op(post_op_ids_idx)
        if not (isinstance(post_op, NpuConv2d)
                or isinstance(post_op, NpuFullConnected)
                or isinstance(post_op, NpuOp)
                or isinstance(post_op, NpuPad)):
            flag = False
    return flag


def _check_post_op_out(net, op):
    """Next ops is all:  NpuReshape, NpuTranspose"""
    flag = True
    post_op_ids = _find_post_op(net, op)
    for post_op_ids_idx in post_op_ids:
        post_op = net.get_op(post_op_ids_idx)
        if not (isinstance(post_op, NpuReshape) or isinstance(post_op, NpuTranspose)):
            flag = False
    return flag


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
    print("mode: ", mode)
    return mode


def _check_post_op_only_cpu_op(net, op):
    """only one post op and its device is cpu: Reshape / Transpose / Softmax"""
    flag = False
    post_op_ids = _find_post_op(net, op)
    if len(post_op_ids) == 1:
        post_op = net.get_op(post_op_ids[0])
        if (isinstance(post_op, NpuReshape)
                or isinstance(post_op, NpuTranspose)
                or isinstance(post_op, NpuSoftmax)):
            flag = True
    return flag


def _update_tensor_record(tensor_record, op):
    if isinstance(op, NpuOp):
        tensor_record.extend(op.fmi_tensor)  # 将一个列表中的所有元素添加到另一个列表的末尾
        tensor_record.extend(op.fmo_tensor)
        # TODO what
        tensor_record.extend(op.short_cut_out_tensor)
    else:
        tensor_record.extend(op.InTensors)
        tensor_record.extend(op.OutTensors)


# run first
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
    for op_idx, op in enumerate(net.AllOps):
        if isinstance(op, NpuOp):
            pre_op_id = _find_pre_npu_op(net, op)
            post_op_id = _find_post_npu_op(net, op)
            op.PreTopOpId = pre_op_id
            op.PostTopOpId = post_op_id
            print(op.NpuOpId, pre_op_id, post_op_id)


@_register_ir_transformation_rule(TransformRule.NPU_RESHAPE_CONV)
def _post_fuse_reshape_conv(net: GraphIR):  # reshape与下一个卷积
    print("-----TransformRule NPU_RESHAPE_CONV-----")
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, NpuReshape):
            if op.OutputShape[0].H == op.OutputShape[0].W == 1:  # 1 x 1 卷积？
                conv_op_ids = []
                conv_ops = []
                post_op_ids = _find_post_op(net, op)
                op_id = net.get_op_idx(op)

                for post_op_ids_idx in post_op_ids:
                    post_op = net.get_op(post_op_ids_idx)
                    if isinstance(post_op, NpuConv2d):
                        conv_ops.append(post_op)
                        conv_op_ids.append(post_op_ids_idx)

                if len(conv_ops) == 1:
                    flag = True
                else:
                    flag = False

                print(flag)
                if flag:
                    conv_op = conv_ops[0]
                    # _copy_opbase_input_info(fc_op, op)
                    replace_tensor_index = conv_op.InTensors.index(op.OutTensors[0])
                    assert len(op.InTensors) == 1  # Reshape的输入只有1个
                    conv_op.InTensors[replace_tensor_index] = op.InTensors[0]  # 卷积的输入换成Reshape的输入
                    net.delete_op(op_id)  # 删除Reshape


# USE "is" LOGIC
@_register_ir_transformation_rule(TransformRule.NPU_PAD_CONV)
def _post_fuse_conv_single(net: GraphIR):  # 融合Pad和Conv
    print("-----TransformRule NPU_PAD_CONV-----")
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, NpuPad):
            conv_op_ids = []
            conv_ops = []
            post_op_ids = _find_post_op(net, op)
            op_id = net.get_op_idx(op)

            for post_op_ids_idx in post_op_ids:
                post_op = net.get_op(post_op_ids_idx)
                if isinstance(post_op, NpuConv2d):
                    conv_ops.append(post_op)
                    conv_op_ids.append(post_op_ids_idx)

            if len(conv_ops) == 1:
                flag = True
            else:
                flag = False

            print(flag)
            if flag:
                conv_op = conv_ops[0]
                conv_op.InputH = op.InputShape[0].H
                conv_op.InputW = op.InputShape[0].W

                conv_op.pad_top += op.pad_top
                conv_op.pad_bottom += op.pad_bottom
                conv_op.pad_left += op.pad_left
                conv_op.pad_right += op.pad_right

                if op.TopOpId == 0:
                    conv_op.FirstLayer = True

                _copy_opbase_input_info(conv_op, op)
                replace_tensor_index = conv_op.InTensors.index(op.OutTensors[0])
                assert len(op.InTensors) == 1
                conv_op.InTensors[replace_tensor_index] = op.InTensors[0]
                net.delete_op(op_id)


@_register_ir_transformation_rule(TransformRule.NPU_PAD_POOL)
def _post_fuse_pool_single(net: GraphIR):
    print("------TransformRule NPU_PAD_POOL-----")
    i = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuPad):
            flag = False
            mode = -1
            pool_op_ids = _find_post_op(net, op)
            pool_ops = []
            concat_op_ids = []
            concat_ops = []
            pad_op_ids = []
            pad_ops = []
            other_ops = []
            if len(pool_op_ids) == 1:
                pool_op = net.get_op(pool_op_ids[0])
                if isinstance(pool_op, NpuPool):
                    _update_tensor_record(tensor_record, pool_op)
                    pool_ops.append(pool_op)
                    post_pool_op_id = _find_post_op(net, pool_op)
                    print(len(post_pool_op_id))
                    for post_pool_op_id_idx in post_pool_op_id:
                        post_pool_op = net.get_op(post_pool_op_id_idx)  # Pool的下一个
                        if isinstance(post_pool_op, NpuConv2d) or isinstance(post_pool_op, NpuOp):  # 下一个是卷积或者是NPU
                            pad_ops.append(post_pool_op)
                            pad_op_ids.append(post_pool_op_id_idx)
                            continue
                        elif isinstance(post_pool_op, NpuConcat):
                            concat_flag = True
                            post_concat_op_ids = _find_post_op(net, post_pool_op)
                            for post_concat_op_ids_idx in post_concat_op_ids:
                                post_concat_op = net.get_op(post_concat_op_ids_idx)
                                if not isinstance(post_concat_op, NpuConv2d):  # 下一个是Concat，并且串联的下一个,每个都是卷积
                                    concat_flag = False
                            if concat_flag:
                                concat_ops.append(post_pool_op)
                                concat_op_ids.append(post_pool_op_id_idx)
                            continue
                        else:  # Pool的下一个，非预期之内
                            other_ops.append(post_pool_op)

                    print(len(concat_ops), len(pad_ops), len(other_ops))
                    if ((len(concat_ops) > 0) or (len(pad_ops) > 0)) and (len(other_ops) == 0):
                        flag = True
            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)  # concat的张量是否完全由已经记录的张量组成

                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0:
                    npu_op.OutTensors = pool_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, pool_ops[0])
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                pad_bottom = pool_ops[0].pad_bottom
                pad_left = pool_ops[0].pad_left
                pad_right = pool_ops[0].pad_right
                pad_top = pool_ops[0].pad_top
                # TODO ？？还会小于零？
                if pad_bottom < 0:
                    op.pad_bottom += pad_bottom
                    assert op.pad_bottom >= 0
                    pool_ops[0].pad_bottom = 0

                if pad_left < 0:
                    op.pad_left += pad_left
                    assert op.pad_left >= 0
                    pool_ops[0].pad_left = 0

                if pad_right < 0:
                    op.pad_right += pad_right
                    assert op.pad_right >= 0
                    pool_ops[0].pad_right = 0

                if pad_top < 0:
                    op.pad_top += pad_top
                    assert op.pad_top >= 0
                    pool_ops[0].pad_top = 0

                npu_op.NpuOpMode = VpuPostOpSetMode.POOL
                npu_op.NpuOpConv = False
                npu_op.NpuOpPad = True
                npu_op.NpuOpPadOp = op
                npu_op.NpuOpPool = True
                npu_op.NpuOpPoolOp = pool_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpPadOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpPoolOp)

                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                op_id = net.get_op_idx(op)
                assert (op_id + 1) == pool_op_ids[0]
                net.delete_op(op_id)
                net.delete_op(op_id)
                if mode == 1:
                    assert (op_id + 2) == concat_op_ids[0]
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


# run first
# conv + [[conv, None, reshape, transpose], [concat, elw] + [conv, None, reshape, transpose]]
@_register_ir_transformation_rule(TransformRule.NPU_CONV)
def _post_fuse_concat_single(net: GraphIR):
    print("------TransformRule NPU_CONV-----")
    i = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuConv2d):  # conv
            flag = False
            mode = -1

            if _check_post_op_out(net, op):  #
                flag = True
            concat_op_ids = []
            concat_ops = []
            post_op_ids = _find_post_op(net, op)  # conv下一个
            for post_op_ids_idx in post_op_ids:
                post_op = net.get_op(post_op_ids_idx)
                if isinstance(post_op, NpuConcat):  # 下一个op中是Concat的
                    if _check_post_op_only_conv_or_out(net, post_op):  # Concat下一个仍然都是卷积类的
                        flag = True
                        concat_ops.append(post_op)
                        concat_op_ids.append(post_op_ids_idx)

            for post_op_ids_idx in post_op_ids:
                post_op = net.get_op(post_op_ids_idx)
                if not isinstance(post_op, NpuConcat):  # 下一个op中不全是Concat
                    flag = False

            if _check_post_op_only_conv_or_out(net, op):  # Conv + Conv
                flag = True

            if _check_post_op_only_cpu_op(net, op):  # Conv + Reshape \ Transpose \ Softmax
                flag = True

            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.NpuOpConvOp = op
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)

                if mode == 0:
                    npu_op.OutTensors = op.OutTensors
                    _copy_opbase_output_info(npu_op, op)
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.NONE
                if mode == 1:  # Concat信息已全
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                op_id = net.get_op_idx(op)
                net.delete_op(op_id)

                if mode == 1:  # Concat信息已全，Concat也融合进去
                    assert op_id + 1 == concat_op_ids[0]
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ELW)
def _post_fuse_elw_single(net: GraphIR):
    print("------TransformRule NPU_CONV_ELW-----")
    i = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuConv2d):  # 第一个卷积
            flag = False
            mode = -1
            op_id = net.get_op_idx(op)
            elw_op_ids = _find_post_op(net, op)
            elw_ops = []
            concat_op_ids = []
            concat_ops = []
            conv_op_ids = []
            conv_ops = []
            other_ops = []
            if len(elw_op_ids) == 1:
                elw_op = net.get_op(elw_op_ids[0])
                if isinstance(elw_op, NpuElemWise):  # Conv + ElemWise
                    _update_tensor_record(tensor_record, elw_op)
                    elw_ops.append(elw_op)
                    post_elw_op_id = _find_post_op(net, elw_op)
                    print(len(post_elw_op_id))
                    for post_elw_op_id_idx in post_elw_op_id:
                        post_elw_op = net.get_op(post_elw_op_id_idx)
                        # Conv + ElemWise + Conv \ Npu \ ElemWise \ Mean
                        if isinstance(post_elw_op, NpuConv2d)   \
                                or isinstance(post_elw_op, NpuOp) \
                                or isinstance(post_elw_op, NpuElemWise) \
                                or isinstance(post_elw_op, NpuMean):
                            conv_ops.append(post_elw_op)
                            conv_op_ids.append(post_elw_op_id_idx)
                        # Conv + ElemWise + Concat
                        elif isinstance(post_elw_op, NpuConcat):
                            concat_flag = True
                            post_concat_op_ids = _find_post_op(net, post_elw_op)
                            for post_concat_op_ids_idx in post_concat_op_ids:
                                post_concat_op = net.get_op(post_concat_op_ids_idx)
                                if not isinstance(post_concat_op, NpuConv2d):
                                    concat_flag = False
                            if concat_flag:
                                concat_ops.append(post_elw_op)
                                concat_op_ids.append(post_elw_op_id_idx)
                        else:
                            other_ops.append(post_elw_op)

                    print(len(concat_ops), len(conv_ops), len(other_ops))
                    if ((len(concat_ops) > 0) or (len(conv_ops) > 0)) \
                            and (len(other_ops) == 0) \
                            and (op_id + 1) == elw_op_ids[0]:
                        flag = True
            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0:
                    npu_op.OutTensors = elw_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, elw_ops[0])
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.ELW
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpElemWise = True
                npu_op.NpuOpElemWiseOp = elw_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpElemWiseOp)

                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                assert (op_id + 1) == elw_op_ids[0]
                net.delete_op(op_id)
                net.delete_op(op_id)
                if mode == 1:
                    assert (op_id + 2) == concat_op_ids[0]
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


# conv + act + [conv, concat + conv]
@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION)
def _post_fuse_activation_single(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION-----")
    i = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            acti_op_ids = _find_post_op(net, op)
            acti_ops = []
            concat_op_ids = []
            concat_ops = []
            conv_op_ids = []
            conv_ops = []
            other_ops = []
            if len(acti_op_ids) == 1:
                acti_op = net.get_op(acti_op_ids[0])
                if isinstance(acti_op, NpuActivation):
                    _update_tensor_record(tensor_record, acti_op)
                    acti_ops.append(acti_op)
                    post_acti_op_id = _find_post_op(net, acti_op)
                    print(len(post_acti_op_id))
                    for post_acti_op_id_idx in post_acti_op_id:
                        post_acti_op = net.get_op(post_acti_op_id_idx)
                        if isinstance(post_acti_op, NpuConv2d) or isinstance(post_acti_op, NpuOp):
                            conv_ops.append(post_acti_op)
                            conv_op_ids.append(post_acti_op_id_idx)
                            continue
                        else:
                            if isinstance(post_acti_op, NpuConcat):
                                concat_flag = True
                                post_concat_op_ids = _find_post_op(net, post_acti_op)
                                for post_concat_op_ids_idx in post_concat_op_ids:
                                    post_concat_op = net.get_op(post_concat_op_ids_idx)
                                    if not isinstance(post_concat_op, NpuConv2d):
                                        concat_flag = False
                                if concat_flag:
                                    concat_ops.append(post_acti_op)
                                    concat_op_ids.append(post_acti_op_id_idx)
                                continue
                            else:
                                other_ops.append(post_acti_op)

                    print(len(concat_ops), len(conv_ops), len(other_ops))
                    if ((len(concat_ops) > 0) or (len(conv_ops) > 0)) and (len(other_ops) == 0):
                        flag = True
            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0:
                    npu_op.OutTensors = acti_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, acti_ops[0])
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpActivate = True
                npu_op.NpuOpActivateOp = acti_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpActivateOp)

                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                op_id = net.get_op_idx(op)
                assert (op_id + 1) == acti_op_ids[0]
                net.delete_op(op_id)
                net.delete_op(op_id)
                if mode == 1:
                    assert (op_id + 2) == concat_op_ids[0]
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


# conv + acti + ['concat + conv, elw + conv,' pool + 'concat' + conv, pool + conv]
@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_POOL)
def _post_fuse_activation_pool(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_POOL-----")
    iter = 0
    tensor_record = []
    for op in net.AllOps:
        _update_tensor_record(tensor_record, op)
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            acti_op_ids = _find_post_op(net, op)
            acti_ops = []
            pool_ops = []
            pool_op_ids = []
            post_pool_ops = []
            post_pool_op_ids = []
            post_acti_pool_num = 0
            post_acti_concat_num = 0
            post_acti_elw_num = 0
            post_acti_conv_num = 0
            post_acti_other_num = 0
            post_pool_conv_num = 0
            post_pool_concat_num = 0
            post_pool_other_num = 0

            if len(acti_op_ids) == 1:
                acti_op = net.get_op(acti_op_ids[0])
                if isinstance(acti_op, NpuActivation):
                    _update_tensor_record(tensor_record, acti_op)
                    acti_ops.append(acti_op)
                    post_acti_op_id = _find_post_op(net, acti_op)
                    print(len(post_acti_op_id))
                    for post_acti_op_id_idx in post_acti_op_id:
                        post_acti_op = net.get_op(post_acti_op_id_idx)
                        _update_tensor_record(tensor_record, post_acti_op)
                        pool_ops.append(post_acti_op)
                        pool_op_ids.append(post_acti_op_id_idx)
                        post_pool_op_id = _find_post_op(net, post_acti_op)
                        for post_pool_op_id_idx in post_pool_op_id:
                            post_pool_op = net.get_op(post_pool_op_id_idx)
                            post_pool_ops.append(post_pool_op)
                            post_pool_op_ids.append(post_pool_op_id_idx)

                    for op_post_acti in pool_ops:
                        if isinstance(op_post_acti, NpuPool):
                            post_acti_pool_num += 1
                        else:
                            if isinstance(op_post_acti, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_acti):
                                    post_acti_concat_num += 1
                            else:
                                if isinstance(op_post_acti, NpuElemWise):
                                    if _check_post_op_only_conv_or_out(net, op_post_acti):
                                        post_acti_elw_num += 1

                                elif isinstance(op_post_acti, NpuConv2d) \
                                        or isinstance(op_post_acti, NpuOp):
                                    post_acti_conv_num += 1

                                else:
                                    post_acti_other_num += 1
                    for op_post_pool in post_pool_ops:
                        if isinstance(op_post_pool, NpuConv2d) or isinstance(op_post_pool, NpuOp):
                            post_pool_conv_num += 1
                        else:
                            if isinstance(op_post_pool, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_pool):
                                    post_pool_concat_num += 1
                            else:
                                post_pool_other_num += 1

                    print("post_acti_pool_num: ", post_acti_pool_num)
                    print("post_acti_concat_num: ", post_acti_concat_num)
                    print("post_acti_elw_num: ", post_acti_elw_num)
                    print("post_acti_conv_num: ", post_acti_conv_num)
                    print("post_acti_other_num: ", post_acti_other_num)
                    print("post_pool_conv_num: ", post_pool_conv_num)
                    print("post_pool_concat_num: ", post_pool_concat_num)
                    print("post_pool_other_num: ", post_pool_other_num)

                    if post_acti_other_num == 0 and post_pool_other_num == 0 and \
                            post_acti_pool_num > 0:
                        flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_acti_concat_num + post_acti_elw_num + post_pool_concat_num == 0:
                    mode = 0
                else:
                    if (post_acti_concat_num + post_acti_elw_num) > 0 and post_pool_concat_num == 0:
                        mode = 1
                    else:
                        if (post_acti_concat_num + post_acti_elw_num) == 0 and post_pool_concat_num > 0:
                            mode = 2
                        else:
                            if (post_acti_concat_num + post_acti_elw_num) > 0 and post_pool_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = pool_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, pool_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_pool_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_pool_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_POOL
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpActivate = True
                npu_op.NpuOpActivateOp = acti_ops[0]
                npu_op.NpuOpPool = True
                npu_op.NpuOpPoolOp = pool_ops[0]
                post_pool_concat_op = None
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpActivateOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpPoolOp)
                if mode == 1 or mode == 3:
                    npu_op.NpuOpActiOut = True
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_pool_concat_op in post_pool_ops:
                        if isinstance(post_pool_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_pool_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                assert (op_id + 1) == acti_op_ids[0]
                assert (op_id + 2) == pool_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete acti
                net.delete_op(op_id)  # delete pool
                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_pool_concat_op)
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_ELW)
def _post_fuse_activation_pool(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_ELW-----")
    iter = 0
    tensor_record = []
    for op in net.AllOps:
        _update_tensor_record(tensor_record, op)
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            acti_op_ids = _find_post_op(net, op)
            acti_ops = []
            elw_ops = []
            elw_op_ids = []
            post_elw_ops = []
            post_elw_op_ids = []
            post_acti_elw_num = 0
            post_acti_concat_num = 0
            post_acti_pool_num = 0
            post_acti_conv_num = 0
            post_acti_other_num = 0
            post_elw_conv_num = 0
            post_elw_concat_num = 0
            post_elw_other_num = 0

            if len(acti_op_ids) == 1:
                acti_op = net.get_op(acti_op_ids[0])
                if isinstance(acti_op, NpuActivation):
                    _update_tensor_record(tensor_record, acti_op)
                    acti_ops.append(acti_op)
                    post_acti_op_id = _find_post_op(net, acti_op)
                    print(len(post_acti_op_id))
                    for post_acti_op_id_idx in post_acti_op_id:
                        post_acti_op = net.get_op(post_acti_op_id_idx)
                        _update_tensor_record(tensor_record, post_acti_op)
                        elw_ops.append(post_acti_op)
                        elw_op_ids.append(post_acti_op_id_idx)
                        post_elw_op_id = _find_post_op(net, post_acti_op)
                        for post_elw_op_id_idx in post_elw_op_id:
                            post_elw_op = net.get_op(post_elw_op_id_idx)
                            post_elw_ops.append(post_elw_op)
                            post_elw_op_ids.append(post_elw_op_id_idx)

                    for op_post_acti in elw_ops:
                        if isinstance(op_post_acti, NpuElemWise):
                            post_acti_elw_num += 1
                        else:
                            if isinstance(op_post_acti, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_acti):
                                    post_acti_concat_num += 1
                            else:
                                if isinstance(op_post_acti, NpuPool):
                                    if _check_post_op_only_conv_or_out(net, op_post_acti):
                                        post_acti_pool_num += 1

                                elif isinstance(op_post_acti, NpuConv2d) \
                                        or isinstance(op_post_acti, NpuOp):
                                    post_acti_conv_num += 1

                                else:
                                    post_acti_other_num += 1
                    for op_post_elw in post_elw_ops:
                        if isinstance(op_post_elw, NpuConv2d) or isinstance(op_post_elw, NpuOp):
                            post_elw_conv_num += 1
                        else:
                            if isinstance(op_post_elw, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_elw):
                                    post_elw_concat_num += 1
                            else:
                                post_elw_other_num += 1

                    print("post_acti_elw_num: ", post_acti_elw_num)
                    print("post_acti_concat_num: ", post_acti_concat_num)
                    print("post_acti_pool_num: ", post_acti_pool_num)
                    print("post_acti_conv_num: ", post_acti_conv_num)
                    print("post_acti_other_num: ", post_acti_other_num)
                    print("post_elw_conv_num: ", post_elw_conv_num)
                    print("post_elw_concat_num: ", post_elw_concat_num)
                    print("post_elw_other_num: ", post_elw_other_num)

                    if post_acti_other_num == 0 and post_elw_other_num == 0 and \
                            post_acti_elw_num > 0:
                        flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_acti_concat_num + post_acti_pool_num + post_elw_concat_num == 0:
                    mode = 0
                else:
                    if (post_acti_concat_num + post_acti_pool_num) > 0 and post_elw_concat_num == 0:
                        mode = 1
                    else:
                        if (post_acti_concat_num + post_acti_pool_num) == 0 and post_elw_concat_num > 0:
                            mode = 2
                        else:
                            if (post_acti_concat_num + post_acti_pool_num) > 0 and post_elw_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = elw_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, elw_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_elw_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_elw_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_ELW
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpActivate = True
                npu_op.NpuOpActivateOp = acti_ops[0]
                npu_op.NpuOpElemWise = True
                npu_op.NpuOpElemWiseOp = elw_ops[0]
                post_elw_concat_op = None
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpActivateOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpElemWiseOp)
                if mode == 1 or mode == 3:
                    npu_op.NpuOpActiOut = True
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_elw_concat_op in post_elw_ops:
                        if isinstance(post_elw_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_elw_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                assert (op_id + 1) == acti_op_ids[0]
                assert (op_id + 2) == elw_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete acti
                net.delete_op(op_id)  # delete elw
                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_elw_concat_op)
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


# conv + acti + ['concat + conv, elw + conv,' pool + 'concat' + conv, pool + conv]
@_register_ir_transformation_rule(TransformRule.NPU_CONV_POOL)
def _post_fuse_pool_single(net: GraphIR):
    print("-----TransformRule NPU_CONV_POOL-----")
    iter = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            pool_op_ids = _find_post_op(net, op)
            pool_ops = []
            concat_op_ids = []
            concat_ops = []
            conv_op_ids = []
            conv_ops = []
            other_ops = []
            if len(pool_op_ids) <= 2:
                for pool_id in pool_op_ids:
                    pool_op = net.get_op(pool_id)
                    if isinstance(pool_op, NpuPool):
                        _update_tensor_record(tensor_record, pool_op)
                        pool_ops.append(pool_op)
                        post_pool_op_id = _find_post_op(net, pool_op)
                        print(len(post_pool_op_id))
                        for post_pool_op_id_idx in post_pool_op_id:
                            post_pool_op = net.get_op(post_pool_op_id_idx)
                            if isinstance(post_pool_op, NpuConv2d) or isinstance(post_pool_op, NpuOp):
                                conv_ops.append(post_pool_op)
                                conv_op_ids.append(post_pool_op_id_idx)
                                continue
                            else:
                                if isinstance(post_pool_op, NpuConcat):
                                    concat_flag = True
                                    post_concat_op_ids = _find_post_op(net, post_pool_op)
                                    for post_concat_op_ids_idx in post_concat_op_ids:
                                        post_concat_op = net.get_op(post_concat_op_ids_idx)
                                        if not isinstance(post_concat_op, NpuConv2d):
                                            concat_flag = False
                                    if concat_flag:
                                        concat_ops.append(post_pool_op)
                                        concat_op_ids.append(post_pool_op_id_idx)
                                    continue
                                else:
                                    other_ops.append(post_pool_op)

                        print(len(concat_ops), len(conv_ops), len(other_ops))
                        if ((len(concat_ops) > 0) or (len(conv_ops) > 0)) and (len(other_ops) == 0):
                            flag = True
            else:
                raise NotImplementedError

            print(flag)

            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0:
                    npu_op.OutTensors = pool_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, pool_ops[0])

                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.POOL
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpPool = True
                npu_op.NpuOpPoolOp = pool_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpPoolOp)

                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                op_id = net.get_op_idx(op)
                assert (op_id + 1) == pool_op_ids[0]
                net.delete_op(op_id)
                net.delete_op(op_id)
                if mode == 1:
                    assert (op_id + 2) == concat_op_ids[0]
                    net.delete_op(op_id)
                npu_op.init_all()
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_POOL_ACTIVATION)
def _post_fuse_pool_activation(net: GraphIR):
    print("-----TransformRule NPU_CONV_POOL_ACTIVATION-----")
    iter = 0
    for op in net.AllOps:
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            pool_op_ids = _find_post_op(net, op)
            pool_ops = []
            acti_ops = []
            acti_op_ids = []
            post_acti_ops = []
            post_acti_op_ids = []
            post_pool_acti_num = 0
            post_pool_concat_num = 0
            post_pool_elw_num = 0
            post_pool_conv_num = 0
            post_pool_other_num = 0
            post_acti_conv_num = 0
            post_acti_concat_num = 0
            post_acti_other_num = 0

            if len(pool_op_ids) == 1:
                pool_op = net.get_op(pool_op_ids[0])
                if isinstance(pool_op, NpuPool):
                    pool_ops.append(pool_op)
                    post_pool_op_id = _find_post_op(net, pool_op)
                    print(len(post_pool_op_id))
                    for post_pool_op_id_idx in post_pool_op_id:
                        post_pool_op = net.get_op(post_pool_op_id_idx)
                        acti_ops.append(post_pool_op)
                        acti_op_ids.append(post_pool_op_id_idx)
                        post_acti_op_id = _find_post_op(net, post_pool_op)
                        for post_acti_op_id_idx in post_acti_op_id:
                            post_acti_op = net.get_op(post_acti_op_id_idx)
                            post_acti_ops.append(post_acti_op)
                            post_acti_op_ids.append(post_acti_op_id_idx)

                    for op_post_pool in acti_ops:
                        if isinstance(op_post_pool, NpuActivation):
                            post_pool_acti_num += 1
                        else:
                            if isinstance(op_post_pool, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_pool):
                                    post_pool_concat_num += 1
                            else:
                                if isinstance(op_post_pool, NpuElemWise):
                                    if _check_post_op_only_conv_or_out(net, op_post_pool):
                                        post_pool_elw_num += 1

                                elif isinstance(op_post_pool, NpuOp) \
                                        or isinstance(op_post_pool, NpuConv2d):
                                    post_pool_conv_num += 1

                                else:
                                    post_pool_other_num += 1

                    print(len(post_acti_ops))
                    for op_post_acti in post_acti_ops:
                        if isinstance(op_post_acti, NpuConv2d) or isinstance(op_post_acti, NpuOp):
                            post_acti_conv_num += 1
                        else:
                            if isinstance(op_post_acti, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_acti):
                                    post_acti_concat_num += 1
                            else:
                                post_acti_other_num += 1
                    print("post_pool_acti_num: ", post_pool_acti_num)
                    print("post_pool_concat_num: ", post_pool_concat_num)
                    print("post_pool_elw_num: ", post_pool_elw_num)
                    print("post_pool_conv_num: ", post_pool_conv_num)
                    print("post_pool_other_num: ", post_pool_other_num)
                    print("post_acti_conv_num: ", post_acti_conv_num)
                    print("post_acti_concat_num: ", post_acti_concat_num)
                    print("post_acti_other_num: ", post_acti_other_num)

                    if post_pool_other_num == 0 and post_acti_other_num == 0 and \
                            post_pool_acti_num > 0:
                        flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_pool_concat_num + post_pool_elw_num + post_acti_concat_num == 0:
                    mode = 0
                else:
                    if (post_pool_concat_num + post_pool_elw_num) > 0 and post_acti_concat_num == 0:
                        mode = 1
                    else:
                        if (post_pool_concat_num + post_pool_elw_num) == 0 and post_acti_concat_num > 0:
                            mode = 2
                        else:
                            if (post_pool_concat_num + post_pool_elw_num) > 0 and post_acti_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = acti_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, acti_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_acti_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_acti_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.POOL_ACTIVATION
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpPool = True
                npu_op.NpuOpPoolOp = pool_ops[0]
                npu_op.NpuOpActivate = True
                npu_op.NpuOpActivateOp = acti_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpPoolOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpActivateOp)

                post_acti_concat_op = None
                if mode == 1 or mode == 3:
                    npu_op.NpuOpActiOut = True
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_acti_concat_op in post_acti_ops:
                        if isinstance(post_acti_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_acti_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                assert (op_id + 1) == pool_op_ids[0]
                assert (op_id + 2) == acti_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete pool
                net.delete_op(op_id)  # delete acti
                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_acti_concat_op)
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_POOL_ELW)
def _post_fuse_pool_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_POOL_ELW-----")
    iter = 0
    for op in net.AllOps:
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            pool_op_ids = _find_post_op(net, op)
            pool_ops = []
            elw_ops = []
            elw_op_ids = []
            post_elw_ops = []
            post_elw_op_ids = []
            post_pool_acti_num = 0
            post_pool_concat_num = 0
            post_pool_elw_num = 0
            post_pool_other_num = 0
            post_elw_conv_num = 0
            post_elw_concat_num = 0
            post_elw_other_num = 0

            if len(pool_op_ids) == 1:
                pool_op = net.get_op(pool_op_ids[0])
                if isinstance(pool_op, NpuPool):
                    pool_ops.append(pool_op)
                    post_pool_op_id = _find_post_op(net, pool_op)
                    print(len(post_pool_op_id))
                    for post_pool_op_id_idx in post_pool_op_id:
                        post_pool_op = net.get_op(post_pool_op_id_idx)
                        elw_ops.append(post_pool_op)
                        elw_op_ids.append(post_pool_op_id_idx)
                        post_elw_op_id = _find_post_op(net, post_pool_op)
                        for post_elw_op_id_idx in post_elw_op_id:
                            post_elw_op = net.get_op(post_elw_op_id_idx)
                            post_elw_ops.append(post_elw_op)
                            post_elw_op_ids.append(post_elw_op_id_idx)

                    for op_post_pool in elw_ops:
                        if isinstance(op_post_pool, NpuElemWise):
                            post_pool_elw_num += 1
                        else:
                            if isinstance(op_post_pool, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_pool):
                                    post_pool_concat_num += 1
                            else:
                                if isinstance(op_post_pool, NpuActivation):
                                    if _check_post_op_only_conv_or_out(net, op_post_pool):
                                        post_pool_acti_num += 1
                                else:
                                    post_pool_other_num += 1
                    print(len(post_elw_ops))
                    for op_post_elw in post_elw_ops:
                        if isinstance(op_post_elw, NpuConv2d) or isinstance(op_post_elw, NpuOp):
                            post_elw_conv_num += 1
                        else:
                            if isinstance(op_post_elw, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_elw):
                                    post_elw_concat_num += 1
                            else:
                                post_elw_other_num += 1
                    print("post_pool_elw_num: ", post_pool_elw_num)
                    print("post_pool_concat_num: ", post_pool_concat_num)
                    print("post_pool_acti_num: ", post_pool_acti_num)
                    print("post_pool_other_num: ", post_pool_other_num)
                    print("post_elw_conv_num: ", post_elw_conv_num)
                    print("post_elw_concat_num: ", post_elw_concat_num)
                    print("post_elw_other_num: ", post_elw_other_num)

                    if post_pool_other_num == 0 and post_elw_other_num == 0 and \
                            post_pool_elw_num > 0:
                        flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_pool_concat_num + post_pool_acti_num + post_elw_concat_num == 0:
                    mode = 0
                else:
                    if (post_pool_concat_num + post_pool_acti_num) > 0 and post_elw_concat_num == 0:
                        mode = 1
                    else:
                        if (post_pool_concat_num + post_pool_acti_num) == 0 and post_elw_concat_num > 0:
                            mode = 2
                        else:
                            if (post_pool_concat_num + post_pool_acti_num) > 0 and post_elw_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = elw_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, elw_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_elw_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_elw_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.POOL_ELW
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpPool = True
                npu_op.NpuOpPoolOp = pool_ops[0]
                npu_op.NpuOpElemWise = True
                npu_op.NpuOpElemWiseOp = elw_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpPoolOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpElemWiseOp)

                post_elw_concat_op = None
                if mode == 1 or mode == 3:
                    npu_op.NpuOpActiOut = True
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_elw_concat_op in post_elw_ops:
                        if isinstance(post_elw_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_elw_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                assert (op_id + 1) == pool_op_ids[0]
                assert (op_id + 2) == elw_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete pool
                net.delete_op(op_id)  # delete elw
                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_elw_concat_op)
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


def _post_fuse_three_op(net: GraphIR, op_pattern):
    assert len(op_pattern) == 3
    op0 = op_pattern[0]
    op1 = op_pattern[1]
    op2 = op_pattern[2]
    i = 0
    for op in net.AllOps:
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            op0_op_ids = _find_post_op(net, op)
            op0_ops = []
            op1_ops = []
            op2_ops = []
            op2_op_ids = []
            post_op2_ops = []
            post_op2_op_ids = []
            post_op1_op2_num = 0
            post_op1_concat_num = 0
            post_op1_elw_num = 0
            post_op1_other_num = 0
            post_op2_conv_num = 0
            post_op2_concat_num = 0
            post_op2_other_num = 0

            if len(op0_op_ids) == 1:
                op0_op = net.get_op(op0_op_ids[0])
                if isinstance(op0_op, op0):
                    op0_ops.append(op0_op)
                    post_op0_op_id = _find_post_op(net, op0_op)
                    print(len(post_op0_op_id))

                    if len(post_op0_op_id) == 1:
                        op1_op = net.get_op(post_op0_op_id[0])
                        if isinstance(op1_op, op1):
                            op1_ops.append(op1_op)
                            post_op1_op_id = _find_post_op(net, op1_op)
                            print(len(post_op1_op_id))
                            for post_op1_op_id_idx in post_op1_op_id:
                                post_op1_op = net.get_op(post_op1_op_id_idx)
                                op2_ops.append(post_op1_op)
                                op2_op_ids.append(post_op1_op_id_idx)
                                post_op2_id = _find_post_op(net, post_op1_op)
                                for post_op2_id_idx in post_op2_id:
                                    post_op2_op = net.get_op(post_op2_id_idx)
                                    post_op2_ops.append(post_op2_op)
                                    post_op2_op_ids.append(post_op2_id_idx)

                    elif len(post_op0_op_id) == 2:
                        op1_op = net.get_op(post_op0_op_id[0])
                        op1_op_1 = net.get_op(post_op0_op_id[1])

                        if isinstance(op1_op, op1) \
                                and (isinstance(op1_op_1, NpuConv2d)
                                     or isinstance(op1_op_1, NpuOp)):
                            op1_ops.append(op1_op)
                            post_op1_op_id = _find_post_op(net, op1_op)
                            print(len(post_op1_op_id))
                            for post_op1_op_id_idx in post_op1_op_id:
                                post_op1_op = net.get_op(post_op1_op_id_idx)
                                op2_ops.append(post_op1_op)
                                op2_op_ids.append(post_op1_op_id_idx)
                                post_op2_id = _find_post_op(net, post_op1_op)
                                for post_op2_id_idx in post_op2_id:
                                    post_op2_op = net.get_op(post_op2_id_idx)
                                    post_op2_ops.append(post_op2_op)
                                    post_op2_op_ids.append(post_op2_id_idx)

            for post_op1_op in op2_ops:
                if isinstance(post_op1_op, op2):
                    post_op1_op2_num += 1
                else:
                    if isinstance(post_op1_op, NpuConcat):
                        if _check_post_op_only_conv_or_out(net, post_op1_op):
                            post_op1_concat_num += 1
                    else:
                        if isinstance(post_op1_op, NpuElemWise):
                            if _check_post_op_only_conv_or_out(net, post_op1_op):
                                post_op1_elw_num += 1
                        else:
                            post_op1_other_num += 1
            print(len(post_op2_ops))
            for op_post_op2 in post_op2_ops:
                if isinstance(op_post_op2, NpuConv2d) or isinstance(op_post_op2, NpuOp):
                    post_op2_conv_num += 1
                else:
                    if isinstance(op_post_op2, NpuConcat):
                        if _check_post_op_only_conv_or_out(net, op_post_op2):
                            post_op2_concat_num += 1
                    else:
                        post_op2_other_num += 1
            print("post_op1_op2_num: ", post_op1_op2_num)
            print("post_op1_concat_num: ", post_op1_concat_num)
            print("post_op1_elw_num: ", post_op1_elw_num)
            print("post_op1_other_num: ", post_op1_other_num)
            print("post_op2_conv_num: ", post_op2_conv_num)
            print("post_op2_concat_num: ", post_op2_concat_num)
            print("post_op2_other_num: ", post_op2_other_num)

            if post_op1_other_num == 0 and post_op2_other_num == 0 and \
                    post_op1_op2_num > 0:
                flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_op1_concat_num + post_op1_elw_num + post_op2_concat_num == 0:
                    mode = 0
                else:
                    if (post_op1_concat_num + post_op1_elw_num) > 0 and post_op2_concat_num == 0:
                        mode = 1
                    else:
                        if (post_op1_concat_num + post_op1_elw_num) == 0 and post_op2_concat_num > 0:
                            mode = 2
                        else:
                            if (post_op1_concat_num + post_op1_elw_num) > 0 and post_op2_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = op2_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, op2_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_op2_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_op2_ops[0])

                if op0 == NpuPool \
                        and op1 == NpuActivation \
                        and op2 == NpuElemWise:

                    npu_op.NpuOpMode = VpuPostOpSetMode.POOL_ACTIVATION_ELW
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op0_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op1_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op2_ops[0]

                elif op0 == NpuResize \
                        and op1 == NpuActivation \
                        and op2 == NpuElemWise:

                    npu_op.NpuOpMode = VpuPostOpSetMode.RESIZE_ACTIVATION_ELW
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op0_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op1_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op2_ops[0]

                elif op0 == NpuPool \
                        and op1 == NpuElemWise \
                        and op2 == NpuActivation:

                    npu_op.NpuOpMode = VpuPostOpSetMode.POOL_ELW_ACTIVATION
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op0_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op1_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op2_ops[0]

                elif op0 == NpuResize \
                        and op1 == NpuElemWise \
                        and op2 == NpuActivation:

                    npu_op.NpuOpMode = VpuPostOpSetMode.POOL_ELW_ACTIVATION
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op0_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op1_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op2_ops[0]

                elif op0 == NpuActivation \
                        and op1 == NpuElemWise \
                        and op2 == NpuPool:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_ELW_POOL
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op0_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op1_ops[0]
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op2_ops[0]

                elif op0 == NpuActivation \
                        and op1 == NpuElemWise \
                        and op2 == NpuResize:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_ELW_RESIZE
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op0_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op1_ops[0]
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op2_ops[0]

                elif op0 == NpuActivation \
                        and op1 == NpuPool \
                        and op2 == NpuElemWise:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_POOL_ELW
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op0_ops[0]
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op1_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op2_ops[0]

                elif op0 == NpuActivation \
                        and op1 == NpuResize \
                        and op2 == NpuElemWise:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_RESIZE_ELW
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op0_ops[0]
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op1_ops[0]
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op2_ops[0]

                elif op0 == NpuElemWise \
                        and op1 == NpuActivation \
                        and op2 == NpuPool:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ELW_ACTIVATION_POOL
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op0_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op1_ops[0]
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op2_ops[0]

                elif op0 == NpuElemWise \
                        and op1 == NpuActivation \
                        and op2 == NpuResize:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ELW_ACTIVATION_RESIZE
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op0_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op1_ops[0]
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op2_ops[0]

                elif op0 == NpuElemWise \
                        and op1 == NpuPool \
                        and op2 == NpuActivation:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ELW_POOL_ACTIVATION
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op0_ops[0]
                    npu_op.NpuOpPool = True
                    npu_op.NpuOpPoolOp = op1_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op2_ops[0]

                elif op0 == NpuElemWise \
                        and op1 == NpuResize \
                        and op2 == NpuActivation:

                    npu_op.NpuOpMode = VpuPostOpSetMode.ELW_RESIZE_ACTIVATION
                    npu_op.NpuOpConvOp = op
                    npu_op.NpuOpElemWise = True
                    npu_op.NpuOpElemWiseOp = op0_ops[0]
                    npu_op.NpuOpResize = True
                    npu_op.NpuOpResizeOp = op1_ops[0]
                    npu_op.NpuOpActivate = True
                    npu_op.NpuOpActivateOp = op2_ops[0]

                npu_op.NpuOpFlow.append(op)
                npu_op.NpuOpFlow.append(op0_ops[0])
                npu_op.NpuOpFlow.append(op1_ops[0])
                npu_op.NpuOpFlow.append(op2_ops[0])

                post_op2_concat_op = None
                if mode == 1 or mode == 3:
                    # npu_op.NpuOpResizeOut = True
                    pass
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_op2_concat_op in post_op2_ops:
                        if isinstance(post_op2_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_op2_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                assert (op_id + 1) == op0_op_ids[0]
                assert (op_id + 2) == post_op0_op_id[0]
                assert (op_id + 3) == op2_op_ids[0]

                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete op0
                net.delete_op(op_id)  # delete op1
                net.delete_op(op_id)  # delete op2

                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_op2_concat_op)
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_POOL_ACTIVATION_ELW)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_POOL_ACTIVATION_ELW-----")
    op_pattern = [NpuPool, NpuActivation, NpuElemWise]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_RESIZE_ACTIVATION_ELW)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_RESIZE_ACTIVATION_ELW-----")
    op_pattern = [NpuResize, NpuActivation, NpuElemWise]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_POOL_ELW_ACTIVATION)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_POOL_ELW_ACTIVATION-----")
    op_pattern = [NpuPool, NpuElemWise, NpuActivation]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_RESIZE_ELW_ACTIVATION)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_RESIZE_ELW_ACTIVATION-----")
    op_pattern = [NpuResize, NpuElemWise, NpuActivation]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_ELW_POOL)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_ELW_POOL-----")
    op_pattern = [NpuActivation, NpuElemWise, NpuPool]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_ELW_RESIZE)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_ELW_RESIZE-----")
    op_pattern = [NpuActivation, NpuElemWise, NpuResize]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_POOL_ELW)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_POOL_ELW-----")
    op_pattern = [NpuActivation, NpuPool, NpuElemWise]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ELW_ACTIVATION_POOL)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ELW_ACTIVATION_POOL-----")
    op_pattern = [NpuElemWise, NpuActivation, NpuPool]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ELW_ACTIVATION_RESIZE)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ELW_ACTIVATION_RESIZE-----")
    op_pattern = [NpuElemWise, NpuActivation, NpuResize]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ELW_POOL_ACTIVATION)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ELW_POOL_ACTIVATION-----")
    op_pattern = [NpuElemWise, NpuPool, NpuActivation]
    _post_fuse_three_op(net, op_pattern)


@_register_ir_transformation_rule(TransformRule.NPU_CONV_ELW_RESIZE_ACTIVATION)
def _post_fuse_compose_pool_activation_elw(net: GraphIR):
    print("-----TransformRule NPU_CONV_ELW_RESIZE_ACTIVATION-----")
    op_pattern = [NpuElemWise, NpuResize, NpuActivation]
    _post_fuse_three_op(net, op_pattern)


# conv + acti + ['concat + conv, elw + conv,' resize + 'concat' + conv, resize + conv]
@_register_ir_transformation_rule(TransformRule.NPU_CONV_ACTIVATION_RESIZE)
def _post_fuse_activation_resize(net: GraphIR):
    print("-----TransformRule NPU_CONV_ACTIVATION_RESIZE-----")
    iter = 0
    for op in net.AllOps:
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            acti_op_ids = _find_post_op(net, op)
            acti_ops = []
            resize_ops = []
            resize_op_ids = []
            post_resize_ops = []
            post_resize_op_ids = []
            post_acti_resize_num = 0
            post_acti_concat_num = 0
            post_acti_elw_num = 0
            post_acti_other_num = 0
            post_resize_conv_num = 0
            post_resize_concat_num = 0
            post_resize_other_num = 0

            if len(acti_op_ids) == 1:
                acti_op = net.get_op(acti_op_ids[0])
                if isinstance(acti_op, NpuActivation):
                    acti_ops.append(acti_op)
                    post_acti_op_id = _find_post_op(net, acti_op)
                    print(len(post_acti_op_id))
                    for post_acti_op_id_idx in post_acti_op_id:
                        post_acti_op = net.get_op(post_acti_op_id_idx)
                        resize_ops.append(post_acti_op)
                        resize_op_ids.append(post_acti_op_id_idx)
                        post_resize_op_id = _find_post_op(net, post_acti_op)
                        for post_resize_op_id_idx in post_resize_op_id:
                            post_resize_op = net.get_op(post_resize_op_id_idx)
                            post_resize_ops.append(post_resize_op)
                            post_resize_op_ids.append(post_resize_op_id_idx)

                    for op_post_acti in resize_ops:
                        if isinstance(op_post_acti, NpuResize):
                            post_acti_resize_num += 1
                        else:
                            if isinstance(op_post_acti, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_acti):
                                    post_acti_concat_num += 1
                            else:
                                if isinstance(op_post_acti, NpuElemWise):
                                    if _check_post_op_only_conv_or_out(net, op_post_acti):
                                        post_acti_elw_num += 1
                                else:
                                    post_acti_other_num += 1
                    print(len(post_resize_ops))
                    for op_post_resize in post_resize_ops:
                        if isinstance(op_post_resize, NpuConv2d) or isinstance(op_post_resize, NpuOp):
                            post_resize_conv_num += 1
                        else:
                            if isinstance(op_post_resize, NpuConcat):
                                if _check_post_op_only_conv_or_out(net, op_post_resize):
                                    post_resize_concat_num += 1
                            else:
                                post_resize_other_num += 1
                    print("post_acti_resize_num: ", post_acti_resize_num)
                    print("post_acti_concat_num: ", post_acti_concat_num)
                    print("post_acti_elw_num: ", post_acti_elw_num)
                    print("post_acti_other_num: ", post_acti_other_num)
                    print("post_resize_conv_num: ", post_resize_conv_num)
                    print("post_resize_concat_num: ", post_resize_concat_num)
                    print("post_resize_other_num: ", post_resize_other_num)

                    if post_acti_other_num == 0 and post_resize_other_num == 0 and \
                            post_acti_resize_num > 0:
                        flag = True

            print("flag: ", flag)
            if flag:
                op_id = net.get_op_idx(op)
                if post_acti_concat_num + post_acti_elw_num + post_resize_concat_num == 0:
                    mode = 0
                else:
                    if (post_acti_concat_num + post_acti_elw_num) > 0 and post_resize_concat_num == 0:
                        mode = 1
                    else:
                        if (post_acti_concat_num + post_acti_elw_num) == 0 and post_resize_concat_num > 0:
                            mode = 2
                        else:
                            if (post_acti_concat_num + post_acti_elw_num) > 0 and post_resize_concat_num > 0:
                                mode = 3
                print("mode: ", mode)
                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0 or mode == 1:
                    npu_op.OutTensors = resize_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, resize_ops[0])
                if mode == 2 or mode == 3:
                    npu_op.OutTensors = post_resize_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, post_resize_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.ACTIVATION_RESIZE
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpActivate = True
                npu_op.NpuOpActivateOp = acti_ops[0]
                npu_op.NpuOpResize = True
                npu_op.NpuOpResizeOp = resize_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpActivateOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpResizeOp)

                post_resize_concat_op = None
                if mode == 1 or mode == 3:
                    npu_op.NpuOpResizeOut = True
                if mode == 2 or mode == 3:
                    npu_op.NpuOpConcat = True
                    for post_resize_concat_op in post_resize_ops:
                        if isinstance(post_resize_concat_op, NpuConcat):
                            npu_op.NpuOpConcatOp = post_resize_concat_op
                            break
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                assert (op_id + 1) == acti_op_ids[0]
                assert (op_id + 2) == resize_op_ids[0]
                net.delete_op(op_id)  # delete conv
                net.delete_op(op_id)  # delete acti
                net.delete_op(op_id)  # delete resize
                if mode == 2 or mode == 3:
                    assert op_id == net.get_op_idx(post_resize_concat_op)
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_CONV_RESIZE)
def _post_fuse_resize_single(net: GraphIR):
    print("-----TransformRule NPU_CONV_RESIZE-----")
    iter = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", iter, op.Name)
        iter += 1
        if isinstance(op, NpuConv2d):
            flag = False
            mode = -1
            resize_op_ids = _find_post_op(net, op)
            resize_ops = []
            concat_op_ids = []
            concat_ops = []
            conv_op_ids = []
            conv_ops = []
            other_ops = []
            if len(resize_op_ids) == 1:
                resize_op = net.get_op(resize_op_ids[0])
                if isinstance(resize_op, NpuResize):
                    _update_tensor_record(tensor_record, resize_op)
                    resize_ops.append(resize_op)
                    post_resize_op_id = _find_post_op(net, resize_op)
                    print(len(post_resize_op_id))
                    for post_resize_op_id_idx in post_resize_op_id:
                        post_resize_op = net.get_op(post_resize_op_id_idx)
                        if isinstance(post_resize_op, NpuConv2d) or isinstance(post_resize_op, NpuOp):
                            conv_ops.append(post_resize_op)
                            conv_op_ids.append(post_resize_op_id_idx)
                            continue
                        else:
                            if isinstance(post_resize_op, NpuConcat):
                                concat_flag = True
                                post_concat_op_ids = _find_post_op(net, post_resize_op)
                                for post_concat_op_ids_idx in post_concat_op_ids:
                                    post_concat_op = net.get_op(post_concat_op_ids_idx)
                                    if not (isinstance(post_concat_op, NpuConv2d) or isinstance(post_concat_op, NpuOp)):
                                        concat_flag = False
                                if concat_flag:
                                    concat_ops.append(post_resize_op)
                                    concat_op_ids.append(post_resize_op_id_idx)
                                continue
                            else:
                                other_ops.append(post_resize_op)

                    print(len(concat_ops), len(conv_ops), len(other_ops))
                    if ((len(concat_ops) > 0) or (len(conv_ops) > 0)) and (len(other_ops) == 0):
                        flag = True
            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                if mode == 0:
                    npu_op.OutTensors = resize_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, resize_ops[0])
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.RESIZE
                npu_op.NpuOpConvOp = op
                npu_op.NpuOpResize = True
                npu_op.NpuOpResizeOp = resize_ops[0]
                npu_op.NpuOpFlow.append(npu_op.NpuOpConvOp)
                npu_op.NpuOpFlow.append(npu_op.NpuOpResizeOp)

                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)

                npu_op.init_all()

                op_id = net.get_op_idx(op)
                assert (op_id + 1) == resize_op_ids[0]
                net.delete_op(op_id)
                net.delete_op(op_id)
                if mode == 1:
                    assert (op_id + 2) == concat_op_ids[0]
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


@_register_ir_transformation_rule(TransformRule.NPU_RESHAPE_FC)
def _post_fuse_reshape_fc(net: GraphIR):
    print("-----TransformRule NPU_RESHAPE_FC-----")
    for _op_id, op in enumerate(net.AllOps):
        if isinstance(op, NpuReshape):
            if op.OutputShape[0].H == op.OutputShape[0].W == 1:
                flag = False
                fc_op_ids = []
                fc_ops = []
                post_op_ids = _find_post_op(net, op)
                op_id = net.get_op_idx(op)

                for post_op_ids_idx in post_op_ids:
                    post_op = net.get_op(post_op_ids_idx)
                    if isinstance(post_op, NpuFullConnected):
                        flag = True
                        fc_ops.append(post_op)
                        fc_op_ids.append(post_op_ids_idx)

                if len(fc_ops) == 1:
                    flag = True
                else:
                    flag = False

                print(flag)
                if flag:
                    fc_op = fc_ops[0]
                    # _copy_opbase_input_info(fc_op, op)
                    replace_tensor_index = fc_op.InTensors.index(op.OutTensors[0])
                    assert len(op.InTensors) == 1
                    fc_op.InTensors[replace_tensor_index] = op.InTensors[0]
                    net.delete_op(op_id)


@_register_ir_transformation_rule(TransformRule.NPU_FC)
def _post_fuse_fc_single(net: GraphIR):  # 全连接层融合
    print("------TransformRule NPU_FC-----")
    i = 0
    tensor_record = []
    for _op_id, op in enumerate(net.AllOps):
        _update_tensor_record(tensor_record, op)
        print("iter: ", i, op.Name)
        i += 1
        if isinstance(op, NpuFullConnected):  # FullConnect
            flag = False
            mode = -1

            if _check_post_op_out(net, op):
                flag = True
            concat_op_ids = []
            concat_ops = []
            post_op_ids = _find_post_op(net, op)
            for post_op_ids_idx in post_op_ids:
                post_op = net.get_op(post_op_ids_idx)
                if isinstance(post_op, NpuConcat):  # FullConnect + Concat
                    if _check_post_op_only_conv_or_out(net, post_op):
                        flag = True
                        concat_ops.append(post_op)
                        concat_op_ids.append(post_op_ids_idx)

            for post_op_ids_idx in post_op_ids:
                post_op = net.get_op(post_op_ids_idx)
                if not isinstance(post_op, NpuConcat):
                    flag = False

            if _check_post_op_only_conv_or_out(net, op):  # FullConnect + Conv / FullConnect / NPU / Pad
                flag = True

            if _check_post_op_only_cpu_op(net, op):  # FullConnect + Reshape / Transpose / Softmax
                flag = True

            print(flag)
            if flag:
                mode = _check_concat_mode(concat_ops, tensor_record)

                npu_op = NpuOp()
                npu_op.NpuOpConv = False
                npu_op.NpuOpFc = True
                npu_op.NpuOpFcOp = op
                npu_op.InTensors = op.InTensors
                _copy_opbase_input_info(npu_op, op)
                npu_op.NpuOpFlow.append(npu_op.NpuOpFcOp)

                if mode == 0:
                    npu_op.OutTensors = op.OutTensors
                    _copy_opbase_output_info(npu_op, op)
                if mode == 1:
                    npu_op.OutTensors = concat_ops[0].OutTensors
                    _copy_opbase_output_info(npu_op, concat_ops[0])

                npu_op.NpuOpMode = VpuPostOpSetMode.NONE
                if mode == 1:
                    npu_op.NpuOpConcat = True
                    npu_op.NpuOpConcatOp = concat_ops[0]
                    npu_op.NpuOpConcatOp.main_input_tensor_id = npu_op.NpuOpFlow[-1].OutTensors
                    npu_op.NpuOpFlow.append(npu_op.NpuOpConcatOp)
                npu_op.init_all()

                op_id = net.get_op_idx(op)
                net.delete_op(op_id)

                if mode == 1:
                    assert op_id == concat_op_ids[0]
                    net.delete_op(op_id)
                net.insert_op(npu_op, op_id)
                print("op_id:", op.TopOpId)
                npu_op.NpuOpId = op.TopOpId


# op_fuse_pass
op_fuse_transform = [TransformRule.ORDER_TOP_OPS,
                     TransformRule.NPU_PAD_CONV,
                     TransformRule.NPU_RESHAPE_CONV,
                     TransformRule.NPU_CONV,
                     TransformRule.NPU_PAD_POOL,
                     TransformRule.NPU_CONV_ELW,
                     TransformRule.NPU_CONV_POOL,
                     TransformRule.NPU_CONV_POOL_ELW,
                     TransformRule.NPU_CONV_POOL_ACTIVATION,
                     TransformRule.NPU_CONV_POOL_ACTIVATION_ELW,
                     TransformRule.NPU_CONV_POOL_ELW_ACTIVATION,
                     TransformRule.NPU_CONV_RESIZE,
                     TransformRule.NPU_CONV_RESIZE_ACTIVATION_ELW,
                     TransformRule.NPU_CONV_RESIZE_ELW_ACTIVATION,
                     TransformRule.NPU_CONV_ACTIVATION,
                     TransformRule.NPU_CONV_ACTIVATION_POOL,
                     TransformRule.NPU_CONV_ACTIVATION_ELW,
                     TransformRule.NPU_CONV_ACTIVATION_POOL_ELW,
                     TransformRule.NPU_CONV_ACTIVATION_RESIZE,
                     TransformRule.NPU_CONV_ACTIVATION_ELW_POOL,
                     TransformRule.NPU_CONV_ACTIVATION_ELW_RESIZE,
                     TransformRule.NPU_CONV_ELW_ACTIVATION_POOL,
                     TransformRule.NPU_CONV_ELW_ACTIVATION_RESIZE,
                     TransformRule.NPU_CONV_ELW_POOL_ACTIVATION,
                     TransformRule.NPU_CONV_ELW_RESIZE_ACTIVATION,
                     TransformRule.NPU_CONV,
                     TransformRule.NPU_FC,
                     TransformRule.NPU_RESHAPE_FC,
                     TransformRule.ORDER_NPU_OPS
                     ]
