from enum import Enum
from typing import Callable, Dict
from ir.dialect.npu.IR_operator import *


class IRTransformer:
    transform_map = []

    def add_transform_option(self, transform_map):
        self.transform_map = transform_map
        for option in transform_map:
            print(option)

    def transform(self, ir_graph):
        for option in self.transform_map:
            TRANSFORM_MAP[option](ir_graph)
        return ir_graph


# 开始优化前，所有优化函数会被映射到该字典内,以供后续调用。以枚举类为键
TRANSFORM_MAP: Dict[Enum, Callable] = {}


def _register_ir_transformation_rule(transform_rule):
    def callback(impl):
        TRANSFORM_MAP[transform_rule] = impl

    return callback


def _find_pre_op(net, op):
    pre_op_id = []
    inputs = op.InTensors
    for pre_op in net.AllOps:
        outputs = pre_op.OutTensors
        for t_name in outputs:
            if t_name in inputs:
                pre_op_id.append(net.get_op_idx(pre_op))

    return pre_op_id


def _find_post_op(net, op):
    post_op_id = []
    outputs = op.OutTensors
    for post_op in net.AllOps:
        inputs = post_op.InTensors
        for t_name in inputs:
            if t_name in outputs:
                post_op_id.append(net.get_op_idx(post_op))

    return post_op_id


def _copy_opbase_input_info(op_dst, op_src):
    op_dst.InTensors = op_src.InTensors[:]
    op_dst.InputShape = op_src.InputShape[:]


def _copy_opbase_output_info(op_dst, op_src):
    op_dst.OutTensors = op_src.OutTensors[:]
    op_dst.OutputShape = op_src.OutputShape[:]


def _order_post_op(net, op):
    post_op_id = []
    if isinstance(op, NpuOp):
        outputs = op.fmo_tensor[:]
        outputs.extend(op.short_cut_out_tensor)
    else:
        outputs = op.OutTensors

    for post_op in net.AllOps:
        if isinstance(post_op, NpuOp):
            inputs = post_op.fmi_tensor[:]
            inputs.extend(post_op.elemwise_input_tensor)
            inputs.extend(post_op.concat_input_tensor)
            for t_name in inputs:
                if t_name in outputs:
                    op_id = net.get_op_idx(post_op)
                    if op_id is not None:
                        post_op_id.append(op_id)
        else:
            inputs = post_op.InTensors
            for t_name in inputs:
                if t_name in outputs:
                    op_id = net.get_op_idx(post_op)
                    if op_id is not None:
                        post_op_id.append(op_id)
    return post_op_id


def _order_pre_op(net, op):
    pre_op_id = []
    if isinstance(op, NpuOp):
        inputs = op.fmi_tensor[:]
        concat_inputs = op.concat_input_tensor
        inputs.extend(concat_inputs)
        elemwise_inputs = op.elemwise_input_tensor
        inputs.extend(elemwise_inputs)
    else:
        inputs = op.InTensors

    for pre_op in net.AllOps:
        if isinstance(pre_op, NpuOp):
            if pre_op.short_cut_out_tensor:
                outputs = pre_op.short_cut_out_tensor[:]
            else:
                outputs = []
            outputs.extend(pre_op.fmo_tensor)  # 当前Op的所有输出
            for tensor in inputs:  # 指定op的输入
                if tensor in outputs:
                    op_id = net.get_op_idx(pre_op)
                    if op_id is not None:
                        pre_op_id.append(op_id)
        else:
            outputs = pre_op.OutTensors
            for tensor in inputs:  # 指定op的输入
                if tensor in outputs:
                    op_id = net.get_op_idx(pre_op)
                    if op_id is not None:
                        pre_op_id.append(op_id)

    return pre_op_id
