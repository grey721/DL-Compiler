from ir.dialect.top.IR_operator import *


def get_top_graph_op_list(top_graph):
    keyword = ["op", "input_h", "input_w", "input_c",
               "output_h", "output_w", "output_c", "ker_h", "ker_w",
               "stride_h", "stride_w", "padding", "pad_h", "pad_w",
               "group", "bias", "input_idx", "output_idx", "MAC",
               "fmi_data_size", "fmo_data_size", "kernel_data_size"]

    graph_op_list = []
    for op in top_graph.AllOps:
        base_dict = {k: '' for k in keyword}
        base_dict['op'] = op.Type
        base_dict['input_h'] = op.InputShape[0].H
        base_dict['input_w'] = op.InputShape[0].W
        base_dict['input_c'] = op.InputShape[0].C
        base_dict['output_h'] = op.OutputShape[0].H
        base_dict['output_w'] = op.OutputShape[0].W
        base_dict['output_c'] = op.OutputShape[0].C

        if isinstance(op, ConvBase) or isinstance(op, Pool):
            base_dict['ker_h'] = op.KerH
            base_dict['ker_w'] = op.KerW
            base_dict['stride_h'] = op.StrideH
            base_dict['stride_w'] = op.StrideW
            base_dict['padding'] = op.Padding
            base_dict['pad_h'] = op.PadH
            base_dict['pad_w'] = op.PadW

        if isinstance(op, ConvBase):  # or isinstance(op, FullConnected):
            base_dict['group'] = op.Group if isinstance(op, ConvBase) else None
            base_dict['bias'] = op.Bias
            base_dict['MAC'] = op.get_mac()
            base_dict['kernel_data_size'] = op.get_weight_size() / 1024

        base_dict['input_idx'] = op.InTensors
        base_dict['output_idx'] = op.OutTensors

        base_dict['fmi_data_size'] = op.get_fmi_size() / 1024
        base_dict['fmo_data_size'] = op.get_fmo_size() / 1024

        graph_op_list.append(base_dict)

    return graph_op_list
