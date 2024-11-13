import math

from ir.conversion.ir_transform import _register_ir_transformation_rule
from ir.dialect.npu.IR_operator import *
from ir.graph.Graph_IR import *


class TransformRule(Enum):
    NOPE = 1

    WEIGHT_MAPPING = 2
    WEIGHT_PADDING = 3


@_register_ir_transformation_rule(TransformRule.WEIGHT_PADDING)  # 填充至符合芯片size的shape
def _weight_padding(net: GraphIR):
    for op_id, npu_op in enumerate(net.AllOps):
        if isinstance(npu_op, NpuOp) and npu_op.NpuOpConv:
            # weight_split
            npu_conv_op = npu_op.NpuOpConvOp
            if npu_conv_op is None:
                continue
            weight = npu_conv_op.WeightValue
            bais = npu_conv_op.BiasValue
            k_n, k_h, k_w, k_c = weight.shape

            if k_n % 16 != 0:
                # TODO 选择
                n_k_n = math.ceil(k_n / 32) * 32
                # if k_n < 32:
                #     n_k_n = 32
                # elif k_n < 64:
                #     n_k_n = 64
                # elif k_n < 128:
                #     n_k_n = 128
                # elif k_n < 256:
                #     n_k_n = 256
                # else:
                #     n_k_n = math.ceil(k_n / 16) * 16
                weight_ = np.zeros([n_k_n, k_h, k_w, k_c])  # .astype(np.int8)
                weight_[0:k_n, :, :, :] = weight
                npu_conv_op.WeightValue = weight_  # 给原weight填充0

                bais_ = np.zeros([n_k_n])  # .astype(np.int32)
                bais_[0:k_n] = bais
                npu_conv_op.BiasValue = bais_

                # TODO Q:?什么时候更新的C  A：在layer_group
                # assert npu_conv_op.OutputShape[0].C == n_k_n
                npu_conv_op.OutputShape[0].C = n_k_n


@_register_ir_transformation_rule(TransformRule.WEIGHT_MAPPING)
def _weight_mapping(net: GraphIR):
    for op_id, npu_op in enumerate(net.AllOps):
        if isinstance(npu_op, NpuOp):
            npu_conv_op = npu_op.NpuOpConvOp
            if npu_conv_op is None:
                continue
            npu_op_id = npu_op.NpuOpId
            k_n = npu_conv_op.WeightValue.shape[0]
            weight = {
                "weight": npu_conv_op.WeightValue.reshape(k_n, -1).transpose(1, 0).tolist(),

                "bias": npu_conv_op.BiasValue.tolist()
            }
            net.add_weight_tensor(npu_op_id, weight)


# weight_mapping_pass
weight_mapping_transform = [
    # TransformRule.EASY_WEIGHT_PADDING,
    TransformRule.WEIGHT_MAPPING,
]
