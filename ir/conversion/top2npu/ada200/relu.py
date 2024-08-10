from compiler.conversion.top2npu.ada200.base import TransformRule, _register_tranformation_rule
from compiler.conversion.top2npu.top_lowing import *
from compiler.dialect.npu.ir.ir_operator import *

#todo 
@_register_tranformation_rule(TransformRule.RELU_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Activation):
            if op.ActiMode != ActivationMode.AM_RELU:
                continue
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op,net):
    npu_activation = NpuActivation()
    npu_activation.__dict__.update(op.__dict__)
    npu_activation.Name = "NpuActivation_Relu"


    return npu_activation


def _lowering_fp32(op):
    raise NotImplementedError
