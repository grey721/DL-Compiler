from enum import Enum

from ir.conversion.ir_transform import _register_ir_transformation_rule
from ir.graph.Graph_IR import *


class TransformRule(Enum):
    NOPE = 1
    EASY_MEMORY = 2


def addr_gener():
    yield


@_register_ir_transformation_rule(TransformRule.EASY_MEMORY)
def _memory_assign(net: GraphIR):
    addr = 0
    addr_dict = {}
    for idx, npu_op in enumerate(net.AllOps):
        print(f"iter:{idx}, {npu_op.Type}")
        post_npu_ops = npu_op.PostOpId

        if len(post_npu_ops) > 1 or npu_op.write:
            addition = []
            if len(npu_op.OutTensors) == 1:
                t = npu_op.OutTensors[0]
                if t not in npu_op.write_list:
                    addition.append(t)

            else:
                flag = True
                for t in npu_op.OutTensors:
                    if flag and t in net.AllOps[idx + 1].InTensors:
                        flag = False
                        continue
                    if t not in npu_op.write_list:
                        addition.append(t)

            if isinstance(npu_op, NpuOp):
                npu_op.NpuOpFlow[-1].write = True
                for t in addition:
                    npu_op.NpuOpFlow[-1].write_list.append(t)
                    npu_op.write_list.append(t)
            else:
                npu_op.write = True
                for t in addition:
                    npu_op.write_list.append(t)

            for t in npu_op.write_list:
                addr_dict[t] = addr

            print("     Write", npu_op.write_list)

        if len(npu_op.InTensors) > 1 or npu_op.read:

            addition = []
            flag = True
            for t in net.AllOps[idx - 1].OutTensors:
                if flag and t in npu_op.InTensors:
                    flag = False
                    continue
                if t not in npu_op.read_list:
                    addition.append(t)

            if isinstance(npu_op, NpuOp):
                npu_op.NpuOpFlow[0].read = True
                npu_op.NpuOpFlow[0].read_list.extend(addition)
            else:
                npu_op.read = True
                npu_op.read_list.extend(addition)

            print("     Read", )

    print(addr_dict)


# memory_assign_pass
memory_assign_transform = [
    TransformRule.EASY_MEMORY
]
