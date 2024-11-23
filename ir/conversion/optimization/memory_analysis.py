from enum import Enum

from ir.conversion.ir_transform import _register_ir_transformation_rule
from ir.graph.Graph_IR import *


class TransformRule(Enum):
    NOPE = 1
    MEMORY_OPERATION_ANALYSIS = 2
    LIFE_CYCLE = 3


def addr_generator(size):
    yield


@_register_ir_transformation_rule(TransformRule.MEMORY_OPERATION_ANALYSIS)
def _memory_analysis(net: GraphIR):
    print("----start TransformRule EASY_MEMORY_ANALYSIS-----")
    for idx, npu_op in enumerate(net.AllOps):
        print(f"iter:{idx}, {npu_op.Type}")
        post_npu_ops = npu_op.PostOpId

        if len(post_npu_ops) > 1 or npu_op.write:
            addition = []
            if len(npu_op.OutTensors) == 1 and len(post_npu_ops) > 1:
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

            if isinstance(npu_op, NpuOp) and addition:
                npu_op.NpuOpFlow[-1].write = True
                npu_op.NpuOpFlow[-1].write_list.extend(addition)
                npu_op.write_list.extend(addition)
            else:
                npu_op.write = True
                for t in addition:
                    npu_op.write_list.append(t)

            print("     Write", npu_op.write_list)

        pre_npu_ops = npu_op.PreOpId
        if len(pre_npu_ops) > 1 or npu_op.read:
            fmi = [t for t in npu_op.InTensors if net.AllTensors[t].Type == TensorType.Intermediate]
            addition = []
            if len(fmi) > 1:
                flag = True
                for t in fmi:
                    if flag and t in net.AllOps[idx - 1].OutTensors:
                        flag = False
                        continue
                    if t not in npu_op.read_list:
                        addition.append(t)

            if isinstance(npu_op, NpuOp) and addition:
                npu_op.NpuOpFlow[0].read = True
                npu_op.NpuOpFlow[0].read_list.extend(addition)
                for t in addition:
                    if t not in npu_op.read_list:
                        npu_op.read_list.append(t)
            else:
                npu_op.read = True
                npu_op.read_list.extend(addition)

            print("     Read", npu_op.read_list)


@_register_ir_transformation_rule(TransformRule.LIFE_CYCLE)
def _memory_assign(net: GraphIR):
    print("----start TransformRule EASY_MEMORY_ASSIGN-----")
    life_cycle = {}
    for npu_op in net.AllOps:
        if isinstance(npu_op, NpuOp):
            for op in npu_op.NpuOpFlow:
                if op.read:
                    for t in op.read_list:
                        if t in life_cycle:
                            life_cycle[t] += 1
                        else:
                            life_cycle[t] = 1
        else:
            if npu_op.read:
                for t in npu_op.read_list:
                    if t in life_cycle:
                        life_cycle[t] += 1
                    else:
                        life_cycle[t] = 1
    print(life_cycle)

    for npu_op in net.AllOps:
        if isinstance(npu_op, NpuOp):
            for op in npu_op.NpuOpFlow:
                if op.write:
                    for t in op.write_list:
                        # TODO 分配地址  {addr: tensor_id, addr: None} tensor_id代表该地址存储的张量，None表示暂空
                        pass

                if op.read:
                    for t in op.read_list:
                        life_cycle[t] -= 1
                        if life_cycle[t] == 0:
                            # TODO 释放该地址，使 {addr: t} -> {addr: None}
                            pass
        else:
            if npu_op.write:
                for t in npu_op.write_list:
                    # TODO 分配地址  {addr: tensor_id, addr: None} tensor_id代表该地址存储的张量，None表示暂空
                    pass

            if npu_op.read:
                for t in npu_op.read_list:
                    life_cycle[t] -= 1
                    if life_cycle[t] == 0:
                        # TODO 释放该地址，使 {addr: t} -> {addr: None}
                        pass


# memory_assign_pass
memory_analysis_transform = [
    TransformRule.MEMORY_OPERATION_ANALYSIS,
    TransformRule.LIFE_CYCLE
]
