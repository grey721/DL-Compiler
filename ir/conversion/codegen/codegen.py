import json
import os
from enum import Enum

from ir.graph.Graph_IR import *
from ir.conversion.ir_transform import _register_ir_transformation_rule


class TransformRule(Enum):
    OUTPUT = 1


def tensors_ids2names(npu_graph: GraphIR, ids):
    return [npu_graph.AllTensors[idx].Name for idx in ids]


@_register_ir_transformation_rule(TransformRule.OUTPUT)
def easy_info(npu_graph: GraphIR):
    path = f'{npu_graph.codegen_path}/{npu_graph.name}'
    if os.path.exists(path):
        _tail = 1
        _path = path + f"_{_tail}"
        while os.path.exists(_path):
            _tail += 1
            _path = path + f"_{_tail}"
        path = _path
        npu_graph.name += f"_{_tail}"
    os.makedirs(path)

    # make_image_to_memory(path, image_size=[128, 128, 3], image_path='')  # 保存图片二进制数据
    net_input_shapes = [npu_graph.AllTensors[idx].Shape.list for idx in npu_graph.NetInTensors]
    net_output_shapes = [npu_graph.AllTensors[idx].Shape.list for idx in npu_graph.NetOutTensors]
    top_info = {
        "name": npu_graph.name,
        "layer_num": len(npu_graph.AllOps),
        "input_num": len(npu_graph.NetInTensors),
        "input_dims": [len(shape) for shape in net_input_shapes],
        "input_shape":  net_input_shapes,
        "output_num": len(npu_graph.NetOutTensors),
        "output_dims": [len(shape) for shape in net_output_shapes],
        "output_shape": net_output_shapes,
    }
    with open(f'{path}/top_info.json', 'w') as f:
        json.dump(top_info, f, indent=4)  # , indent=4

    # 输出权重
    for idx, op in enumerate(npu_graph.AllOps):
        layer_path = f'{path}/layer_{idx}'
        if not os.path.exists(layer_path):
            os.makedirs(layer_path)

        ElemWiseOp = None

        # Op输出
        if isinstance(op, NpuOp):
            op_dict = {"type": op.Type,
                       "provider": op.PreOpId,
                       "consumer": op.PostOpId,
                       "input_dim": [shape.list for shape in op.InputShape],
                       "input_dim_num": [len(t.list) for t in op.InputShape],
                       "output_dim": [shape.list for shape in op.OutputShape],
                       "output_dim_num": [len(t.list) for t in op.OutputShape],
                       "flow": []
                       }
            for p in op.NpuOpFlow:
                param_dict = p.to_param_dict()
                param_dict["InTensors"] = tensors_ids2names(npu_graph, param_dict["InTensors"])
                param_dict["OutTensors"] = tensors_ids2names(npu_graph, param_dict["OutTensors"])
                op_dict["flow"].append(param_dict)

            if op.NpuOpElemWise:
                ElemWiseOp = op.NpuOpElemWiseOp

        else:
            param_dict = op.to_param_dict()
            param_dict["InTensors"] = tensors_ids2names(npu_graph, param_dict["InTensors"])
            param_dict["OutTensors"] = tensors_ids2names(npu_graph, param_dict["OutTensors"])

            op_dict = {"type": op.Type,
                       "provider": op.PreOpId,
                       "consumer": op.PostOpId,
                       "input_dim": param_dict["InputShape"],
                       "input_dim_num": [len(t) for t in param_dict["InputShape"]],
                       "output_dim": param_dict["OutputShape"],
                       "output_dim_num": [len(t) for t in param_dict["OutputShape"]],
                       "flow": [param_dict]
                       }

            if isinstance(op, NpuElemWise):
                ElemWiseOp = op

        try:
            with open(f'{layer_path}/operator.json', 'w') as f:
                json.dump(op_dict, f, indent=4)  # , indent=4
        except:
            print(op.Type)
            for key in op_dict:
                print(key, type(op_dict[key]))
                print(op_dict[key])
            raise ValueError
        op_id = op.NpuOpId

        if ElemWiseOp is not None:
            if ElemWiseOp.Mode < 0 and ElemWiseOp.B == 0:
                with open(f'{layer_path}/B.json', 'w') as f:
                    tensorB = npu_graph.AllTensors[ElemWiseOp.InTensors[1]]
                    json.dump({"B": tensorB.Data.tolist()}, f, indent=4)  # , indent=4

        # Weight 输出
        if op_id in npu_graph.WeightTensorIds:
            weight_idx = npu_graph.WeightTensorIds.index(op_id)
            res = npu_graph.WeightTensors[weight_idx]

            with open(f'{layer_path}/weight.json', 'w') as f:
                json.dump(res, f, indent=4)  # , indent=4

    import pickle
    with open(f"{path}/npu_graph.pkl", "wb") as f:
        pickle.dump(npu_graph, f)


codegen_transform = [
    TransformRule.OUTPUT,
]
