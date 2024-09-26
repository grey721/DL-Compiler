import json
import os
from enum import Enum

import cv2

from ir.conversion.codegen.base import *
from ir.conversion.ir_transform import _register_ir_transformation_rule


class TransformRule(Enum):
    CREATE_REGISTER = 1
    CREATE_INFO = 2  # txt保存模型输入输出，层，block等信息;dma_read_list非空，则保存json包括txt的信息和输入输出形状信息和dma的信息
    CREATE_XLSX = 3  # 将block信息保存成xlsx
    CREATE_TOP_INFO = 4
    CALCULATING_THE_WEIGHT = 5
    CREATE_REGISTER_WITH_DMA = 6
    EASY_OUTPUT = 7


def make_weight(top_info_path, npu_graph):  # 将权重保存成文件
    weight_path = f'{top_info_path}/weight'
    weight = npu_graph.WeightTensors
    f = open(weight_path, 'w')
    for i in weight:
        for j in i:
            f.write(j + '\n')
    f.close()


def make_image_to_memory(top_info_path, image_size=(128, 128, 3), image_path=''):  # 保存图片二进制数据
    top_info_path = f'{top_info_path}/image'
    if image_path == '':
        image = (np.random.rand(image_size[0] * image_size[1] * image_size[2]).reshape(image_size) * 255 - 128
                 ).astype(np.int8)  # 随机生成图片，随机生成图片size,*255是为了值[0~255)，-128是是为了[-128,127)，
    else:
        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("image is not exist")

    f = open(top_info_path, '+w')
    string = ''
    n = 0
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            for z in range(image_size[2]):
                number = intToBin(image[i][j][z], 8, feature=True)
                number = binTohex(number, 8)
                string = number + string
                n += 1
                if n == 3:  # 3通道？固定3一回车？
                    f.write(string + '\n')
                    n = 0
                    string = ''
    f.close()
    pass


def gather_register(res_list, top_head, head_list, finally_path):
    f = open(finally_path, 'w')
    for i in range(len(top_head)):
        f.write(top_head[i] + '\n')
    for i in range(len(res_list)):
        for j in range(len(head_list[i])):
            f.write(head_list[i][j] + '\n')
        for j in range(len(res_list[i])):
            f.write(res_list[i][j])
    f.close()
    pass


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (int(round(dw - 0.1)), int(round(dh - 0.1)))


def make_image(image_path, Scale, ZeroPoint, size=(416, 416, 3)):
    im = cv2.imread(image_path)
    shape = im.shape
    im, ratio, (dw, dh) = letterbox(im, size, stride=32, auto=False)
    im = im.transpose((2, 0, 1))[::-1].transpose((1, 2, 0))
    im = np.ascontiguousarray(im)

    # im = im[..., ::-1]  # conver RGB to BGR
    x = im.astype(np.float32)
    x /= 255
    x = (x / Scale + ZeroPoint).astype(np.int8)
    return x


def fill_zero(block_tensor):
    h, w, c = block_tensor.shape
    tem_tensor = np.zeros(shape=[h, w, 8]).astype(np.int8)
    tem_tensor[:, :, 2] = block_tensor[:, :, 2]
    tem_tensor[:, :, 1] = block_tensor[:, :, 1]
    tem_tensor[:, :, 0] = block_tensor[:, :, 0]
    return tem_tensor


def Array2Txt_hwc_hex_bank(x, in_bit=8, out_bit=64, path_="", bank_start=0):
    n = 0
    path = path_
    for i in range(len(x)):
        f = open(path, "w")
        string = ""
        for j in range(len(x[i])):
            number = intToBin(x[i][j], in_bit, feature=True)
            number = binTohex(number, int(out_bit / in_bit))
            string = number + string
            n += 1
            if n == 8:
                string += "\n"
                f.write(string)
                string = ""
                n = 0
        f.close()


@_register_ir_transformation_rule(TransformRule.EASY_OUTPUT)
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

    make_image_to_memory(path, image_size=[128, 128, 3], image_path='')  # 保存图片二进制数据
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
                p_dict = p.to_param_dict()
                op_dict["flow"].append(p_dict)

        else:
            param_dict = op.to_param_dict()
            op_dict = {"type": op.Type,
                       "provider": op.PreOpId,
                       "consumer": op.PostOpId,
                       "input_dim": param_dict["InputShape"],
                       "input_dim_num": [len(t) for t in param_dict["InputShape"]],
                       "output_dim": param_dict["OutputShape"],
                       "output_dim_num": [len(t) for t in param_dict["OutputShape"]],
                       "flow": [param_dict]
                       }
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

        # Weight 输出
        if op_id in npu_graph.WeightTensorIds:
            weight_idx = npu_graph.WeightTensorIds.index(op_id)
            res = npu_graph.WeightTensors[weight_idx]

            with open(f'{layer_path}/weight.json', 'w') as f:
                json.dump(res, f, indent=4)  # , indent=4

    # output_dict = {"node_list": []}
    # for op in ops:
    #     output_dict["node_list"].append(op.to_param_dict())
    # print(output_dict)
    # with open(f'{path}/info.json', 'w') as f:
    #     json.dump(output_dict, f, indent=4)

    # with open('config.json', 'w') as f:
    #             json.dump(content, f, indent=4)
    # import pickle
    # with open(f"{path}/npu_graph.pkl", "wb") as f:
    #     pickle.dump(npu_graph, f)


codegen_transform = [
    TransformRule.EASY_OUTPUT,
]
