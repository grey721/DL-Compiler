import cv2
import os
import onnx
import onnxruntime as ort
import numpy as np
from ir.graph.Graph_IR import *
import json


class ONNXRUNER:
    def __init__(self, model_path, input_path, result_path):
        # 准备输入数据（这里以单张图片为例，且需要调整为模型所需的输入尺寸）
        # 假设你的输入图片已经加载到变量img中，并且已经调整到了正确的尺寸
        # 你还需要将图片数据转换为模型所需的格式（通常是NCHW，即[batch_size, channels, height, width]）
        # 这里仅作为示例，没有展示图片加载和预处理的具体代码
        img = cv2.imread(input_path)
        img = cv2.resize(img, (640, 640))  # 假设模型输入尺寸为640x640
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR到RGB，HWC到CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0  # 归一化

        # 获取模型的输入信息
        # 加载ONNX模型
        model = onnx.load(model_path)

        # 将每个算子的输出添加为模型的输出
        for node in model.graph.node:
            for output in node.output:
                if not any(output == o.name for o in model.graph.output) and node.op_type != 'Constant':
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        # 序列化模型
        model_serialized = model.SerializeToString()

        # 使用ONNX Runtime加载模型
        self.session = ort.InferenceSession(model_serialized)

        # 准备输入数据
        input_name = self.session.get_inputs()[0].name
        input_data = img

        # 执行模型推理
        self.outputs = [x.name for x in self.session.get_outputs()]
        self.results = self.session.run(None, {input_name: input_data})

        self.verification(result_path)

    def print_all_result(self):
        # 打印每个算子的输出
        for output, result in zip(self.outputs, self.results):
            print(f"Output {output}:")
            print(np.array(result).shape)

    def get_output_tensors(self, tensor_names):
        return [self.results[self.outputs.index(tensor_name)] for tensor_name in tensor_names]

    def get_op_outputs_with_tensor_ids(self, net: GraphIR, out_tensors):
        output = [net.AllTensors[t_idx].Name for t_idx in out_tensors]
        output = [self.results[self.outputs.index(t_name)] for t_name in output if t_name in self.outputs]
        return output

    def get_op_outputs(self, net: GraphIR, op):
        output = [net.AllTensors[t_idx].Name for t_idx in op.OutTensors]
        output = [self.results[self.outputs.index(t_name)] for t_name in output if t_name in self.outputs]
        return output

    def get_op_outputs_with_op_idx(self, net: GraphIR, op_idx):
        output = [net.AllTensors[t_idx].Name for t_idx in net.AllOps[op_idx].OutTensors]
        output = [self.results[self.outputs.index(t_name)] for t_name in output if t_name in self.outputs]
        return output

    def verification(self, path):
        print("verificate：", path)
        dir_list = os.listdir(path)
        ver_res = {}
        if 'top_info.json' in dir_list:
            with open(f'{path}/top_info.json', 'r', encoding='utf-8') as f:
                top_info = json.load(f)  # 从文件对象中加载 JSON 数据
            layer_num = top_info["layer_num"]
            for i in range(layer_num):
                json_path = f"layer_{i}"
                with open(f'{path}/{json_path}/operator.json', 'r', encoding='utf-8') as j:
                    info = json.load(j)
                    op_info = info["flow"][-1]

                    temp = {}
                    temp["output"] = self.get_output_tensors(op_info["OutTensors"])
                    temp["output_shape"] = [t.shape for t in temp["output"]]
                    temp["output_dims"] = [len(shape) for shape in temp["output_shape"]]
                    temp["output"] = [t.tolist() for t in temp["output"]]
                    ver_res[json_path] = temp

            with open(f'{path}/verification.json', 'w') as f:
                json.dump(ver_res, f, indent=4)  # , indent=4
        else:
            raise FileNotFoundError(f'Can not find top_info.json in "{path}"')


if __name__ == "__main__":
    run = ONNXRUNER(
        model_path='assets/yolov5s.onnx',
        input_path='verification/input/xiaoxin.jpg',
        result_path='output/yolov5s'
    )
