import json

from frontend.tool.constant_ONNX import *
from ir.graph.Graph_IR import *


def get_np_data_from_attribute(attr):
    """tensor in initializer"""
    # 根据onnx中标记的类型，映射np的类型
    dtype = onnx2np_dtype_mapping[attr.data_type]
    # shape = []
    if attr.dims == [0]:
        return np.array([], dtype) if dtype else np.array([], np.float32), dtype
    shape = attr.dims
    # for n in attr.dims:
    #    shape.append(n)
    if len(attr.raw_data) != 0:  # 通常用于存储二进制格式的权重和偏置等参数
        # raw_data是字节数组，所以用np.from buffer读取数据;如果没有dtype，就默认类型，根据字节数据赋值
        np_data = np.frombuffer(attr.raw_data, dtype=dtype) if dtype else np.frombuffer(attr.raw_data)
    elif len(attr.float_data) != 0:  # 用于存储浮点数类型的数据，如模型的某些参数或中间结果。
        np_data = np.array(attr.float_data, dtype=np.float32)
    elif len(attr.int32_data) != 0:  # 分别用于存储32位和64位整数类型的数据。这些数据可能用于表示模型的某些属性、索引或其他整数类型的信息。
        np_data = np.array(attr.int32_data, dtype=np.int32)
    elif len(attr.int64_data) != 0:
        np_data = np.array(attr.int64_data, dtype=np.int64)
    else:
        raise AttributeError(f"Invalid Data Type : {attr}")
    # reshape(shape)按照shape改变np_data相应维度的形状
    return np_data.reshape(shape), dtype


class ONNX2TopIR:
    def __init__(self, model_path, config_path=None):
        # 初始化
        self.model = onnx.load(model_path)
        self.print_model_info()
        self.graph = GraphIR()  # 创建图IR对象

        # 量化
        if config_path is None:
            self.quantization_config = None
        else:
            with open(config_path, 'rb') as f:
                self.quantization_config = json.load(f)

        # 解析
        self.fused_ops = []
        for n, op in enumerate(self.model.graph.node):
            if self._get_op_code(op) not in FUSIBLE_OPs:
                self.fused_ops.append(op)
                continue
            print(op.name, 'not in FUSIBLE_OPs')
            # TODO "OVERLAPPED"是什么状态？为什么跳过
            in_tensors_name = op.input
            state = self.quantization_config["configs"][op.name][in_tensors_name[0]]['state']
            if state == "OVERLAPPED":  # 避免重复处理：在某些情况下，"OVERLAPPED"状态可能表示该算子已经以某种方式被处理过？好像中间变量都是OVERLAPPED？
                continue
            self.fused_ops.append(op)
        print("Operators_len:", len(self.fused_ops))

    def _get_op_code(self, op):  # 获取算子类型的id
        if op.op_type not in ONNXType2OperatorType:
            raise NotImplementedError(f"Unconverted op type: {op.op_type}")
        op_code = ONNXType2OperatorType[op.op_type]
        if op_code == 3:  # 卷积
            if op.attribute[1].i != 1:  # "kernel_shape"假如是一维，那么i=1;二维时，则ins=[3，3]
                return 4  # 2D卷积
        if op_code == 23:  # resize放缩
            if op.attribute[-1].name == "nearest_mode":  # [-1]表示最后一个元素，最近插值模式
                return 97  # 最近插值法缩放，图像缩放时，根据相对位置等因素填充像素
        return op_code

    def print_model_info(self):  # 展示和验证模型信息
        # 加载模型注释
        if self.model.doc_string:
            print(self.model.doc_string)
        if self.model.ir_version:
            print(f"ONNX IR Version : {self.model.ir_version}")
        if self.model.model_version:
            print(f"ONNX Model Version : {self.model.model_version}")
        # 模型所需要的算子集
        print(f"ONNX Model OpSet : {self.model.opset_import[0].version}")
        assert isinstance(self.model, onnx.ModelProto), \
            f'onnx load failed, only ProtoBuffer object is expected here, while {type(self.model)} is loaded.'

    def quantize2int(self, tensor_float, tensor_id, bit_width=8) -> np.ndarray:
        tensor = self.graph.AllTensors[tensor_id]
        # 读取量化参数
        scale = tensor.Scale[0]
        zero_point = tensor.ZeroPoint[0]
        q_min = tensor.Q_min
        q_max = tensor.Q_max
        n = tensor_float.shape[0]

        tensor_int = np.zeros_like(tensor_float)  # 形状与tensor_float的零矩阵

        if isinstance(scale, np.ndarray):  # per-channel
            assert len(scale) == n
            for i in range(n):
                # tensor_int[i, ...]第一个维度为i，第二个维度开始的所有维度及其对应的数据。
                tensor_int[i, ...] = tensor_float[i, ...] / scale[i] + zero_point[i]
        else:  # per-tensor
            tensor_int = tensor_float / scale + zero_point

        tensor_int[tensor_int < q_min] = q_min  # 限幅，遍历tensor_int中的每一个元素,小于q_min的都是q_min
        tensor_int[tensor_int > q_max] = q_max  # 大于的都是q_max

        if bit_width == 8:
            return np.round(tensor_int).astype(np.int8)  # 四舍五入，并且转换为8位有符号整数
        elif bit_width == 32:
            return np.round(tensor_int).astype(np.int32)
        else:
            raise "bit_width must be 8 or 32"

    def load_ir_tensor_info(self, name, tensor_idx, op):
        ir_tensor = IRTensor()  # 张量类
        ir_tensor.Name = name

        # 赋值形状
        tensor_shape = []
        for t in self.model.graph.value_info:  # 在中间张量中寻找name
            if name == t.name:
                for dim in t.type.tensor_type.shape.dim:
                    tensor_shape.append(dim.dim_value)
                break
        else:
            for t in self.model.graph.initializer:  # 否则在initializer中找
                if name == t.name:
                    for dim in t.dims:
                        tensor_shape.append(dim)
                    break

        if len(tensor_shape) > 4:
            print(f"张量{name}拥有 四维度以上信息！{tensor_shape}")
        ir_tensor.Shape = Shape(tensor_shape)

        if self.quantization_config is None:
            ir_tensor.Id = name
        else:
            # 量化参数加载
            op_hash = self.quantization_config["configs"][op.name][name]['hash']
            ir_tensor.Id = op_hash
            state = self.quantization_config["configs"][op.name][name]['state']
            if state != 'FP32':
                dominator = str(self.quantization_config["configs"][op.name][name]['dominator'])

                ir_tensor.Scale = np.array([self.quantization_config["values"][dominator]['scale']])
                ir_tensor.ZeroPoint = np.array([self.quantization_config["values"][dominator]['zero_point']]).astype(
                    np.int8)
                ir_tensor.Q_min = np.array([self.quantization_config["configs"][op.name][name]['quant_min']])
                ir_tensor.Q_max = np.array([self.quantization_config["configs"][op.name][name]['quant_max']])

        # add tensor
        ir_tensor.idx = tensor_idx
        self.graph.add_tensor(ir_tensor)

    def load_all_tensor(self):
        inputs = [t.name for t in self.model.graph.input]
        outputs = [t.name for t in self.model.graph.output]
        # load all input tensor
        index = 0
        for op in self.fused_ops:
            for name in op.output:
                self.load_ir_tensor_info(name, index, op)
                # 加载网络输出
                if name in outputs:
                    self.graph.load_output_id(index)
                    self.graph.AllTensors[index].Type = TensorType.Output

                index += 1
            # 加载常数数值
            if op.op_type == "Constant":
                self.load_constant(op)
        # check tensors
        for op in self.fused_ops:
            for name in op.input:
                if not self.graph.check_tensor(name):
                    self.load_ir_tensor_info(name, index, op)
                    # 加载网络输出
                    if name in inputs:
                        self.graph.load_input_id(index)
                        self.graph.AllTensors[index].Type = TensorType.Input

                    index += 1
        print(f'已导入 {index} 个张量')

    def load_element(self, op, op_idx, mode):
        # 算子初始化
        elem_op = ElemWise()  # 元张量操作
        elem_op.Name = op.name
        elem_op.Mode = mode
        elem_op.TopOpId = op_idx

        # 输入
        in_tensors_name = op.input
        for name in in_tensors_name:
            if self.quantization_config is None:
                input_name = name
            else:
                input_name = self.quantization_config["configs"][op.name][name]['hash']

            in_tensor_id = self.graph.AllTensorIds.index(input_name)
            elem_op.load_input_id(in_tensor_id)
            elem_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
            self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输出
        out_tensors_name = op.output
        if self.quantization_config is None:
            out_name = out_tensors_name[0]
        else:
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        out_tensor_id = self.graph.AllTensorIds.index(out_name)
        elem_op.load_output_id(out_tensor_id)
        elem_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx  # 将指定张量的绑定到指定算子上

        # 不保留整个张量类信息，因为 其他信息不是经常用，所以 等用的时候直接通过函数查询，减少内存占用？
        self.graph.insert_op(elem_op, op_idx)

    def load_conv(self, op, op_idx, code):
        conv_op = Conv2d()

        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            fea_name = in_tensors_name[0]
            weight_name = in_tensors_name[1]
            out_name = out_tensors_name[0]
        else:
            fea_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            weight_name = self.quantization_config["configs"][op.name][in_tensors_name[1]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        fea_tensor_id = self.graph.AllTensorIds.index(fea_name)
        weight_tensor_id = self.graph.AllTensorIds.index(weight_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        conv_op.Name = op.name
        conv_op.TopOpId = op_idx

        # 加载输入输出ID
        conv_op.load_input_id(fea_tensor_id)
        conv_op.load_input_id(weight_tensor_id)
        conv_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[fea_tensor_id].ConsumerOp = op_idx
        self.graph.AllTensors[weight_tensor_id].Type = TensorType.Weight
        self.graph.AllTensors[weight_tensor_id].ConsumerOp = op_idx

        if op_idx == 0:
            conv_op.FirstLayer = True

        # 获取权重张量
        if self.graph.AllTensors[weight_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_data, weights_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError(f'无法找到{op.name}的权重信息！')
            if self.quantization_config is not None:
                weight_data = self.quantize2int(np_data, weight_tensor_id, bit_width=8)
            else:
                weight_data = np_data
            weight_data_int8_NHWC = np.transpose(weight_data, [0, 2, 3, 1])
            self.graph.AllTensors[weight_tensor_id].load_data(weight_data_int8_NHWC)  # weight_data_int8_NHWC

        c = self.graph.AllTensors[weight_tensor_id].Shape.N
        h = self.graph.AllTensors[out_tensor_id].Shape.H
        w = self.graph.AllTensors[out_tensor_id].Shape.W
        n = self.graph.AllTensors[out_tensor_id].Shape.N

        output_c = self.graph.AllTensors[out_tensor_id].Shape.C
        assert c == output_c, f'op {op.name}中,weight:{weight_tensor_id}, OutTensor:{out_tensor_id}'

        # 形状加载
        conv_op.InputShape.append(self.graph.AllTensors[fea_tensor_id].Shape)
        conv_op.OutputShape.append(Shape([n, c, h, w]))

        if code == OperatorType.CONV_2D:
            conv_op.Group = 1
        elif code == OperatorType.DEPTHWISE_CONV_2D:
            conv_op.Type = "DepthWiseConv2d"
            conv_op.Group = conv_op.InputShape[0].C

        conv_op.kerM_16 = True if conv_op.OutputShape[0].C % 16 == 0 else False

        for a in op.attribute:
            if a.name == "strides":
                conv_op.StrideH = a.ints[0]
                conv_op.StrideW = a.ints[1]
            elif a.name == "pads":
                assert a.ints[0] == a.ints[2]  # 上下相同
                assert a.ints[1] == a.ints[3]  # 左右相同
                conv_op.PadH = a.ints[0]
                conv_op.PadW = a.ints[1]
                if conv_op.PadH == 0 and conv_op.PadW == 0:
                    conv_op.Padding = 1  # Padding.Padding.VALID  VALID填充，确保卷积核保持在有区
                else:
                    conv_op.Padding = 0  # Padding.Padding.SAME  SAME填充，足够的填充以确保输出数据与输入数据具有相同的尺寸
            elif a.name == "kernel_shape":
                conv_op.KerH = a.ints[0]
                conv_op.KerW = a.ints[1]

        # 偏置项
        if len(in_tensors_name) == 3:
            if self.quantization_config is None:
                bias_name = in_tensors_name[2]
            else:
                bias_name = self.quantization_config["configs"][op.name][in_tensors_name[2]]['hash']
            bias_tensor_id = self.graph.AllTensorIds.index(bias_name)

            self.graph.AllTensors[bias_tensor_id].Type = TensorType.Bias
            self.graph.AllTensors[bias_tensor_id].ConsumerOp = op_idx
            conv_op.load_input_id(bias_tensor_id)

            if self.graph.AllTensors[bias_tensor_id].Data is None:
                for tensor in self.model.graph.initializer:
                    if tensor.name == in_tensors_name[2]:
                        np_data, dtype = get_np_data_from_attribute(tensor)
                        break
                else:
                    raise NotImplementedError(f'无法找到{op.name}的偏置信息！')
                if self.quantization_config is None:
                    bias_data = np_data
                else:
                    bias_data = self.quantize2int(np_data, weight_tensor_id, bit_width=32)
                self.graph.AllTensors[bias_tensor_id].load_data(bias_data)  # bias_data_int32

            conv_op.Bias = True
        else:
            conv_op.Bias = False

        self.graph.insert_op(conv_op, op_idx)
        # print(conv_op)

    def load_fullconnected(self, op, op_idx):
        fc_op = FullConnected()

        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            fea_name = in_tensors_name[0]
            weight_name = in_tensors_name[1]
            out_name = out_tensors_name[0]
        else:
            fea_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            weight_name = self.quantization_config["configs"][op.name][in_tensors_name[1]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        fea_tensor_id = self.graph.AllTensorIds.index(fea_name)
        weight_tensor_id = self.graph.AllTensorIds.index(weight_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        fc_op.Name = op.name
        fc_op.TopOpId = op_idx

        # 加载输入输出ID
        fc_op.load_input_id(fea_tensor_id)
        fc_op.load_input_id(weight_tensor_id)
        fc_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[fea_tensor_id].ConsumerOp = op_idx
        self.graph.AllTensors[weight_tensor_id].Type = TensorType.Weight
        self.graph.AllTensors[weight_tensor_id].ConsumerOp = op_idx

        if self.graph.AllTensors[weight_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_data, weights_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError(f'无法找到{op.name}的偏置信息！')
            if self.quantization_config is None:
                weight_data = np_data
            else:
                weight_data = self.quantize2int(np_data, weight_tensor_id, bit_width=8)
            self.graph.get_tensor(weight_tensor_id).load_data(weight_data)

        # 形状加载
        fc_op.InputShape.append(self.graph.AllTensors[fea_tensor_id].Shape)
        fc_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        weight_data = self.graph.AllTensors[weight_tensor_id].Data
        outc, input_c = weight_data.shape
        fc_op.OutputDim, output_c = len(weight_data.shape), self.graph.AllTensors[out_tensor_id].Shape.C
        assert outc == output_c
        assert fc_op.InputShape[0].C == input_c

        if len(in_tensors_name) == 3:
            if self.quantization_config is None:
                bias_name = in_tensors_name[2]
            else:
                bias_name = self.quantization_config["configs"][op.name][in_tensors_name[2]]['hash']
            bias_tensor_id = self.graph.AllTensorIds.index(bias_name)

            self.graph.AllTensors[bias_tensor_id].Type = TensorType.Bias
            self.graph.AllTensors[bias_tensor_id].ConsumerOp = op_idx
            fc_op.load_input_id(bias_tensor_id)

            if self.graph.AllTensors[bias_tensor_id].Data is None:
                for tensor in self.model.graph.initializer:
                    if tensor.name == in_tensors_name[2]:
                        np_data, dtype = get_np_data_from_attribute(tensor)
                        break
                else:
                    raise NotImplementedError(f'无法找到{op.name}的偏置信息！')
                if self.quantization_config is None:
                    bias_data = np_data
                else:
                    bias_data = self.quantize2int(np_data, weight_tensor_id, bit_width=32)
                self.graph.AllTensors[bias_tensor_id].load_data(bias_data)  # bias_data_int32

            fc_op.Bias = True
        else:
            fc_op.Bias = False

        self.graph.insert_op(fc_op, op_idx)

    def load_pool(self, op, op_idx, op_code):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            output_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            output_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(output_name)

        # 算子初始化
        pool_op = Pool()
        pool_op.Name = op.name
        pool_op.TopOpId = op_idx

        # 加载输入输出ID
        pool_op.load_input_id(in_tensor_id)
        pool_op.load_output_id(out_tensor_id)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        pool_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        pool_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        if op_code == OperatorType.MAX_POOL_2D:
            pool_op.Type = "MaxPool"
            pool_op.Mode = PoolMode.POOL_MAX
        elif op_code == OperatorType.AVERAGE_POOL_2D:
            pool_op.Type = "AvgPool"
            pool_op.Mode = PoolMode.POOL_AVG
        else:
            raise ValueError(f'Unsupported code of pool was identified: {op.name}')

        # pool parameters
        if pool_op.OutputShape[0].H == 1 and pool_op.OutputShape[0].W == 1 and len(op.attribute) == 0:  # GlobalPool
            pool_op.StrideH = 1
            pool_op.StrideW = 1
            pool_op.PadH = 0
            pool_op.PadW = 0
            pool_op.Padding = 1
            pool_op.KerH = pool_op.InputShape[0].H
            pool_op.KerW = pool_op.InputShape[0].W
            self.graph.insert_op(pool_op, op_idx)
            return

        for a in op.attribute:
            if a.name == "strides":
                pool_op.StrideH = a.ints[0]
                pool_op.StrideW = a.ints[1]
            elif a.name == "pads":
                assert a.ints[0] == a.ints[2]  # 上下相同
                assert a.ints[1] == a.ints[3]  # 左右相同
                pool_op.PadH = a.ints[0]
                pool_op.PadW = a.ints[1]
                if pool_op.PadH == 0 and pool_op.PadW == 0:
                    pool_op.Padding = 1  # Padding.Padding.VALID
                else:
                    pool_op.Padding = 0  # Padding.Padding.SAME
            elif a.name == "kernel_shape":
                pool_op.KerH = a.ints[0]
                pool_op.KerW = a.ints[1]

        self.graph.insert_op(pool_op, op_idx)

    def load_constant(self, op, op_idx=None):
        if op_idx is None:
            if op.attribute[0].name == 'value':
                out_tensors_name = op.output

                if self.quantization_config is None:
                    output_name = out_tensors_name[0]
                else:
                    output_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']
                out_tensor_id = self.graph.AllTensorIds.index(output_name)

                self.graph.AllTensors[out_tensor_id].Type = TensorType.Const

                val_type = op.attribute[0].type  # 确定常数类型

                if (val_type == onnx.AttributeProto.TENSOR and
                        not self.graph.AllTensors[out_tensor_id].Data):  # 是张量
                    np_data, dtype = get_np_data_from_attribute(op.attribute[0].t)
                    self.graph.AllTensors[out_tensor_id].load_data(np_data)
                    self.graph.AllTensors[out_tensor_id].Shape = Shape(np_data.shape.tolist())

                else:
                    raise NotImplementedError("有非张量类型常数")

        else:
            out_tensors_name = op.output

            if self.quantization_config is None:
                out_name = out_tensors_name[0]
            else:
                out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

            out_tensor_id = self.graph.AllTensorIds.index(out_name)

            const_op = Constant()
            const_op.Name = op.name
            const_op.TopOpId = op_idx

            const_op.load_output_id(out_tensor_id)
            self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx

            const_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

            self.graph.insert_op(const_op, op_idx)

    def load_activation(self, op, op_idx, op_code):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        act_op = Activation()
        act_op.Name = op.name
        act_op.TopOpId = op_idx

        # 加载输入输出ID
        act_op.load_input_id(in_tensor_id)
        act_op.load_output_id(out_tensor_id)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        act_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        act_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        assert act_op.InputShape[0].H == act_op.OutputShape[0].H

        if op_code == OperatorType.LEAKY_RELU:
            act_op.Type = 'LeakyRELU'
            act_op.Mode = ActivationMode.LEAKY_RELU
            act_op.Alpha = op.attribute[0].f
        elif op_code == OperatorType.PRELU:
            act_op.Type = 'PRELU'
            act_op.Mode = ActivationMode.PRELU
            act_op.Alpha = op.attribute[0].f
        elif op_code == OperatorType.LOGISTIC:
            act_op.Type = 'Sigmoid'
            act_op.Mode = ActivationMode.SIGMOID
        elif op_code == OperatorType.RELU:
            act_op.Type = 'RELU'
            act_op.Mode = ActivationMode.RELU

        self.graph.insert_op(act_op, op_idx)

    def load_transpose(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        transpose_op = Transpose()
        transpose_op.Name = op.name
        transpose_op.TopOpId = op_idx

        transpose_op.load_input_id(in_tensor_id)
        transpose_op.load_output_id(out_tensor_id)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        transpose_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        transpose_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        # ints的顺序与Numpy中的transpose用法一致
        transpose_op.OutDimOrder = np.array(op.attribute[0].ints)

        self.graph.insert_op(transpose_op, op_idx)

    def load_reshape(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            shape_name = in_tensors_name[1]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            shape_name = self.quantization_config["configs"][op.name][in_tensors_name[1]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        shape_tensor_id = self.graph.AllTensorIds.index(shape_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        reshape_op = Reshape()
        reshape_op.Name = op.name
        reshape_op.TopOpId = op_idx

        # 加载输入输出ID
        reshape_op.load_input_id(in_tensor_id)
        reshape_op.load_input_id(shape_tensor_id)
        reshape_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        self.graph.AllTensors[shape_tensor_id].Type = TensorType.Parameter
        self.graph.AllTensors[shape_tensor_id].ConsumerOp = op_idx

        if self.graph.AllTensors[shape_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_data, weights_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError(f'无法找到{op.name}的形状信息！')
            self.graph.AllTensors[shape_tensor_id].load_data(np_data)

        np_data = self.graph.AllTensors[shape_tensor_id].Data
        reshape_op.Target = np_data.tolist()

        # 输入输出形状
        reshape_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        reshape_op.OutputShape.append(Shape(reshape_op.Target))

        assert np_data[0] == reshape_op.OutputShape[0].N
        assert np_data[1] == reshape_op.OutputShape[0].C
        assert np_data[2] == reshape_op.OutputShape[0].H
        assert np_data[3] == reshape_op.OutputShape[0].W

        self.graph.insert_op(reshape_op, op_idx)

    def load_concat(self, op, op_idx):
        # 算子初始化
        concat_op = Concat()
        concat_op.Name = op.name
        concat_op.TopOpId = op_idx

        # 换成NHWC中对应的轴
        concat_op.Axis = op.attribute[0].i

        # 输入
        in_tensors_name = op.input
        data = None
        for name in in_tensors_name:
            if self.quantization_config is None:
                input_name = name
            else:
                input_name = self.quantization_config["configs"][op.name][name]['hash']

            in_tensor_id = self.graph.AllTensorIds.index(input_name)
            concat_op.load_input_id(in_tensor_id)
            concat_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
            self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

            if self.graph.AllTensors[in_tensor_id].Data is not None:
                if data is None:
                    data = self.graph.AllTensors[in_tensor_id].Data
                else:
                    data = np.concatenate((data,self.graph.AllTensors[in_tensor_id].Data), axis=concat_op.Axis)

                # 输出
        out_tensors_name = op.output
        if self.quantization_config is None:
            out_name = out_tensors_name[0]
        else:
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        out_tensor_id = self.graph.AllTensorIds.index(out_name)
        concat_op.load_output_id(out_tensor_id)
        concat_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx

        assert concat_op.FusedActFunc == 0

        if data is not None:
            self.graph.AllTensors[out_tensor_id].Data = data
            self.graph.AllTensors[out_tensor_id].Shape = Shape(data.shape.tolist())

        self.graph.insert_op(concat_op, op_idx)

    def load_resize(self, op, op_idx):
        in_tensors_name = op.input
        in_tensors_name = [item for item in in_tensors_name if item != ""]
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            resize_name = in_tensors_name[1]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            resize_name = self.quantization_config["configs"][op.name][in_tensors_name[1]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        resize_tensor_id = self.graph.AllTensorIds.index(resize_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        resize_op = Resize()
        resize_op.Name = op.name
        resize_op.TopOpId = op_idx

        resize_op.load_input_id(in_tensor_id)
        resize_op.load_input_id(resize_tensor_id)
        resize_op.load_output_id(out_tensor_id)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx
        self.graph.AllTensors[resize_tensor_id].Type = TensorType.Parameter
        self.graph.AllTensors[resize_tensor_id].ConsumerOp = op_idx

        resize_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        resize_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        # 插值方法 和 定位点
        for a in op.attribute:
            if a.name == "mode":
                resize_model = a.s.decode()  # decode() 方法主要用于将字节串（byte string，即 bytes 类型）解码成字符串。
                if resize_model == "nearest":
                    resize_op.Type = "resize_nearest"
                    resize_op.Mode = ResizeMode.RESIZE_NEAREST
                elif resize_model == "linear":
                    resize_op.Type = "resize_bilinear"
                    resize_op.Mode = ResizeMode.RESIZE_BILINEAR
                else:
                    raise NotImplementedError('无法识别的Resize模式')

            elif a.name == "coordinate_transformation_mode":
                coordinate_transformation_mode = a.s
                if coordinate_transformation_mode == "align_corners":
                    resize_op.AlignCorners = True
                if coordinate_transformation_mode == "half_pixel":
                    resize_op.HalfPixelCenters = True

        # 缩放因子
        if self.graph.AllTensors[resize_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_scales, scales_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError(f'无法找到"{op.name}"的设置信息"{in_tensors_name[1]}"！')
            self.graph.AllTensors[resize_tensor_id].load_data(np_scales)

        np_scales = self.graph.AllTensors[resize_tensor_id].Data
        y_scale, x_scale = np_scales[2], np_scales[3]  # 放大倍数
        assert resize_op.OutputShape[0].H / resize_op.InputShape[0].H == y_scale
        assert resize_op.OutputShape[0].W / resize_op.InputShape[0].W == x_scale
        assert y_scale == x_scale
        resize_op.ScaleFactor = y_scale

        self.graph.insert_op(resize_op, op_idx)

    def load_pad(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            loc_name = in_tensors_name[1]
            val_name = in_tensors_name[2]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            loc_name = self.quantization_config["configs"][op.name][in_tensors_name[1]]['hash']
            val_name = self.quantization_config["configs"][op.name][in_tensors_name[2]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        loc_tensor_id = self.graph.AllTensorIds.index(loc_name)
        val_tensor_id = self.graph.AllTensorIds.index(val_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        pad_op = Pad()
        pad_op.Name = op.name
        pad_op.TopOpId = op_idx

        # 加载输入输出ID
        pad_op.load_input_id(in_tensor_id)
        pad_op.load_input_id(loc_tensor_id)
        pad_op.load_input_id(val_tensor_id)

        pad_op.load_output_id(out_tensor_id)
        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        self.graph.AllTensors[loc_tensor_id].Type = TensorType.Parameter
        self.graph.AllTensors[loc_tensor_id].ConsumerOp = op_idx
        self.graph.AllTensors[val_tensor_id].Type = TensorType.Parameter
        self.graph.AllTensors[val_tensor_id].ConsumerOp = op_idx

        pad_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        pad_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        # Pad Shape
        if self.graph.AllTensors[loc_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_loc_data, loc_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError
            self.graph.AllTensors[loc_tensor_id].load_data(np_loc_data)
        pad_loc = list(self.graph.AllTensors[loc_tensor_id].Data)

        # pads should be a 1D tensor of shape [2 * input_rank]. pads format should be:
        # [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is the number of
        # pad values added at the beginning of axis i and xi_end, the number of pad
        # values added at the end of axis i.
        # [前一半为各个维度开始的填充层数, 后一半为各维度结束的填充层数]

        pad_op.pad_top, pad_op.pad_bottom = pad_loc[2], pad_loc[6]
        pad_op.pad_left, pad_op.pad_right = pad_loc[3], pad_loc[7]

        # Pad Value
        if self.graph.AllTensors[val_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[2]:
                    np_pads_data, pads_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError
            self.graph.AllTensors[val_tensor_id].load_data(np_pads_data)

        float_pad_val = float(self.graph.AllTensors[val_tensor_id].Data)

        # 量化
        if self.quantization_config is not None:
            pad_op.pad_val = (float_pad_val / self.graph.AllTensors[in_tensor_id].Scale[0]
                              + self.graph.AllTensors[in_tensor_id].ZeroPoint[0])
        pad_op.pad_val = float_pad_val

        self.graph.insert_op(pad_op, op_idx)

    def load_split(self, op, op_idx):
        # 算子初始化
        split_op = Split()
        split_op.Name = op.name
        split_op.TopOpId = op_idx

        # 输入
        in_tensors_name = op.input
        if self.quantization_config is None:
            input_name = in_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
        split_name = in_tensors_name[1]

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        split_op.load_input_id(in_tensor_id)
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx
        split_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)

        split_tensor_id = self.graph.AllTensorIds.index(split_name)
        split_op.load_input_id(split_tensor_id)
        self.graph.AllTensors[split_tensor_id].Type = TensorType.Parameter
        self.graph.AllTensors[split_tensor_id].ConsumerOp = op_idx

        # 输出
        out_tensors_name = op.output
        for name in out_tensors_name:
            if self.quantization_config is None:
                out_name = name
            else:
                out_name = self.quantization_config["configs"][op.name][name]['hash']
            out_tensor_id = self.graph.AllTensorIds.index(out_name)
            split_op.load_output_id(out_tensor_id)
            split_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)
            self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx

        # Split Shape
        if self.graph.AllTensors[split_tensor_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_split_data, dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError
            self.graph.AllTensors[split_tensor_id].load_data(np_split_data)
        split_op.split_shape = self.graph.AllTensors[split_tensor_id].Data.tolist()

        # Split Axis
        split_op.Axis = op.attribute[0].i
        # in_axis = self.graph.AllTensors[in_tensor_id].Shape.get_n_shape()[op.attribute[0].i]
        # # para_axis = sum(split_op.split_shape)
        # # assert in_axis == para_axis

        for i in range(len(split_op.OutTensors)):
            assert split_op.split_shape[i] == split_op.OutputShape[i].get_n_shape(1)[op.attribute[0].i]

        self.graph.insert_op(split_op, op_idx)

    def load_shape(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        shape_op = OpShape()
        shape_op.Name = op.name
        shape_op.TopOpId = op_idx

        # 加载输入输出ID
        shape_op.load_input_id(in_tensor_id)
        shape_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        shape_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        shape_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        if self.graph.AllTensors[in_tensor_id].Data is not None:
            self.graph.AllTensors[out_tensor_id].Data = shape_op.InputShape[0].get_shape_as_np()

        self.graph.insert_op(shape_op, op_idx)

    def load_floor(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        floor_op = Floor()
        floor_op.Name = op.name
        floor_op.TopOpId = op_idx

        # 加载输入输出ID
        floor_op.load_input_id(in_tensor_id)
        floor_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        floor_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        floor_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        if self.graph.AllTensors[in_tensor_id].Data is not None:
            self.graph.AllTensors[out_tensor_id].Data = np.floor(self.graph.AllTensors[in_tensor_id].Data)

        self.graph.insert_op(floor_op, op_idx)

    def load_cast(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        cast_op = Cast()
        cast_op.Name = op.name
        cast_op.TopOpId = op_idx

        # 加载输入输出ID
        cast_op.load_input_id(in_tensor_id)
        cast_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        cast_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        cast_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        cast_op.Target = onnx2np_dtype_mapping[op.attribute[0].i]

        if self.graph.AllTensors[in_tensor_id].Data is not None:
            self.graph.AllTensors[out_tensor_id].Data = \
                self.graph.AllTensors[in_tensor_id].Data.astype(cast_op.Target)

        self.graph.insert_op(cast_op, op_idx)

    def load_slice(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']
        start_name = in_tensors_name[1]
        end_name = in_tensors_name[2]
        axis_name = in_tensors_name[3]

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        start_id = self.graph.AllTensorIds.index(start_name)
        end_id = self.graph.AllTensorIds.index(end_name)
        axis_id = self.graph.AllTensorIds.index(axis_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        slice_op = Slice()
        slice_op.Name = op.name
        slice_op.TopOpId = op_idx

        # 加载输入输出ID
        slice_op.load_input_id(in_tensor_id)
        slice_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        slice_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        slice_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        slice_op.start = self.graph.AllTensors[start_id].Data
        slice_op.end = self.graph.AllTensors[end_id].Data
        slice_op.axis = self.graph.AllTensors[axis_id].Data

        if self.graph.AllTensors[in_tensor_id].Data is not None:
            axes = slice_op.axis
            starts = slice_op.start
            ends = slice_op.end
            data = self.graph.AllTensors[in_tensor_id].Data
            if axes is None:
                axes = range(len(starts))  # 如果没有指定轴，则对所有维度进行切片

            # 确保starts和ends的长度相匹配
            assert len(starts) == len(ends)

            # 创建一个空的切片元组，用于最终的切片操作
            slicing = [slice(None)] * data.ndim

            # 替换切片元组中指定轴上的切片对象
            for axis, start, end in zip(axes, starts, ends):
                slicing[axis] = slice(start, end)
            self.graph.AllTensors[out_tensor_id].Data = data[tuple(slicing)]

        self.graph.insert_op(slice_op, op_idx)

    def load_unsqueeze(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        if self.quantization_config is None:
            input_name = in_tensors_name[0]
            out_name = out_tensors_name[0]
        else:
            input_name = self.quantization_config["configs"][op.name][in_tensors_name[0]]['hash']
            out_name = self.quantization_config["configs"][op.name][out_tensors_name[0]]['hash']
        unsqueeze_name = in_tensors_name[1]

        in_tensor_id = self.graph.AllTensorIds.index(input_name)
        unsqueeze_id = self.graph.AllTensorIds.index(unsqueeze_name)
        out_tensor_id = self.graph.AllTensorIds.index(out_name)

        # 算子初始化
        unsqueeze_op = Unsqueeze()
        unsqueeze_op.Name = op.name
        unsqueeze_op.TopOpId = op_idx

        # 加载输入输出ID
        unsqueeze_op.load_input_id(in_tensor_id)
        unsqueeze_op.load_output_id(out_tensor_id)

        self.graph.AllTensors[out_tensor_id].OwnerOp = op_idx
        self.graph.AllTensors[in_tensor_id].ConsumerOp = op_idx

        # 输入输出形状
        unsqueeze_op.InputShape.append(self.graph.AllTensors[in_tensor_id].Shape)
        unsqueeze_op.OutputShape.append(self.graph.AllTensors[out_tensor_id].Shape)

        if self.graph.AllTensors[unsqueeze_id].Data is None:
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[1]:
                    np_data, dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError
            self.graph.AllTensors[unsqueeze_id].load_data(np_data)
        unsqueeze_op.axis = self.graph.AllTensors[unsqueeze_id].Data.tolist()

        if self.graph.AllTensors[in_tensor_id].Data is  not None:
            self.graph.AllTensors[out_tensor_id].Data = \
                np.expand_dims(self.graph.AllTensors[in_tensor_id].Data, axis=unsqueeze_op.axis[0])

        self.graph.insert_op(unsqueeze_op, op_idx)

    def parse_operator(self):
        unsupported = []
        for op_idx, op in enumerate(self.fused_ops):
            op_code = self._get_op_code(op)
            # print("op_code: ", op_code)
            if op_code not in SUPPORTED_OPs:
                print(f"Unsupported operator code:{op_code}")
                if op.op_type not in unsupported:
                    unsupported.append(op.op_type)

            elif op_code == OperatorType.CONV_2D:
                self.load_conv(op, op_idx, op_code)

            elif op_code == OperatorType.DEPTHWISE_CONV_2D:
                self.load_conv(op, op_idx, op_code)

            elif op_code == OperatorType.FULLY_CONNECTED:
                self.load_fullconnected(op, op_idx)

            elif op_code == OperatorType.ADD:
                self.load_element(op, op_idx, ElementWiseMode.ELW_ADD)

            elif op_code == OperatorType.MUL:
                self.load_element(op, op_idx, ElementWiseMode.ELW_MUL)

            elif op_code == OperatorType.POW:
                self.load_element(op, op_idx, ElementWiseMode.ELW_POW)

            elif op_code == OperatorType.MAX_POOL_2D:
                self.load_pool(op, op_idx, op_code)

            elif op_code == OperatorType.AVERAGE_POOL_2D:
                self.load_pool(op, op_idx, op_code)

            elif op_code == OperatorType.LOGISTIC:
                self.load_activation(op, op_idx, op_code)

            elif op_code == OperatorType.LEAKY_RELU:
                self.load_activation(op, op_idx, op_code)

            elif op_code == OperatorType.PRELU:
                self.load_activation(op, op_idx, op_code)

            elif op_code == OperatorType.TRANSPOSE:
                self.load_transpose(op, op_idx)

            elif op_code == OperatorType.PAD:
                self.load_pad(op, op_idx)

            elif op_code == OperatorType.RESIZE_NEAREST_NEIGHBOR:
                self.load_resize(op, op_idx)

            elif op_code == OperatorType.RESIZE_BILINEAR:
                self.load_resize(op, op_idx)

            elif op_code == OperatorType.RESHAPE:
                self.load_reshape(op, op_idx)

            elif op_code == OperatorType.CONCATENATION:
                self.load_concat(op, op_idx)

            elif op_code == OperatorType.SPLIT:
                self.load_split(op, op_idx)

            elif op_code == OperatorType.SHAPE:
                self.load_shape(op, op_idx)

            elif op_code == OperatorType.UNSQUEEZE:
                self.load_unsqueeze(op, op_idx)

            elif op_code == OperatorType.FLOOR:
                self.load_floor(op, op_idx)

            elif op_code == OperatorType.SLICE:
                self.load_slice(op, op_idx)

            elif op_code == OperatorType.CAST:
                self.load_cast(op, op_idx)

            elif op_code == OperatorType.CONSTANT:
                self.load_constant(op, op_idx)

            else:
                print(f"Unhandled operator:{op_code}")

            print('load Op:', op.name, op_idx)

            if op_idx + 1 != len(self.graph.AllOps):
                raise ValueError(f"{op.name}")

        if unsupported:
            raise NotImplementedError(f"\nUnsupported operator:{unsupported}\n总计: {len(unsupported)}种")

        # self.CompleteDAG()

    def CompleteDAG(self):
        dag = {}
        for tensor in self.graph.AllTensors:
            if (tensor.Type == TensorType.Intermediate or
                    tensor.Type == TensorType.Input or
                    tensor.Type == TensorType.Output or
                    tensor.Type == TensorType.Const or
                    tensor.Type == TensorType.Parameter):

                if tensor.Name not in dag:
                    dag[tensor.Name] = [None, []]
                if tensor.OwnerOp is not None:
                    dag[tensor.Name][0] = tensor.OwnerOp
                if tensor.ConsumerOp is not None:
                    dag[tensor.Name][1].append(tensor.ConsumerOp)

        def list_all_op(name):
            if dag[name][1]:
                for op_idx in dag[name][1]:
                    if (dag[name][0] is not None) and (op_idx not in self.graph.AllOps[dag[name][0]].PostTopOpId):
                        self.graph.AllOps[dag[name][0]].PostTopOpId.append(op_idx)
                    if dag[name][0] not in self.graph.AllOps[op_idx].PreTopOpId:
                        self.graph.AllOps[op_idx].PreTopOpId.append(dag[name][0])

                    for t_idx in self.graph.AllOps[op_idx].OutTensors:
                        next_name = self.graph.AllTensors[t_idx].Name
                        list_all_op(next_name)

        for in_idx in self.graph.NetInTensors:
            list_all_op(self.graph.AllTensors[in_idx].Name)


if __name__ == "__main__":
    m = ONNX2TopIR('assets/yolov3.onnx', 'assets/yolov3.json')  # 'assets/yolov3.json'
    m.load_all_tensor()
    m.parse_operator()
