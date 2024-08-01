from tool.constant_ONNX import *
from tool.IR_operator import *
from tool.IR_tensor import *
import numpy as np
import onnx
import json


from tool.my_tool import *


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
        # raw_data是字节数组，所以用np.frombuffer读取数据;如果没有dtype，就默认类型，根据字节数据赋值
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
    def __init__(self, model_path):
        # 初始化
        self.model = onnx.load(model_path)
        self.print_model_info()
        self.graph = GraphIR()  # 创建图IR对象
        # TODO what this
        self.fused_ops = []
        for n, op in enumerate(self.model.graph.node):
            if self._get_op_code(op) not in FUSIBLE_OPs:  # 不会存在一个不可融合算子，也是OVERLAPPED吗？？？？？
                self.fused_ops.append(op)
                continue
            print(op.name, 'not in FUSIBLE_OPs')
            # in_tensors_name = op.input
            # if len(in_tensors_name) == 1:
            # state = self.quantization_config["configs"][op.name][in_tensors_name[0]]['state']
            # if state == "OVERLAPPED":  # 表示量化参数可能与其他参数重叠，为了安全暂时不对他们优化 ???可能表示该算子已经在其他部分的处理流程中被考虑或融合过??????????????不确定
            #    continue
            self.fused_ops.append(op)
        print("Operators_len:", len(self.fused_ops))

    def _get_op_code(self, op):  # 获取算子类型的id
        if op.op_type not in ONNXType2OperatorType:
            raise NotImplementedError(f"Unconverted op type: {op.op_type}")
        op_code = ONNXType2OperatorType[op.op_type]
        if op_code == 3:   # 卷积
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

    def load_ir_tensor_info(self, name, tensor_idx):
        # assert self.model.graph.value_info[0], f'model.graph.value_info为空'
        ir_tensor = IRTensor()  # 张量类
        ir_tensor.Name = name
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
        dims = len(tensor_shape)
        if dims == 4:  # 不同情况下赋值赋值shape
            ir_tensor.Shape.N, ir_tensor.Shape.C, ir_tensor.Shape.H, ir_tensor.Shape.W = tensor_shape
        elif dims == 3:
            ir_tensor.Shape.C, ir_tensor.Shape.H, ir_tensor.Shape.W = tensor_shape
            ir_tensor.Shape.N = 1
        elif dims == 2:
            ir_tensor.Shape.N, ir_tensor.Shape.C = tensor_shape
            ir_tensor.Shape.W = 1
            ir_tensor.Shape.H = 1
        elif dims == 1:
            ir_tensor.Shape.C = tensor_shape[0]
            ir_tensor.Shape.N = 1
            ir_tensor.Shape.H = 1
            ir_tensor.Shape.W = 1
        elif dims != 0:
            ir_tensor.Shape = tensor_shape
            print(f"张量{name}拥有未知的维度信息！{tensor_shape}")
        # add tensor
        ir_tensor.Tensor_idx = tensor_idx
        self.graph.add_tensor(ir_tensor)

    def load_all_tensor(self):
        # load all input tensor
        index = 0
        for op in self.fused_ops:
            for name in op.output:
                # print(f'load {index} tensor: {name}')
                self.load_ir_tensor_info(name, index)
                index += 1
            # check tensors
        for op in self.fused_ops:
            for name in op.input:
                if not self.graph.check_tensor(name):
                    # print(f'load {index} tensor: {name}')
                    self.load_ir_tensor_info(name, index)
                    index += 1

        print(f'已导入 {index} 个张量')

        # 加载模型的输入和输出
        inputs = self.model.graph.input
        outputs = self.model.graph.output
        for t in inputs:
            self.graph.load_input_id(t.name)
            self.graph.get_tensor(t.name).Type = TensorType.Input
            self.graph.get_tensor(t.name).OwnerOp = -1
        for t in outputs:
            self.graph.load_output_id(t.name)
            self.graph.get_tensor(t.name).Type = TensorType.Output
            self.graph.get_tensor(t.name).ConsumerOp = [-2]

    def load_element(self, op, op_idx, mode):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in1_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        in2_tensor_id = self.graph.AllTensorNames[in_tensors_name[1]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        elem_op = ElemWise()  # 元张量操作
        elem_op.Name = op.name
        elem_op.Mode = mode
        elem_op.TopOpId = op_idx

        # 添加到输入输出列表
        elem_op.load_input_id(in1_tensor_id)
        elem_op.load_input_id(in2_tensor_id)
        elem_op.load_output_id(out_tensor_id)
        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx  # 将指定张量的绑定到指定算子上
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)
        self.graph.get_tensor(in_tensors_name[1]).ConsumerOp.append(op_idx)

        # 不保留整个张量类信息，因为 其他信息不是经常用，所以 等用的时候直接通过函数查询，减少内存占用？
        elem_op.InputShape = self.graph.AllTensors[in1_tensor_id].Shape
        elem_op.Input1Shape = self.graph.AllTensors[in2_tensor_id].Shape
        elem_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape

        self.graph.insert_op(elem_op, op_idx)
        # print(elem_op)

    def quantize2int(self):
        pass

    def load_conv(self, op, op_idx, code):
        conv_op = Conv2d()

        in_tensors_name = op.input
        out_tensors_name = op.output

        fea_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        weight_tensor_id = self.graph.AllTensorNames[in_tensors_name[1]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        conv_op.Name = op.name
        conv_op.TopOpId = op_idx
        
        # 加载输入输出ID
        conv_op.load_input_id(fea_tensor_id)
        conv_op.load_input_id(weight_tensor_id)
        conv_op.load_output_id(out_tensor_id)

        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)
        self.graph.get_tensor(in_tensors_name[1]).Type = TensorType.Weight
        self.graph.get_tensor(in_tensors_name[1]).ConsumerOp.append(op_idx)

        if op_idx == 0:
            conv_op.FirstLayer = True

        # 获取权重张量
        for tensor in self.model.graph.initializer:
            if tensor.name == in_tensors_name[1]:
                np_data, weights_dtype = get_np_data_from_attribute(tensor)
                break
        else:
            raise NotImplementedError(f'无法找到{op.name}的权重信息！')

        # weight_data_int8 = self.quantize2int(np_data, weight_tensor_id)
        # weight_data_int8_NHWC = np.transpose(weight_data_int8, [0, 2, 3, 1])
        self.graph.AllTensors[weight_tensor_id].load_data(np_data)  # weight_data_int8_NHWC

        conv_op.InputShape = self.graph.AllTensors[fea_tensor_id].Shape
        conv_op.OutputShape.C = self.graph.AllTensors[weight_tensor_id].Shape.N

        if code == OperatorType.CONV_2D:
            conv_op.Group = 1
        elif code == OperatorType.DEPTHWISE_CONV_2D:
            conv_op.Type = "DepthWiseConv2d"
            conv_op.Group = conv_op.InputShape.C

        conv_op.OutputShape.H = self.graph.AllTensors[out_tensor_id].Shape.H
        conv_op.OutputShape.W = self.graph.AllTensors[out_tensor_id].Shape.W
        output_c = self.graph.AllTensors[out_tensor_id].Shape.C

        assert conv_op.OutputShape.C == output_c, f'op {op.name}中,weight:{weight_tensor_id}, OutTensor:{out_tensor_id}'

        # TODO 正好CIM有什么用
        conv_op.kerM_16 = True if conv_op.OutputShape.C % 16 == 0 else False

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
            self.graph.get_tensor(in_tensors_name[2]).Type = TensorType.Bias
            self.graph.get_tensor(in_tensors_name[2]).ConsumerOp.append(op_idx)

            bias_tensor_id = self.graph.AllTensorNames[in_tensors_name[2]]
            conv_op.load_input_id(bias_tensor_id)
            for tensor in self.model.graph.initializer:
                if tensor.name == in_tensors_name[2]:
                    np_data, weights_dtype = get_np_data_from_attribute(tensor)
                    break
            else:
                raise NotImplementedError(f'无法找到{op.name}的偏置信息！')

            # bias_data_int32 = self.quantize2int(np_data, weight_tensor_id, bit_width=32)
            self.graph.AllTensors[bias_tensor_id].load_data(np_data)  # bias_data_int32

            conv_op.Bias = True
        else:
            conv_op.Bias = False

        self.graph.insert_op(conv_op, op_idx)
        # print(conv_op)

    def load_pool(self, op, op_idx, op_code):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        pool_op = Pool()
        pool_op.Name = op.name
        pool_op.TopOpId = op_idx

        # 加载输入输出ID
        pool_op.load_input_id(in_tensor_id)
        pool_op.load_output_id(out_tensor_id)
        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)

        # 输入输出形状
        pool_op.InputShape = self.graph.AllTensors[in_tensor_id].Shape
        pool_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape

        if op_code == OperatorType.MAX_POOL_2D:
            pool_op.Type = "MaxPool"
            pool_op.Mode = PoolMode.POOL_MAX
        elif op_code == OperatorType.AVERAGE_POOL_2D:
            pool_op.Type = "AvgPool"
            pool_op.Mode = PoolMode.POOL_AVG
        else:
            raise ValueError(f'Unsupported code of pool was identified: {op.name}')

        # pool parameters
        if pool_op.OutputShape.H == 1 and pool_op.OutputShape.W == 1 and len(op.attribute) == 0:  # GlobalPool
            pool_op.StrideH = 1
            pool_op.StrideW = 1
            pool_op.PadH = 0
            pool_op.PadW = 0
            pool_op.Padding = 1
            pool_op.KerH = pool_op.InputShape.H
            pool_op.KerW = pool_op.InputShape.W
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

    def load_constant(self, op, op_idx):
        if op.attribute[0].name == 'value':

            out_tensors_name = op.output

            out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

            # 算子初始化
            constant_op = Constant()
            constant_op.Name = op.name
            constant_op.TopOpId = op_idx

            constant_op.load_output_id(out_tensor_id)
            self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx

            constant_op.OutputShape = self.graph.get_tensor(out_tensors_name[0]).Shape

            constant_op.Mode = op.attribute[0].type  # 确定常数类型
            if constant_op.Mode == onnx.AttributeProto.TENSOR:  # 是张量
                np_data, dtype = get_np_data_from_attribute(op.attribute[0].t)
                self.graph.AllTensors[out_tensor_id].load_data(np_data)
                self.graph.get_tensor(out_tensors_name[0]).DataType = dtype

            else:
                raise NotImplementedError("有非张量类型常数")

            self.graph.insert_op(constant_op, op_idx)

    def load_relu(self, op, op_idx, op_code):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        relu_op = RELU()
        relu_op.Name = op.name
        relu_op.TopOpId = op_idx

        # 加载输入输出ID
        relu_op.load_input_id(in_tensor_id)
        relu_op.load_output_id(out_tensor_id)
        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)

        # 输入输出形状
        relu_op.InputShape = self.graph.AllTensors[in_tensor_id].Shape
        relu_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape

        assert relu_op.InputShape.H == relu_op.OutputShape.H

        if op_code == OperatorType.LEAKY_RELU:
            relu_op.Type = 'LeakyRELU'
            relu_op.Mode = RELUMode.LEAKY_RELU
            relu_op.Alpha = op.attribute[0].f
        elif op_code == OperatorType.PRELU:
            relu_op.Type = 'PRELU'
            relu_op.Mode = RELUMode.PRELU
            relu_op.Alpha = op.attribute[0].f

        self.graph.insert_op(relu_op, op_idx)

    def load_sigmoid(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        sigmoid_op = Sigmoid()
        sigmoid_op.Name = op.name
        sigmoid_op.TopOpId = op_idx

        # 加载输入输出ID
        sigmoid_op.load_input_id(in_tensor_id)
        sigmoid_op.load_output_id(out_tensor_id)
        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)

        # 输入输出形状
        sigmoid_op.InputShape = self.graph.AllTensors[in_tensor_id].Shape
        sigmoid_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape

        assert sigmoid_op.InputShape.H == sigmoid_op.OutputShape.H

        self.graph.insert_op(sigmoid_op, op_idx)

    def load_transpose(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        transpose_op = Transpose()
        transpose_op.Name = op.name
        transpose_op.TopOpId = op_idx

        transpose_op.load_input_id(in_tensor_id)
        transpose_op.load_output_id(out_tensor_id)
        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)

        transpose_op.InputShape = self.graph.AllTensors[in_tensor_id].Shape
        transpose_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape

        # 可能是因为，后面直接根据NHCW的顺序而调用Shape的属性值，因此转置这里就最好提前改变成NHCW，以供之后直接通过np.transpose转置？
        transpose_op.OutDimOrder = [op.attribute[0].ints[0],  # ints的顺序与Numpy中的transpose用法一致
                                    op.attribute[0].ints[3],
                                    op.attribute[0].ints[1],
                                    op.attribute[0].ints[2]]  # 调整成 NHCW

        self.graph.insert_op(transpose_op, op_idx)

    def load_reshape(self, op, op_idx):
        in_tensors_name = op.input
        out_tensors_name = op.output

        in_tensor_id = self.graph.AllTensorNames[in_tensors_name[0]]
        shape_tensor_id = self.graph.AllTensorNames[in_tensors_name[1]]
        out_tensor_id = self.graph.AllTensorNames[out_tensors_name[0]]

        # 算子初始化
        reshape_op = Reshape()
        reshape_op.Name = op.name
        reshape_op.TopOpId = op_idx

        # 加载输入输出ID
        reshape_op.load_input_id(in_tensor_id)
        reshape_op.load_input_id(shape_tensor_id)
        reshape_op.load_output_id(out_tensor_id)

        self.graph.get_tensor(out_tensors_name[0]).OwnerOp = op_idx
        self.graph.get_tensor(in_tensors_name[0]).ConsumerOp.append(op_idx)
        self.graph.get_tensor(in_tensors_name[1]).Type = TensorType.Shape
        self.graph.get_tensor(in_tensors_name[1]).ConsumerOp.append(op_idx)

        # 输入输出形状
        reshape_op.InputShape = self.graph.AllTensors[in_tensor_id].Shape
        reshape_op.OutputShape = self.graph.AllTensors[out_tensor_id].Shape  # 固定输出形状

        for tensor in self.model.graph.initializer:
            if tensor.name == in_tensors_name[1]:
                np_data, weights_dtype = get_np_data_from_attribute(tensor)
                break
        else:
             raise NotImplementedError(f'无法找到{op.name}的形状信息！')

        self.graph.AllTensors[shape_tensor_id].load_data(np_data)

        assert self.graph.AllTensors[shape_tensor_id].Data[0] == reshape_op.OutputShape.N
        assert self.graph.AllTensors[shape_tensor_id].Data[1] == reshape_op.OutputShape.C
        assert self.graph.AllTensors[shape_tensor_id].Data[2] == reshape_op.OutputShape.H
        assert self.graph.AllTensors[shape_tensor_id].Data[3] == reshape_op.OutputShape.W

        self.graph.insert_op(reshape_op, op_idx)

    # op_idx是op在AllOps和AllOpIds中的索引值，index
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
                continue

            elif op_code == OperatorType.ADD:
                self.load_element(op, op_idx, ElementWiseMode.ELW_ADD)

            elif op_code == OperatorType.MUL:  # NEW
                self.load_element(op, op_idx, ElementWiseMode.ELW_MUL)

            elif op_code == OperatorType.POW:
                self.load_element(op, op_idx, ElementWiseMode.ELW_POW)

            elif op_code == OperatorType.MAX_POOL_2D:
                self.load_pool(op, op_idx, op_code)

            elif op_code == OperatorType.AVERAGE_POOL_2D:
                self.load_pool(op, op_idx, op_code)

            elif op_code == OperatorType.SIGMOID:
                self.load_sigmoid(op, op_idx)

            elif op_code == OperatorType.LEAKY_RELU:
                self.load_relu(op, op_idx, op_code)

            elif op_code == OperatorType.PRELU:
                self.load_relu(op, op_idx, op_code)

            elif op_code == OperatorType.TRANSPOSE:
                self.load_transpose(op, op_idx)

            elif op_code == OperatorType.PAD:
                continue

            elif op_code == OperatorType.RESIZE_NEAREST_NEIGHBOR:
                continue  # TODO

            elif op_code == OperatorType.RESIZE_BILINEAR:
                continue  # TODO

            elif op_code == OperatorType.RESHAPE:
                self.load_reshape(op, op_idx)

            elif op_code == OperatorType.CONCATENATION:
                continue  # TODO

            elif op_code == OperatorType.SPLIT:
                continue  # TODO

            elif op_code == OperatorType.CONSTANT:
                self.load_constant(op, op_idx)

            else:
                print(f"Unhandled operator:{op_code}")

        if unsupported:
            raise NotImplementedError(f"\nUnsupported operator:{unsupported}\n总计: {len(unsupported)}个")

    def op2json(self, save_path):
        pass


if __name__ == "__main__":
    m = ONNX2TopIR('yolov3-tiny_128.onnx')
    # m = ONNX2TopIR('yolov5n.onnx')
    m.load_all_tensor()
    # toolkit = ONNXToolkit('yolov3-tiny_128.onnx')
    # toolkit.check_requirement_based_code(m._get_op_code, SUPPORTED_OPs)
    m.parse_operator()
