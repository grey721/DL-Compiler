import onnx
import onnx.shape_inference

# 加载 ONNX 模型
# moudel_path = 'assets/yolov5s.onnx'
# moudel_path = 'assets/yolov3.onnx'
moudel_path = 'assets/yolov5s_1.onnx'

onnx_model = onnx.load(moudel_path)

# 打印模型名字
# print(onnx_model.graph.name)

# 列出 onnx_model 对象的所有属性和方法
# print(dir(onnx_model))

# 显示 onnx_model 的文档
# help(onnx_model)

# 输出节点信息
# for node in onnx_model.graph.node:
#     print(f"Node Name: {node.name}")
#     print(f"Operation Type: {node.op_type}")
#     print(f"Inputs: {node.input}")
#     print(f"Outputs: {node.output}")
#     print(f"Attributes: {node.attribute}")
#     print("-----------")

# 获取模型的图结构
# onnx_model.graph

# 遍历模型中的节点
# for node in onnx_model.graph.node:
#     print(f"Operation: {node.op_type}, Name: {node.name}")
#     for attribute in node.attribute:
#         print(f"Attribute Name: {attribute.name}")
#         print(f"Attribute Type: {attribute.type}")
#         if attribute.type == onnx.AttributeProto.INT:
#             print(f"Attribute Value (INT): {attribute.i}")
#         elif attribute.type == onnx.AttributeProto.FLOAT:
#             print(f"Attribute Value (FLOAT): {attribute.f}")
#         elif attribute.type == onnx.AttributeProto.STRING:
#             print(f"Attribute Value (STRING): {attribute.s.decode('utf-8')}")
#         elif attribute.type == onnx.AttributeProto.INTS:
#             print(f"Attribute Value (INTS): {attribute.ints}")
#         elif attribute.type == onnx.AttributeProto.FLOATS:
#             print(f"Attribute Value (FLOATS): {attribute.floats}")
#         elif attribute.type == onnx.AttributeProto.TENSOR:
#             print(f"Attribute Value (TENSOR): {attribute.t}")
#         elif attribute.type == onnx.AttributeProto.GRAPH:
#             print(f"Attribute Value (GRAPH): {attribute.g}")
#         print()




# 第0个op
# print (onnx_model.graph.node)
# print (onnx_model.graph.node[0])
# print (onnx_model.graph.node[1])
# print (onnx_model.graph.node[2])

# 输入参数
# input = onnx_model.graph.node[0].input
inputs = [t.name for t in onnx_model.graph.input]
outputs = [t.name for t in onnx_model.graph.output]
# print (inputs)
# print('###################################################################')
# print (outputs)
# print (input)


print(onnx_model.graph.value_info)
# print(onnx_model.graph.value_info[1])
print('########################################')
# print(onnx_model.graph.node[0])

# 变量的名称，类型，维度
tensor_shape = []
# 适用于yolov3
if moudel_path == 'assets/yolov3.onnx':
    for t in onnx_model.graph.value_info:  # 在中间张量中寻找name
        if t.name == '28':
            for dim in t.type.tensor_type.shape.dim:
                tensor_shape.append(dim.dim_value)
            break
# 适用于yolov5s
elif moudel_path == 'assets/yolov5s.onnx':
    for node in onnx_model.graph.node:
        pass
print(tensor_shape)
# for i in range(3):
#     print(value_info[i])

# print (value_info)
# print(onnx_model.graph)

# 维度
# dim = onnx_model.graph.value_info[1].type.tensor_type.shape.dim
# print (dim)
# print (dim[0].dim_value)

# 模型所需的算子集（OpSet）版本信息
# print(f"ONNX Model OpSet : {onnx_model.opset_import[0].version}")
# print(onnx_model.opset_import)



# 迷惑点#######################################################################################################################
#
# 1. frontend\ONNX_processor.py 第50行，if self._get_op_code(op) not in FUSIBLE_OPs返回的不是个数字吗？
# 2. frontend\ONNX_processor.py 第52行 continue的位置是不是放错了
# 3. 第121行，yolov5s没有value_info这个属性
# 
# ############################################################################################################################

# yolov5s有哪些op#############################################################################################################
# Conv, Sigmoid, Mul, Concat, Add, MaxPool, Floor, Shape, Unsqueeze, Slice, Cast, Resize, Transpose, Reshape, split, Pow
# ############################################################################################################################

# yolov5s的前三个op############################################################################################################
# input: "images"
# input: "model.0.conv.weight"
# input: "model.0.conv.bias"
# output: "/model.0/conv/Conv_output_0"
# name: "/model.0/conv/Conv"
# op_type: "Conv"
# attribute {
#   name: "dilations"
#   type: INTS
#   ints: 1
#   ints: 1
# }
# attribute {
#   name: "group"
#   type: INT
#   i: 1
# }
# attribute {
#   name: "kernel_shape"
#   type: INTS
#   ints: 6
#   ints: 6
# }
# attribute {
#   name: "pads"
#   type: INTS
#   ints: 2
#   ints: 2
#   ints: 2
#   ints: 2
# }
# attribute {
#   name: "strides"
#   type: INTS
#   ints: 2
#   ints: 2
# }

# input: "/model.0/conv/Conv_output_0"
# output: "/model.0/act/Sigmoid_output_0"
# name: "/model.0/act/Sigmoid"
# op_type: "Sigmoid"

# input: "/model.0/conv/Conv_output_0"
# input: "/model.0/act/Sigmoid_output_0"
# output: "/model.0/act/Mul_output_0"
# name: "/model.0/act/Mul"
# op_type: "Mul"
# ############################################################################################################################

# yolov3的前三个op#############################################################################################################
# input: "images"
# input: "model.0.conv.weight"
# input: "model.0.conv.bias"
# output: "28"
# name: "Conv_0"
# op_type: "Conv"
# attribute {
#   name: "dilations"
#   type: INTS
#   ints: 1
#   ints: 1
# }
# attribute {
#   name: "group"
#   type: INT
#   i: 1
# }
# attribute {
#   name: "kernel_shape"
#   type: INTS
#   ints: 3
#   ints: 3
# }
# attribute {
#   name: "pads"
#   type: INTS
#   ints: 1
#   ints: 1
#   ints: 1
#   ints: 1
# }
# attribute {
#   name: "strides"
#   type: INTS
#   ints: 1
#   ints: 1
# }

# input: "28"
# output: "29"
# name: "LeakyRelu_1"
# op_type: "LeakyRelu"
# attribute {
#   name: "alpha"
#   type: FLOAT
#   f: 0.1
# }

# input: "29"
# output: "30"
# name: "MaxPool_2"
# op_type: "MaxPool"
# attribute {
#   name: "ceil_mode"
#   type: INT
#   i: 0
# }
# attribute {
#   name: "kernel_shape"
#   type: INTS
#   ints: 2
#   ints: 2
# }
# attribute {
#   name: "pads"
#   type: INTS
#   ints: 0
#   ints: 0
#   ints: 0
#   ints: 0
# }
# attribute {
#   name: "strides"
#   type: INTS
#   ints: 2
#   ints: 2
# }
# ############################################################################################################################

# yolov3前三个info#############################################################################################################
# name: ""
# type {
#   tensor_type {
#     elem_type: 1
#   }
# }

# name: "28"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_value: 1
#       }
#       dim {
#         dim_value: 16
#       }
#       dim {
#         dim_value: 128
#       }
#       dim {
#         dim_value: 128
#       }
#     }
#   }
# }

# name: "30"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_value: 1
#       }
#       dim {
#         dim_value: 16
#       }
#       dim {
#         dim_value: 64
#       }
#       dim {
#         dim_value: 64
#       }
#     }
#   }
# }
# ############################################################################################################################

# yolov3前三个info#############################################################################################################

