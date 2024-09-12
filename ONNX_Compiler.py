from frontend.ONNX_processor import *
from ir.conversion.top2npu.top2npu_pass import *
from ir.conversion.ir_transform import *
from ir.conversion.optimization.op_fuse import *
from ir.conversion.optimization.subnet import *
from ir.conversion.optimization.layer_group import *
from ir.conversion.optimization.weight_reorder import *
from ir.conversion.codegen.codegen import *


if __name__ == '__main__':
    # config
    model_path = 'assets/yolov5s.onnx'
    config_path = None  # 'assets/yolov3.json'
    quantization_mode = None  # mode="int8"

    # 解析
    model_processor = ONNX2TopIR(model_path=model_path,
                                 config_path=config_path,
                                 )  # config_path
    model_processor.load_all_tensor()
    model_processor.parse_operator()
    top_graph = model_processor.graph

    # lowing
    t2n = Top2Npu(mode=quantization_mode)
    npu_graph = t2n.transform(top_graph)

    # pass
    ir_transformer = IRTransformer()

    ir_transformer.add_transform_option(op_fuse_transform)
    ir_transformer.transform(npu_graph)

    print('all ops')
    print(npu_graph.AllOps)

    # ir_transformer.add_transform_option(subnet_transform)
    # ir_transformer.transform(npu_graph)

    # ir_transformer.add_transform_option(layer_group_transform)
    # ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(weight_mapping_transform)
    ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(codegen_transform)
    ir_transformer.transform(npu_graph)


