from frontend.ONNX_processor import *
from frontend.tool.utils import *
from ir.conversion.top2npu.top2npu_pass import *
from ir.conversion.optimize.ir_transform import *
from ir.conversion.optimize.op_fuse import *
from ir.conversion.optimize.subnet import *


if __name__ == '__main__':
    # config
    model_path = 'assets/yolov3.onnx'
    config_path = 'assets/yolov3.json'

    # 解析
    model_processor = ONNX2TopIR(model_path, config_path)
    model_processor.load_all_tensor()
    model_processor.parse_operator()
    top_graph = model_processor.graph

    # lowing
    t2n = Top2Npu()
    npu_graph = t2n.transform(top_graph)

    # pass
    ir_transformer = IRTransformer()

    ir_transformer.add_transform_option(op_fuse_transform)
    ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(subnet_transform)
    ir_transformer.transform(npu_graph)

else:

    ir_transformer.add_transform_option(layer_group_transform)
    ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(weight_mapping_transform)
    ir_transformer.transform(npu_graph)

    # ir_transformer.add_transform_option(memory_assign_transform)
    # ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(codegen_transform)
    ir_transformer.transform(npu_graph)

    import pickle
    with open("output/top_graph.pkl", "wb") as f:
        pickle.dump(top_graph, f)

    # profile = False
    # if profile: