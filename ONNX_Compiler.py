from frontend.onnx_processor import *
from frontend.tool.utils import *
from ir.conversion.top2npu.top2npu_pass import *


if __name__ == '__main__':
    # config
    model_path = 'assets/yolov3.onnx'
    config_path = 'assets/yolov3.json'

    model_processor = ONNX2TopIR(model_path, config_path)
    model_processor.load_all_tensor()
    model_processor.parse_operator()

    top_graph = model_processor.graph
    top_graph_op_list = get_top_graph_op_list(top_graph)

    t2n = Top2Npu()


else:

    npu_graph = t2n.transform(top_graph)

    ir_transformer = IRTransformer()
    ir_transformer.add_transform_option(op_fuse_transform)
    ir_transformer.transform(npu_graph)

    import pickle
    with open("output/top_graph.pkl", "wb") as f:
        pickle.dump(top_graph, f)

    # profile = False
    # if profile:
