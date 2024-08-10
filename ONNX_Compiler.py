from frontend.onnx_processor import *
from frontend.tool.utils import *


if __name__ == '__main__':
    # config
    model_path = 'assets/yolov3.onnx'
    config_path = 'assets/yolov3.json'

    model_processor = ONNX2TopIR(model_path, config_path)
    model_processor.load_all_tensor()
    model_processor.parse_operator()

    top_graph = model_processor.graph
    top_graph_op_list = get_top_graph_op_list(top_graph)
else:
    t2n = top2npu()
    npu_graph = t2n.transform(top_graph)

    ir_transformer = IRTransformer()
    ir_transformer.add_transform_option(op_fuse_transform)
    ir_transformer.transform(npu_graph)

    # profile = False
    # if profile:
