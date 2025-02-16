# Parse Model
from frontend.ONNX_processor import *
# IR
from ir.conversion.top2npu.top2npu import *
from ir.conversion.ir_transform import *
from ir.conversion.optimization.op_fuse import *
from ir.conversion.optimization.memory_analysis import *
from ir.conversion.optimization.weight_reorder import *
from ir.conversion.codegen.codegen import *


if __name__ == '__main__':
    # config
    model_path = 'assets/yolov5s.onnx'

    # chip
    chip = "ada300"

    # 量化
    config_path = None  # 'assets/yolov3.json'
    quantization_mode = DataType.INT8

    # 推理结果输出
    verification = False
    # 默认input_path = 'verification/input'
    input_name = "xiaoxin.jpg"

    # 解析
    model_type = model_path.split(".")[-1]
    if model_type == "onnx":
        frontend = ONNX2TopIR(model_path=model_path,
                              config_path=config_path,
                              )  # config_path\
    else:
        raise NotImplementedError

    top_graph = frontend.graph

    # lowing
    if config_path is None:
        quantization_mode = None
    t2n = Top2Npu(chip=chip, mode=quantization_mode)
    npu_graph = t2n.transform(top_graph)

    # pass
    ir_transformer = IRTransformer()

    ir_transformer.add_transform_option(op_fuse_transform)
    ir_transformer.transform(npu_graph)

    ir_transformer.add_transform_option(memory_analysis_transform)
    ir_transformer.transform(npu_graph)

    # 后端，根据芯片实际参数优化
    if chip == "ada300":
        backend = Ada300(npu_graph)
    else:
        raise NotImplementedError

    npu_graph = backend.graph

    ir_transformer.add_transform_option(weight_mapping_transform)
    ir_transformer.transform(npu_graph)

    # ir_transformer.add_transform_option(codegen_transform)
    # ir_transformer.transform(npu_graph)

    if verification:
        if model_type == "onnx":
            from verification.onnx_runtime import *
            run = ONNXRUNER(
                model_path=model_path,
                file_name=input_name,
                result_path=f'{npu_graph.codegen_path}/{npu_graph.name}'
            )
        else:
            raise NotImplementedError

        run.verify(json_output=False)
