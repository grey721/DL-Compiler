import numpy as np
import onnx


def for_seek(ob, attr: str, aim: str, out_f=0, fail_f=0):
    for i in ob:
        if getattr(i, attr) == aim:
            if out_f:
                print('========================seek result============================')
                print(i)
                print('===============================================================')
            return i
    else:
        if fail_f:
            print('======================== Seek: ', aim, 'is not here!', '============================')
        return False


class ONNXToolkit:
    def __init__(self, path):
        self.model = onnx.load(path)
        self.requirement = []
        self.print_model_info()

    # 展示和验证模型信息
    def print_model_info(self):
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
        self.get_op_requirement()

    # 获得该模型需要的算子
    def get_op_requirement(self):
        if not self.requirement:
            for op in self.model.graph.node:
                if op.op_type not in self.requirement:
                    self.requirement.append(op.op_type)
        print(f'该模型总计需要支持{len(self.requirement)}种算子，分别为:')
        print(self.requirement)

    # 检查已有算子是否支持该模型
    def check_op_requirement(self, supported_list) -> list:
        unsupported_list = []
        for op in self.requirement:
            if op not in supported_list:
                unsupported_list.append(op)
        if unsupported_list:
            print(f'该模型还需要支持{len(unsupported_list)}种算子，分别为:')
            print(unsupported_list)
        else:
            print('当前算子组已支持该模型')
        return unsupported_list

    # 输入获取代码函数，已支持算子列表、枚举
    def check_requirement_based_code(self, code_func, supported_list) -> list:
        unsupported = []
        for op in self.model.graph.node:
            code = code_func(op)
            if code not in supported_list:
                if op.op_type not in unsupported:
                    unsupported.append(op.op_type)
        if unsupported:
            print(f'该模型还需要支持{len(unsupported)}种算子，分别为:')
            print(unsupported)
        else:
            print('当前算子组已支持该模型')
        return unsupported

    # 获取该该张量的信息
    def get_tensor_with_name(self, name):
        aim = for_seek(self.model.graph.value_info, 'name', name, 0, 0)
        if aim:
            print(aim)
            return aim
        else:
            aim = for_seek(self.model.graph.initializer, 'name', name, 0, 0)
            if aim:
                print(aim)
                return aim
            else:
                print('no this tensor info!')
                return False


if __name__ == '__main__':
    # toolkit1 = ONNXToolkit('assets/yolov5s.onnx')
    # for_seek(toolkit1.model.graph.node, 'op_type', 'Split',1,1)
    # print(sys.getsizeof(my_list))\
    # 创建一个空的切片元组，用于最终的切片操作
    simu_out = 0.9882352941176471
    compile_out = 0.9882352948188782

    print(f"模拟器输出：{simu_out:.16f}")
    print(f"将模拟器输出转化为FP64:{np.float64(simu_out):.16f}")
    print(f"将模拟器输出转化为FP32:{np.float32(simu_out):.16f}")
    print(f"      编译器推理输出值:{compile_out:.16f}")
