from ir.conversion.top2npu.ada200.operator_lowing.base import OpTransformRule, _register_op_transformation_rule
from ir.dialect.npu.IR_operator import *
from ir.conversion.top2npu.top_lowing import *
from python.vpu import *
from python.memory import *
from python.util import *


@_register_op_transformation_rule(OpTransformRule.LOGISTIC_LOWERING)
def _lowering(net, mode):
    for op in net.AllOps:
        if isinstance(op, Activation):
            if op.Mode != ActivationMode.SIGMOID:
                continue
            if mode == "int8":
                NpuOp = _lowering_int8(op, net)
            if mode == "fp32":
                NpuOp = _lowering_fp32(op, net)

            op_id = net.get_op_idx(op)
            net.delete_op(op_id)
            net.insert_op(NpuOp, op_id)


def _lowering_int8(op,net):
   
    npu_logistic = NpuActivation()
    npu_logistic.__dict__.update(op.__dict__)
    npu_logistic.Name = "NpuActivation_Logistic"

    input_scale = op.GetQuantInputScaleNumpy(net)
    output_scale = op.GetQuantOutputScaleNumpy(net)
    
    input_zero_point = op.get_input_zero_point_numpy(net)
    output_zero_point = op.get_output_zero_point_numpy(net)
    
    npu_logistic.input_offset = input_zero_point
    npu_logistic.output_offset = output_zero_point

    npu_logistic.output_multiplier = QuantizeMultiplier(input_scale, output_scale)
    #todo test logistic in test project 
    raise NotImplementedError



def _lowering_fp32(op,net):
    raise NotImplementedError



def sigmoid_int8(x, y, input_shape, shape_len, input_offset,  input_scale):
    param = SIGMOID_PARAM(input_offset, input_scale)
    base_param = BASE_PARAM()
    base_param.sigmoid_param = param
    vpu_type = 3
    vpu_base_param = VPU_BASE_PARAM(base_param, vpu_type) 
    py_values = [vpu_base_param]
    vpu_base_param_list = (VPU_BASE_PARAM * len(py_values))(*py_values)
    vpu_mode = 0
    vpu_base_param_list_len = 1
    vpu_parm = VPU_PARAM(vpu_base_param_list, vpu_base_param_list_len, vpu_mode)

    x = ThreeD2OneD(x).tolist()
    buffer_size = len(x)
    buffer_addr = 0
    memory_type = 0
    data_type = 0 # mean uint8
    input_data = (c_int8 * buffer_size)(*x)
    base_data = BASE_DATA()
    base_data.i8_data = input_data
    input = MEMORY_ADDRES_MAP(buffer_addr, buffer_size,
                        memory_type, data_type, base_data)

    buffer_size = len(y)
    buffer_addr = 0
    memory_type = 0
    data_type = 0 # mean uint8
    output_data = (c_int8 * len(y))(*y)
    base_data = BASE_DATA()
    base_data.i8_data = output_data

    output = MEMORY_ADDRES_MAP(buffer_addr, buffer_size, 
                            memory_type, data_type, base_data)

    vpu_interface(vpu_parm, input, output, 1)
    result = output.get_data()
    assert (shape_len == 3), 'shape error'
    h,w,c = input_shape
    result = OneD2ThreeD(result, h, w, c)
    
    return result

def cmodel(x,input_shape):
    y = np.zeros([input_shape[0] * input_shape[1] * input_shape[2]], dtype=np.int8)
    input_shape = np.array(input_shape, dtype=np.int32)
    shape_len = np.int32(len(input_shape))
    input_offset = para['input_offset'][0]
    input_scale = para['input_scale'][0]
    y = sigmoid_int8(x, y, input_shape, shape_len, input_offset,  input_scale)
    return y 

