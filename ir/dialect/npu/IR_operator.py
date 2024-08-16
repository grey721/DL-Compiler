from ir.dialect.top.IR_operator import *
from ir.dialect.npu.IR_tensor import *
from copy import deepcopy
import math


class NpuConv2d(ConvBase):
    Type = "NpuConv2d"
    fmi_size = None
    fmo_size = None
    weight_size = None
    input_offset = None  # 偏移量
    weights_offset = None
    output_offset = None
    output_multiplier = None
    output_shift = None
    quantized_activation_min = None
    quantized_activation_max = None
    conv_type = None

    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()
        self.conv_type = 0


class NpuActivation(Activation):
    Type = "NpuActivation"

    def __init__(self):
        super().__init__()

    def shape_reverse_inference(self, address_list):
        return address_list


class NpuLeakyRelu(NpuActivation):
    Type = "NpuLeakyRelu"
    input_offset = None
    output_offset = None
    identity_output_multiplier = None
    alpha_output_multiplier = None
    identity_output_shift = None
    alpha_output_shift = None

    def __init__(self):
        super().__init__()
        self.lut_dict = None


class NpuLogistic(NpuActivation):
    Type = "NpuLogistic"
    input_offset = None
    input_multiplier = None
    output_offset = None
    output_multiplier = None

    quantized_activation_min = None
    quantized_activation_max = None

    def __init__(self):
        super().__init__()
        self.lut_dict = None


class NpuPool(Pool):
    Type = "NpuPool"
    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()


class NpuResize(Resize):
    Type = "NpuResize"
    ratio_w = None
    ratio_h = None

    def __init__(self):
        super().__init__()

class NpuConcat(Concat):
    Type =  "NpuConcat"
    def __init__(self):
        super().__init__()
        self.main_input_tensor_id = None


class NpuElemWise(ElemWise):
    Type = "NpuElemWise"
    left_shift = None
    input_multiplier = None
    input_shift = None
    input_offset = None
    input1_multiplier = None
    input1_shift = None
    input1_offset = None
    output_multiplier = None
    output_shift = None
    output_offset = None
    quantized_activation_min = None
    quantized_activation_max = None

    def __init__(self):
        super().__init__()

    def shape_reverse_inference(self, address_list):
        return address_list


class NpuSplit(Split):
    Device = "cpu"
    TimeStep = None
    Type = "NpuSplit"

    def __init__(self):
        super().__init__()


    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuReshape(Reshape):
    Type ="NpuReshape"
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuTranspose(Transpose):
    Type = "NpuTranspose"
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuMean(Mean):
    Type = "NpuMean"
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuSoftmax(NpuActivation):
    Type = "NpuSoftmax"
    Device = "cpu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuPad(Pad):
    Type = "NpuPad"
    Device = "npu"
    TimeStep = None

    def __init__(self):
        super().__init__()

    def set_time_step(self, time_step):
        self.TimeStep = time_step


class NpuFullConnected(FullConnected):
    Type = "NpuFullConnected"
    fmi_size = None
    fmo_size = None
    weight_size = None
    input_offset = None
    weights_offset = None
    output_offset = None
    output_multiplier = None
    output_shift = None
    quantized_activation_min = None
    quantized_activation_max = None
    conv_type = None

    def __init__(self):
        super().__init__()


class NpuOp(OpBase):
    Type = "NpuOp"
    # NPU OP MODE
    NpuOpMode = None
    NpuShortCutMode = None

    # PAD OP
    NpuOpPad = False
    NpuOpPadOp = None

    # CONV OP
    NpuOpConv = True
    NpuOpConvOp = None

    NpuOpFc = False
    NpuOpFcOp = None

    # POST OP
    NpuOpActivate = False
    NpuOpActivateOp = None

    NpuOpElemWise = False
    NpuOpElemWiseOp = None

    NpuOpPool = False
    NpuOpPoolOp = None
    NpuOpResize = False
    NpuOpResizeOp = None
    NpuOpConcat = False
    NpuOpConcatOp = None

    NpuOpReshape = False
    NpuOpReshapeOp = None

    # POST OP OUT
    NpuOpActiOut = False
    NpuOpPoolOut = False
    NpuOpResizeOut = False
    NpuOpElwOut = False

    NpuOpShortCutOut = False
    NpuOpShortCutOp = None

    NpuOpFmiSize = None
    NpuOpFmoSize = None
    NpuOpFmo1Size = None

    # time
    TimeStep = None
    Device = "npu"
    NpuOpId = None

    def __init__(self):
        super().__init__()
        self.Name = "NpuOp"
        self.NpuOpFlow = []
        self.fmi_tensor = []
        self.fmo_tensor = []
        self.weight_tensor = []
        self.concat_input_tensor = []
        self.short_cut_out_tensor = []  # 允许网络中的某些层直接跳过一些层与后续层相连?
        self.elemwise_input_tensor = []
        self.output_tensor_for_cancat = []
        self.concat_output = False
        self.concat_output_shape = None
        self.fmi_from_global_memory = False

    def add_fmi_tensor(self, ir_tensor_id):
        self.fmi_tensor.append(ir_tensor_id)

    def add_fmo_tensor(self, ir_tensor_id):
        self.fmo_tensor.append(ir_tensor_id)

    def add_weight_tensor(self, ir_tensor_id):
        self.weight_tensor.append(ir_tensor_id)

    def add_concat_input_tensor(self, ir_tensor_id):
        self.concat_input_tensor.append(ir_tensor_id)

    def add_short_cut_out_tensor(self, ir_tensor_id):
        self.short_cut_out_tensor.append(ir_tensor_id)

    def add_elemwise_input_tensor(self, ir_tensor_id):
        self.elemwise_input_tensor.append(ir_tensor_id)

    def add_output_tensor_for_cancat(self, ir_tensor_id):
        self.output_tensor_for_cancat.append(ir_tensor_id)

    def set_time_step(self, time_step):
        self.TimeStep = time_step

    def set_fmi_size(self):
        self.NpuOpFmiSize = self.NpuOpFlow[0].get_fmi_size()

    def set_fmo_size(self):
        self.NpuOpFmoSize = self.NpuOpFlow[-1].get_fmo_size()

    def set_fmo1_size(self):
        if self.NpuOpShortCutOut:
            self.NpuOpFmo1Size = self.NpuOpShortCutOp.get_fmo_size()

    def init_tensor(self):

        self.add_fmi_tensor(self.NpuOpFlow[0].InTensors[0])
        self.add_fmo_tensor(self.NpuOpFlow[-1].OutTensors[0])

        for idx in self.NpuOpFlow[0].InTensors[1:3]:
            self.add_weight_tensor(idx)

        if self.NpuOpShortCutOut:
            self.add_short_cut_out_tensor(self.NpuOpShortCutOp.OutTensors[0])

        if self.NpuOpConcat:
            npu_flow_tensor_record = []
            for npu_op in self.NpuOpFlow:
                if not isinstance(npu_op, NpuConcat):
                    npu_flow_tensor_record.extend(npu_op.InTensors)
                    npu_flow_tensor_record.extend(npu_op.OutTensors)
                else:
                    break

            for in_tensor in self.NpuOpConcatOp.InTensors:
                if in_tensor not in npu_flow_tensor_record:
                    self.add_concat_input_tensor(in_tensor)

        if self.NpuOpElemWise:
            npu_flow_tensor_record = []
            for npu_op in self.NpuOpFlow:
                if not isinstance(npu_op, NpuElemWise):
                    npu_flow_tensor_record.extend(npu_op.InTensors)
                    npu_flow_tensor_record.extend(npu_op.OutTensors)
                else:
                    break

            for in_tensor in self.NpuOpElemWiseOp.InTensors:
                if in_tensor not in npu_flow_tensor_record:
                    self.add_elemwise_input_tensor(in_tensor)
                    elew_input_len = len(self.NpuOpElemWiseOp.InTensors)
                    assert elew_input_len == 2
                    other_inp_index = self.NpuOpElemWiseOp.InTensors.index(in_tensor)
                    if other_inp_index == 0:
                        self.NpuOpElemWiseOp.input_index_for_tflite = 1
                    else:
                        self.NpuOpElemWiseOp.input_index_for_tflite = 0

    def set_short_cut_op(self):
        flow_len = len(self.NpuOpFlow)
        for i, p in enumerate(self.NpuOpFlow):
            if i < flow_len - 1:
                if len(p.PostTopOpId) > 1:
                    self.NpuOpShortCutOut = True
                    self.NpuOpShortCutOp = p
                    if isinstance(p, NpuActivation):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ACTIVATION_SHORT_CUT_OUTPUT

                    if isinstance(p, NpuConv2d):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.INPUT_SHORT_CUT_OUTPUT

                    if isinstance(p, NpuPool):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.POOL_SHORT_CUT_OUTPUT

                    if isinstance(p, NpuResize):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.RESIZE_SHORT_CUT_OUTPUT

                    if isinstance(p, NpuElemWise):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ELEW_SHORT_CUT_OUTPUT

    def init_all(self):
        self.set_short_cut_op()
        self.set_fmi_size()
        self.set_fmo_size()
        self.set_fmo1_size()
        self.init_tensor()


class VpuPostOpSetMode(object):
    NONE = 0

    ACTIVATION = 1
    ELW = 2
    RESIZE = 3
    POOL = 3

    ACTIVATION_ELW = 4
    ACTIVATION_RESIZE = 5
    ACTIVATION_POOL = 5

    ELW_ACTIVATION = 6
    ELW_RESIZE = 7
    ELW_POOL = 7

    POOL_ELW = 8
    POOL_ACTIVATION = 9

    RESIZE_ELW = 8
    RESIZE_ACTIVATION = 9

    ACTIVATION_ELW_POOL = 10
    ACTIVATION_ELW_RESIZE = 10

    ACTIVATION_POOL_ELW = 11
    ACTIVATION_RESIZE_ELW = 11

    ELW_ACTIVATION_POOL = 12
    ELW_ACTIVATION_RESIZE = 12

    ELW_POOL_ACTIVATION = 13
    ELW_RESIZE_ACTIVATION = 13

    POOL_ELW_ACTIVATION = 14
    RESIZE_ELW_ACTIVATION = 14

    POOL_ACTIVATION_ELW = 15
    RESIZE_ACTIVATION_ELW = 15


class VpuAdditionOutputSelection(object):
    INPUT_SHORT_CUT_OUTPUT = 0
    ACTIVATION_SHORT_CUT_OUTPUT = 1
    RESIZE_SHORT_CUT_OUTPUT = 2
    POOL_SHORT_CUT_OUTPUT = 2
    ELEW_SHORT_CUT_OUTPUT = 3
