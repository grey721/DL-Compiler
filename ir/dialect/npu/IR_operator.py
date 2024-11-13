import math

from ir.dialect.top.IR_tensor import *
from ir.dialect.top.IR_operator import *


class NpuConv2d(Conv2d):
    fmi_size = None
    fmo_size = None
    weight_size = None

    conv_type = None

    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()
        self.conv_type = 0  # deconv:1 conv:0
        self.split_shape = []
        self.K_map_shape = None

    def kernel2col(self):
        self.K_map_shape = (self.KerH * self.KerW * self.KerC, self.KerM)


class NpuActivation(Activation):
    def __init__(self):
        super().__init__()


class NpuLeakyRelu(NpuActivation):
    def __init__(self):
        super().__init__()
        # self.lut_dict = None


class NpuLogistic(NpuActivation):
    def __init__(self):
        super().__init__()
        # self.lut_dict = None


class NpuPool(Pool):
    pad_top = None
    pad_bottom = None
    pad_left = None
    pad_right = None

    def __init__(self):
        super().__init__()


class NpuResize(Resize):
    def __init__(self):
        super().__init__()


class NpuConcat(Concat):
    def __init__(self):
        super().__init__()
        self.main_input_tensor_id = None


class NpuSplit(Split):
    def __init__(self):
        super().__init__()


class NpuElemWise(ElemWise):
    def __init__(self):
        super().__init__()


class NpuReshape(Reshape):
    def __init__(self):
        super().__init__()


class NpuTranspose(Transpose):
    def __init__(self):
        super().__init__()


class NpuSoftmax(NpuActivation):
    def __init__(self):
        super().__init__()


class NpuMean(Mean):
    def __init__(self):
        super().__init__()


class NpuPad(Pad):
    def __init__(self):
        super().__init__()


class NpuFullConnected(FullConnected):
    conv_type = None

    def __init__(self):
        super().__init__()


class NpuOp(OpBase):
    # NPU OP MODE
    NpuOpMode = None
    NpuShortCutMode = None

    # PAD OP
    NpuOpPad = False
    NpuOpPadOp = None

    # CONV OP
    NpuOpConv = False
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

    NpuOpTranspose = False
    NpuOpTransposeOp = None

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
        self.Type = "NpuOp"
        self.Name = "NpuOp"
        self.NpuOpFlow = []
        self.fmi_tensor = []
        self.fmo_tensor = []
        self.weight_tensor = []
        self.concat_input_tensor = []
        self.short_cut_out_tensor = []  # 允许网络中的某些层直接跳过一些层与后续层相连?
        self.elemwise_input_tensor = []
        self.output_tensor_for_concat = []
        self.concat_output = False
        self.concat_output_shape = None
        self.fmi_from_global_memory = False

    def __repr__(self):
        return f"""
    ######################## NpuOp.{self.NpuOpId} #############################
    Device: {self.Device}
    NpuOpMode: {self.NpuOpMode}
    NpuShortCutMode: {self.NpuShortCutMode}
    NpuOpPad: {self.NpuOpPad}
    NpuOpConv: {self.NpuOpConv}
    NpuOpFc: {self.NpuOpFc}
    NpuOpActivate: {self.NpuOpActivate}
    NpuOpElemWise: {self.NpuOpElemWise}
    NpuOpPool: {self.NpuOpPool}
    NpuOpResize: {self.NpuOpResize}
    NpuOpConcat: {self.NpuOpConcat}
    NpuOpReshape: {self.NpuOpReshape}
    NpuOpTranspose:{self.NpuOpTranspose}

    NpuOpActiOut: {self.NpuOpActiOut}
    NpuOpPoolOut: {self.NpuOpPoolOut}
    NpuOpResizeOut: {self.NpuOpResizeOut}
    NpuOpElwOut: {self.NpuOpElwOut}
    NpuOpShortCutOut: {self.NpuOpShortCutOut}
    ######################## NpuOp.{self.NpuOpId} #############################
    """

    def fuse_ops(self, ops):
        try:
            for op in ops:
                if isinstance(op, NpuConv2d):
                    if self.NpuOpConv:
                        return False
                elif isinstance(op, NpuConcat):
                    if self.NpuOpConcat:
                        return False
                elif isinstance(op, NpuElemWise):
                    if self.NpuOpElemWise:
                        return False
                elif isinstance(op, NpuActivation):
                    if self.NpuOpActivate:
                        return False
                elif isinstance(op, NpuPool):
                    if self.NpuOpPool:
                        return False
                elif isinstance(op, NpuResize):
                    if self.NpuOpResize:
                        return False
                elif isinstance(op, NpuTranspose):
                    if self.NpuOpTranspose:
                        return False
                elif isinstance(op, NpuReshape):
                    if self.NpuOpReshape:
                        return False
                elif isinstance(op, NpuPad):
                    if self.NpuOpPad:
                        return False

                else:
                    raise NotImplementedError(op.Type)

            self.NpuOpFlow.extend(ops)
            self.init_all()
            return True

        except TypeError:
            return self.fuse_ops([ops])

    def add_fmi_tensor(self, ir_tensor_id):
        self.fmi_tensor.append(ir_tensor_id)

    def add_fmo_tensor(self, ir_tensor_id):
        self.fmo_tensor.append(ir_tensor_id)

    def add_weight_tensor(self, ir_tensor_id):
        self.weight_tensor.append(ir_tensor_id)

    def add_concat_input_tensor(self, ir_tensor_id):
        if ir_tensor_id not in self.concat_input_tensor:
            self.concat_input_tensor.append(ir_tensor_id)

    def add_short_cut_out_tensor(self, ir_tensor_id):
        if ir_tensor_id not in self.short_cut_out_tensor:
            self.short_cut_out_tensor.append(ir_tensor_id)

    def add_elemwise_input_tensor(self, ir_tensor_id):
        if ir_tensor_id not in self.elemwise_input_tensor:
            self.elemwise_input_tensor.append(ir_tensor_id)

    def add_output_tensor_for_cancat(self, ir_tensor_id):
        if ir_tensor_id not in self.output_tensor_for_concat:
            self.output_tensor_for_concat.append(ir_tensor_id)

    def set_fmi_tensors(self, ir_tensor_ids):
        self.fmi_tensor = ir_tensor_ids[0:1]

    def set_fmo_tensors(self, ir_tensor_ids):
        self.fmo_tensor = ir_tensor_ids[0:1]

    def set_weight_tensors(self, ir_tensor_ids):
        self.weight_tensor = ir_tensor_ids

    def set_concat_input_tensors(self, ir_tensor_ids):
        self.concat_input_tensor = ir_tensor_ids

    def set_short_cut_out_tensors(self, ir_tensor_ids):
        self.short_cut_out_tensor = ir_tensor_ids

    def set_elemwise_input_tensors(self, ir_tensor_ids):
        self.elemwise_input_tensor = ir_tensor_ids

    def set_output_tensors_for_cancat(self, ir_tensor_ids):
        self.output_tensor_for_concat = ir_tensor_ids

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

        self.InTensors = self.NpuOpFlow[0].InTensors
        self.InputShape = self.NpuOpFlow[0].InputShape

        self.OutTensors = self.NpuOpFlow[-1].OutTensors
        self.OutputShape = self.NpuOpFlow[-1].OutputShape

        self.set_fmi_tensors(self.NpuOpFlow[0].InTensors)
        self.set_fmo_tensors(self.NpuOpFlow[-1].OutTensors)

        self.set_fmi_size()
        self.set_fmo_size()
        self.set_fmo1_size()

        if self.NpuOpConv:
            self.set_weight_tensors(self.NpuOpConvOp.InTensors[1:3])

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
        self.write_list = []
        self.read_list = []
        for i, p in enumerate(self.NpuOpFlow):
            if i < flow_len - 1:
                if len(p.PostTopOpId) > 1:  # 之后有多个op，存在捷径
                    self.NpuOpShortCutOut = True
                    self.NpuOpShortCutOp = p
                    if isinstance(p, NpuActivation):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ACTIVATION_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuConv2d):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.INPUT_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuPool):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.POOL_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuResize):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.RESIZE_SHORT_CUT_OUTPUT

                    elif isinstance(p, NpuElemWise):
                        vpu_add_out_mode = VpuAdditionOutputSelection()
                        self.NpuShortCutMode = vpu_add_out_mode.ELEW_SHORT_CUT_OUTPUT

                    # 保存需要存储的张量id
                    self.write = True
                    p.write = True
                    n_list = []
                    if len(p.OutTensors) == 1:
                        t = p.OutTensors[0]
                        n_list.append(t)
                    else:
                        flag = True
                        for t in p.OutTensors:
                            if flag and t in self.NpuOpFlow[i + 1].InTensors:
                                flag = False
                                continue
                            n_list.append(t)

                    p.write_list = n_list
                    self.write_list.extend(n_list)

            if 1 < i < flow_len:
                if len(p.InTensors) > 1:
                    # 保存需要读取的张量id
                    self.read = True
                    p.read = True
                    flag = True
                    n_list = []

                    for t in p.InTensors:
                        if flag and t in self.NpuOpFlow[i - 1].OutTensors:
                            flag = False
                            continue
                        if t not in p.read_list:
                            n_list.append(t)

                    p.read_list = n_list
                    self.read_list.extend(n_list)

    def gen_info_with_flow(self):
        self.Type = ""
        for op in self.NpuOpFlow:
            if isinstance(op, NpuConv2d):
                self.NpuOpConv = True
                self.NpuOpConvOp = op
            elif isinstance(op, NpuConcat):
                self.NpuOpConcat = True
                self.NpuOpConcatOp = op
            elif isinstance(op, NpuElemWise):
                self.NpuOpElemWise = True
                self.NpuOpElemWiseOp = op
            elif isinstance(op, NpuActivation):
                self.NpuOpActivate = True
                self.NpuOpActivateOp = op
            elif isinstance(op, NpuPool):
                self.NpuOpPool = True
                self.NpuOpPoolOp = op
            elif isinstance(op, NpuResize):
                self.NpuOpResize = True
                self.NpuOpResizeOp = op
            elif isinstance(op, NpuTranspose):
                self.NpuOpTranspose = True
                self.NpuOpTransposeOp = op
            elif isinstance(op, NpuReshape):
                self.NpuOpReshape = True
                self.NpuOpReshapeOp = op
            elif isinstance(op, NpuPad):
                self.NpuOpPad = True
                self.NpuOpPadOp = op

            else:
                print(self.NpuOpFlow)
                raise NotImplementedError(op.Type)

            self.Type += op.Type[0]

    def init_all(self):
        self.gen_info_with_flow()
        self.set_short_cut_op()
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
