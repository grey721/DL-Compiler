from ir.graph.Graph_IR import *
from ir.dialect.npu.IR_operator import *
from ir.conversion.optimization.ir_transform import _register_ir_transformation_rule
import multiprocessing
from enum import Enum
import math


def make_list_array(shape):
    lenth = len(shape)
    assert (lenth == 2 or lenth == 3 or lenth == 1)
    if lenth == 1:
        array = []
        for i in range(shape[0]):
            array.append([])
        return array
    if lenth == 2:
        array = []
        for i in range(shape[0]):
            tem = []
            for j in range(shape[1]):
                tem.append([])
            array.append(tem)
        return array
    if lenth == 3:
        array = []
        for i in range(shape[0]):
            tem1 = []
            for j in range(shape[1]):
                tem2 = []
                for z in range(shape[2]):
                    tem2.append([])
                tem1.append(tem2)
            array.append(tem1)
        return array


def intToBin(number, index, feature=True):
    '''index为该数据位宽,number为待转换数据,
    feature为True则进行十进制转二进制,为False则进行二进制转十进制。
    '''
    if (feature == True):  # 十进制转换为二进制
        if (number >= 0):
            b = bin(number)
            b = '0' * (index + 2 - len(b)) + b
        else:
            b = 2 ** (index) + number
            b = bin(b)
            b = '1' * (index + 2 - len(b)) + b  # 注意这里算出来的结果是补码
        b = b.replace("0b", '')
        b = b.replace('-', '')

        return b
    elif (feature == False):  # 二进制转换为十进制
        i = int(str(number), 2)
        if (i >= 2 ** (index - 1)):  # 如果是负数
            i = -(2 ** index - i)
            return i
        else:
            return i


def strToInt(nums):
    tem = []
    for i in nums:
        tem.append(int(i))
    return tem


def binTohex(binNums, bit_nums):
    nums = bit_nums / 4
    res = hex(int(binNums, 2))
    res = res.split("0x")[-1]
    if len(res) != nums:
        res = "0" * int(nums - len(res)) + res
    return res


def tensorToBintensor(tensor):
    tensor_tem = make_list_array([tensor.shape[0]])
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            nums = intToBin(tensor[i][j], 8, feature=True)
            nums = strToInt(nums)
            tensor_tem[i].append(nums)
    tensor_tem = np.array(tensor_tem).astype(np.int8)
    tensor_tem = tensor_tem.reshape(tensor_tem.shape[0], -1)

    return tensor_tem


def tensorToHextensor(weight, sparsity_tensor):
    tensor_tem = []
    shape = weight.shape
    shape = [shape[0], shape[1], shape[2], shape[3], int(shape[4] * shape[5] / 8) + 24]
    weight = weight.reshape(shape[0], shape[1], shape[2], shape[3], -1)
    weight = weight.reshape(-1, weight.shape[-1])
    sparsity_tensor = sparsity_tensor.reshape(-1, sparsity_tensor.shape[-1])

    for i in range(weight.shape[0]):
        n = 0
        string = ''
        for j in range(weight.shape[1]):
            number = intToBin(weight[i][j], 8, feature=True)
            number = binTohex(number, 8)
            string = number + string
            n += 1
            if n == 8:
                tensor_tem.append(string)
                n = 0
                string = ''

        sparsity_string = ''
        for j in range(sparsity_tensor.shape[1]):
            number = intToBin(sparsity_tensor[i][j], 3, feature=True)
            sparsity_string += number
        # n = 0
        # string = ''
        # for st in sparsity_string:
        #     string += st
        #     n += 1
        #     if n == 64:
        #         number = binTohex(string, 64)
        #         tensor_tem.append(number)
        #         n = 0
        #         string = ''

        sparsity_tem = []
        for k in range(3):
            string = ''
            for i in range(int(len(sparsity_string) / 3)):
                string += sparsity_string[i * 3 + 2 - k]
            for j in range(int(len(string) / 64)):
                number = binTohex(string[j * 64:(j + 1) * 64], 64)
                reverse_number = ""
                for z in range(int(len(number) / 2)):
                    reverse_number = number[z * 2:(z * 2) + 2] + reverse_number
                sparsity_tem.append(reverse_number)

        sparsity_tem2 = []
        for k in range(4):
            for i in range(3):
                sparsity_tem2.append(sparsity_tem[i * 8 + k * 2])
                sparsity_tem2.append(sparsity_tem[i * 8 + k * 2 + 1])

        for sp in sparsity_tem2:
            tensor_tem.append(sp)

    return shape, tensor_tem


def cluster_format(weight, cluster_psum, cim_psum, split_c, frist_layer):
    assert (cluster_psum == 1 or cluster_psum == 2 or cluster_psum % 4 == 0)
    weight_n = weight.shape[0]
    weight_h = weight.shape[1]
    weight_w = weight.shape[2]
    weight_c = weight.shape[3]
    # assert(weight_c%16==0)

    weight = weight.reshape([weight_n, weight_h, weight_w, split_c, -1])
    weight = np.transpose(weight, (3, 0, 1, 2, 4))
    weight_c = weight.shape[-1]
    cim_nums = cluster_psum * cim_psum * int(weight_n / 16)
    # if cluster_psum == 1:
    #     return weight
    cluster_nums = math.ceil(cim_nums / 4)
    cycle_nums = math.ceil(cluster_nums / 4)

    weight = weight.reshape([split_c, cycle_nums, -1, weight_h, weight_w, cluster_psum, int(weight_c / cluster_psum)])
    weight = np.transpose(weight, (0, 1, 2, 5, 3, 4, 6))

    weight = weight.reshape([weight.shape[0], weight.shape[1], int(weight.shape[2] / 16), 16, -1])
    shape = weight.shape

    weight = weight.reshape(-1, shape[-1])
    if frist_layer:
        pass
    else:
        for i in range(weight.shape[0]):
            weight_res = []
            for j in range(cluster_psum):
                weight_tem = weight[i][
                             j * int(weight.shape[1] / cluster_psum):(j + 1) * int(weight.shape[1] / cluster_psum)]
                weight_dst = make_list_array([cim_psum])
                n = 0
                while n < weight_tem.shape[0]:
                    for z in range(cim_psum):
                        weight_dst[z].append(weight_tem[n:n + 8])
                        n += 8
                weight_res.append(weight_dst)
            weight_res = np.array(weight_res).astype(np.int8).reshape(-1)
            weight[i] = weight_res

    weight = weight.reshape([shape[0], shape[1], shape[2], shape[3], cluster_psum * cim_psum, -1])
    weight = np.transpose(weight, (0, 1, 2, 4, 5, 3))

    return weight


def sparsity_class(weight):
    shape = weight.shape
    weight = weight.reshape(-1, shape[-2], shape[-1])
    sparsity_tensor = []
    for i in range(weight.shape[0]):
        cim_tem = np.zeros([256, 16]).astype(np.int8)
        cim_tem[:shape[-2], :] = weight[i]
        cim_tem = tensorToBintensor(cim_tem).reshape([4, 64, 128])
        # cim_tem = np.transpose(cim_tem,(0,2,1))
        for ii in range(4):
            for jj in range(128):
                nn = 0
                for zz in range(64):
                    nn += cim_tem[ii][zz][jj]
                if nn == 0:
                    sparsity_tensor.append(0)
                elif nn == 1:
                    sparsity_tensor.append(1)
                elif nn >= 2 and nn <= 3:
                    sparsity_tensor.append(2)
                elif nn >= 4 and nn <= 7:
                    sparsity_tensor.append(3)
                elif nn >= 8 and nn <= 15:
                    sparsity_tensor.append(4)
                elif nn >= 16 and nn <= 31:
                    sparsity_tensor.append(5)
                elif nn >= 32 and nn <= 63:
                    sparsity_tensor.append(6)
                elif nn == 64:
                    sparsity_tensor.append(7)
    sparsity_tensor = np.array(sparsity_tensor).astype(np.int8)
    sparsity_tensor = sparsity_tensor.reshape([shape[0], shape[1], shape[2], shape[3], -1])
    return sparsity_tensor


def add_bais_shift_quatization(shape, res, max, min, output_offset, \
                               output_shift, output_multiplier, BiasValue):
    new_res = []
    out_ch = int(output_multiplier.shape[0] / shape[1])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for z in range(shape[2] * shape[3] * shape[4]):
                new_res.append(
                    res[i * shape[1] * shape[2] * shape[3] * shape[4] + j * shape[2] * shape[3] * shape[4] + z])
            if i == shape[0] - 1:
                BiasValue_tem = BiasValue[j * out_ch:(j + 1) * out_ch]
                output_shift_tem = output_shift[j * out_ch:(j + 1) * out_ch]
                output_multiplier_tem = output_multiplier[j * out_ch:(j + 1) * out_ch]
                for m in range(int(out_ch / 16)):
                    n = 0
                    string = ''
                    for ch in range(16):
                        bin = intToBin(BiasValue_tem[m * 16 + ch], 32, feature=True)
                        hex_nums = binTohex(bin, 32)
                        string = hex_nums + string
                        n += 1
                        if n == 2:
                            new_res.append(string)
                            n = 0
                            string = ''

                for m in range(int(out_ch / 16)):
                    n = 0
                    string = ''
                    for ch in range(16):
                        bin = intToBin(output_multiplier_tem[m * 16 + ch], 32, feature=True)
                        hex_nums = binTohex(bin, 32)
                        string = hex_nums + string
                        n += 1
                        if n == 2:
                            new_res.append(string)
                            n = 0
                            string = ''

                    n = 0
                    string = ''
                    for ch in range(16):
                        bin = intToBin(output_shift_tem[m * 16 + ch], 8, feature=True)
                        hex_nums = binTohex(bin, 8)
                        string = hex_nums + string
                        n += 1
                        if n == 8:
                            new_res.append(string)
                            n = 0
                            string = ''

                    string = intToBin(0, 40, feature=True)
                    bin = intToBin(max, 8, feature=True)
                    string += bin
                    bin = intToBin(min, 8, feature=True)
                    string += bin
                    bin = intToBin(output_offset[0], 8, feature=True)
                    string += bin
                    hex_nums = binTohex(string, 64)
                    new_res.append(hex_nums)

    shape = [[shape[0], shape[1], shape[2], shape[3], shape[4]], [shape[1], int(out_ch * (19) / 16)]]

    return shape, new_res


manager = multiprocessing.Manager()
weight_dict = manager.dict()


def wm(op: block_param):
    npu_conv_op = op.get_npu_conv_op()
    weight = npu_conv_op.WeightValue
    output_multiplier = npu_conv_op.output_multiplier
    output_offset = npu_conv_op.output_offset
    output_shift = npu_conv_op.output_shift
    weights_offset = npu_conv_op.weights_offset
    BiasValue = npu_conv_op.BiasValue

    k_n, k_h, k_w, k_c = weight.shape
    cluster_psum = op.weight_mapping_dict['cluster_psum']
    cim_psum = op.weight_mapping_dict['cim_psum']
    split_c = op.block_split_mode.c
    frist_layer = npu_conv_op.FirstLayer

    weight_format = []
    weight = cluster_format(weight, cluster_psum, cim_psum, split_c, frist_layer)
    weight_format.append(weight.shape)
    sparsity_tensor = sparsity_class(weight)
    weight_format.append(sparsity_tensor.shape)
    shape, res = tensorToHextensor(weight, sparsity_tensor)
    weight_format.append(shape)
    shape, res = add_bais_shift_quatization(shape, res, 127, -128, output_offset, \
                                            output_shift, output_multiplier, BiasValue)
    weight_format.append(shape)
    # op.weight_mapping_dict["weight_format"] = weight_format
    weight_dict[op.npu_op_id] = [weight_format, res]
    # print(op.npu_op_id)
    return 1


class TransformRule(Enum):
    NOPE = 1

    WEIGHT_PADDING = 2
    WEIGHT_MAPPING = 3
    WEIGHT_MAPPING_MULTI_PROCSS = 4


@_register_ir_transformation_rule(TransformRule.WEIGHT_PADDING)
def _weight_padding(net: GraphIR):
    for group_op_id, group_op in enumerate(net.AllOps):
        for op in group_op.block_list:
            if isinstance(op, block_param):
                npu_conv_op = op.get_npu_conv_op()
                if npu_conv_op is not None:
                    # weight_split
                    weight = npu_conv_op.WeightValue
                    bais = npu_conv_op.BiasValue
                    output_shift = npu_conv_op.output_shift
                    weights_offset = npu_conv_op.weights_offset
                    output_multiplier = npu_conv_op.output_multiplier
                    k_n, k_h, k_w, k_c = weight.shape

                    if k_n % 16 != 0:
                        assert op.output_block == True
                        if k_n < 32:
                            n_k_n = 32
                        elif k_n < 64:
                            n_k_n = 64
                        elif k_n < 128:
                            n_k_n = 128
                        elif k_n < 256:
                            n_k_n = 256
                        else:
                            n_k_n = math.ceil(k_n / 16) * 16
                        weight_ = np.zeros([n_k_n, k_h, k_w, k_c]).astype(np.int8)
                        weight_[0:k_n, :, :, :] = weight
                        npu_conv_op.WeightValue = weight_

                        bais_ = np.zeros([n_k_n]).astype(np.int32)
                        bais_[0:k_n] = bais
                        npu_conv_op.BiasValue = bais_

                        output_shift_ = np.zeros([n_k_n]).astype(np.int32)
                        output_shift_[0:k_n] = output_shift
                        npu_conv_op.output_shift = output_shift_

                        weights_offset_ = np.zeros([n_k_n]).astype(np.int32)
                        weights_offset_[0:k_n] = weights_offset
                        npu_conv_op.weights_offset = weights_offset_

                        output_multiplier_ = np.zeros([n_k_n]).astype(np.int32)
                        output_multiplier_[0:k_n] = output_multiplier
                        npu_conv_op.output_multiplier = output_multiplier_
                        assert npu_conv_op.OutputC == n_k_n


@_register_ir_transformation_rule(TransformRule.WEIGHT_MAPPING)
def _weight_mapping(net: GraphIR):
    for group_op_id, group_op in enumerate(net.AllOps):
        for op in group_op.block_list:
            if isinstance(op, block_param):
                npu_conv_op = op.get_npu_conv_op()
                if npu_conv_op is not None:
                    if net.check_weight_tensor(op.npu_op_id) == True:
                        op.weight_mapping_dict["weight_format"] = net.get_weight_format(op.npu_op_id)
                        continue

                    weight = npu_conv_op.WeightValue
                    output_multiplier = npu_conv_op.output_multiplier
                    output_offset = npu_conv_op.output_offset
                    output_shift = npu_conv_op.output_shift
                    weights_offset = npu_conv_op.weights_offset
                    BiasValue = npu_conv_op.BiasValue

                    k_n, k_h, k_w, k_c = weight.shape
                    cluster_psum = op.weight_mapping_dict['cluster_psum']
                    cim_psum = op.weight_mapping_dict['cim_psum']
                    split_c = op.block_split_mode.c
                    frist_layer = npu_conv_op.FirstLayer

                    weight_format = []
                    weight = cluster_format(weight, cluster_psum, cim_psum, split_c, frist_layer)
                    weight_format.append(weight.shape)
                    sparsity_tensor = sparsity_class(weight)
                    weight_format.append(sparsity_tensor.shape)
                    shape, res = tensorToHextensor(weight, sparsity_tensor)
                    weight_format.append(shape)
                    shape, res = add_bais_shift_quatization(shape, res, 127, -128, output_offset, \
                                                            output_shift, output_multiplier, BiasValue)
                    weight_format.append(shape)
                    op.weight_mapping_dict["weight_format"] = weight_format
                    net.add_weight_tensor(op.npu_op_id, res)
                    net.add_weight_format(weight_format)


@_register_ir_transformation_rule(TransformRule.WEIGHT_MAPPING_MULTI_PROCSS)
def _weight_mapping_multi_procss(net: GraphIR):
    target_op_list = []
    for group_op_id, group_op in enumerate(net.AllOps):
        for op in group_op.block_list:
            if isinstance(op, block_param):
                npu_conv_op = op.get_npu_conv_op()
                if npu_conv_op is not None:
                    target_op_list.append(op)

    pool = multiprocessing.Pool()

    result = pool.map(wm, target_op_list)
    assert sum(result) == len(target_op_list)

    for target_op in target_op_list:
        npu_op_id = target_op.npu_op_id
        weight_format, res = weight_dict[npu_op_id]

        target_op.weight_mapping_dict["weight_format"] = weight_format

        if net.check_weight_tensor(npu_op_id) != True:
            net.add_weight_tensor(npu_op_id, res)
            net.add_weight_format(weight_format)


# weight_mapping_pass
weight_mapping_transform = [TransformRule.WEIGHT_PADDING,
                            # weight_mapping_transform.append(TransformRule.WEIGHT_MAPPING)
                            TransformRule.WEIGHT_MAPPING_MULTI_PROCSS
                            ]

