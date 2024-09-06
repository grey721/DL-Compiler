from ir.conversion.optimization.ir_transform import _register_ir_transformation_rule
from enum import Enum
from tool.utils import *
import cv2
import os

import json

class TransformRule(Enum):
    CREATE_REGISTER = 1
    CREATE_INFO = 2  # txt保存模型输入输出，层，block等信息;dma_read_list非空，则保存json包括txt的信息和输入输出形状信息和dma的信息
    CREATE_XLSX = 3  # 将block信息保存成xlsx
    CREATE_TOP_INFO = 4
    CALCULATING_THE_WEIGHT = 5
    CREATE_REGISTER_WITH_DMA = 6


def intToBin(number, index, feature=True):
    # index is the bit width of the data and number is the data to be converted. 
    # If feature is True, decimal to binary (complement code) is performed; 
    # if feature is False, binary to decimal is performed. 
    assert (isinstance(number, int)
            or isinstance(number, np.int32)
            or isinstance(number, np.int64)
            or isinstance(number, np.int8)), 'the type of number must be int'
    if feature is True:
        if number >= 0:

            b = bin(number)
            b = '0' * (index + 2 - len(b)) + b
        else:
            b = 2 ** index + number
            b = bin(b)
            b = '1' * (index + 2 - len(b)) + b
        b = b.replace("0b", '')
        b = b.replace('-', '')
        assert (len(b) == index), "out of bitnums"
        return b
    elif feature is False:
        i = int(str(number), 2)
        if i >= 2 ** (index - 1):
            i = -(2 ** index - i)
            return i
        else:
            return i


def bin_listTobin(bin_list):
    bin = ""
    for i in bin_list:
        bin += i
    assert (len(bin) == 32), f"value of bitnums {len(bin)}==32"
    return bin


def binTohex(binNums, bit_nums):
    nums = bit_nums / 4
    res = hex(int(binNums, 2))
    res = res.split("0x")[-1]
    if len(res) != nums:
        res = "0" * int(nums - len(res)) + res
    return res


def get_list(nums=22, val=0):
    l = [val] * nums
    return l


def read_file(path, file_list):
    with open(path, 'r') as f:
        line = f.readline()
        file_list.append(line)
        while line:
            line = f.readline()
            if line != '':
                file_list.append(line)


def read_files(path):
    file_list = []
    read_file(f'{path}/pre_param', file_list)
    read_file(f'{path}/cim_cluster_param', file_list)
    read_file(f'{path}/vpu_param', file_list)
    return file_list


def make_weight(top_info_path, npu_graph):  # 将权重保存成文件
    weight_path = f'{top_info_path}/weight'
    weight = npu_graph.WeightTensors
    f = open(weight_path, 'w')
    for i in weight:
        for j in i:
            f.write(j + '\n')
    f.close()
    pass


def make_image_to_memory(top_info_path, image_size=(128, 128, 3), image_path=''):  # 保存图片二进制数据
    top_info_path = f'{top_info_path}/image'
    if image_path == '':
        image = (np.random.rand(image_size[0] * image_size[1] * image_size[2]).reshape(image_size) * 255 - 128
                 ).astype(np.int8)  # 随机生成图片，随机生成图片size,*255是为了值[0~255)，-128是是为了[-128,127)，
    else:
        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("image is not exist")

    f = open(top_info_path, '+w')
    string = ''
    n = 0
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            for z in range(image_size[2]):
                number = intToBin(image[i][j][z], 8, feature=True)
                number = binTohex(number, 8)
                string = number + string
                n += 1
                if n == 3:  # 3通道？固定3一回车？
                    f.write(string + '\n')
                    n = 0
                    string = ''
    f.close()
    pass


def gather_register(res_list, top_head, head_list, finally_path):
    f = open(finally_path, 'w')
    for i in range(len(top_head)):
        f.write(top_head[i] + '\n')
    for i in range(len(res_list)):
        for j in range(len(head_list[i])):
            f.write(head_list[i][j] + '\n')
        for j in range(len(res_list[i])):
            f.write(res_list[i][j])
    f.close()
    pass


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (int(round(dw - 0.1)), int(round(dh - 0.1)))


def make_image(image_path, Scale, ZeroPoint, size=(416, 416, 3)):
    im = cv2.imread(image_path)
    shape = im.shape
    im, ratio, (dw, dh) = letterbox(im, size, stride=32, auto=False)
    im = im.transpose((2, 0, 1))[::-1].transpose((1, 2, 0))
    im = np.ascontiguousarray(im)

    # im = im[..., ::-1]  # conver RGB to BGR
    x = im.astype(np.float32)
    x /= 255
    x = (x / Scale + ZeroPoint).astype(np.int8)
    return x


def fill_zero(block_tensor):
    h, w, c = block_tensor.shape
    tem_tensor = np.zeros(shape=[h, w, 8]).astype(np.int8)
    tem_tensor[:, :, 2] = block_tensor[:, :, 2]
    tem_tensor[:, :, 1] = block_tensor[:, :, 1]
    tem_tensor[:, :, 0] = block_tensor[:, :, 0]
    return tem_tensor


def Array2Txt_hwc_hex_bank(x, in_bit=8, out_bit=64, path_="", bank_start=0):
    n = 0
    path = path_
    for i in range(len(x)):
        f = open(path, "w")
        string = ""
        for j in range(len(x[i])):
            number = intToBin(x[i][j], in_bit, feature=True)
            number = binTohex(number, int(out_bit / in_bit))
            string = number + string
            n += 1
            if n == 8:
                string += "\n"
                f.write(string)
                string = ""
                n = 0
        f.close()


@_register_ir_transformation_rule(TransformRule.CREATE_TOP_INFO)
def create_info(npu_graph):
    path_base = npu_graph.codegen_path
    top_info_path = f'{path_base}_top_info'
    if not os.path.exists(top_info_path):
        os.makedirs(top_info_path)
    make_weight(top_info_path, npu_graph)  # 将权重保存文件
    make_image_to_memory(top_info_path, image_size=[128, 128, 3], image_path='')  # 保存图片二进制数据

    sub_block_nums = npu_graph.sub_block_nums
    total_layer_num = len(sub_block_nums)
    tem_file_list = [''] * REGISTER_NUMS
    res_list = []
    total_sub_block_nums = 0
    head_list = []
    for i in range(total_layer_num):
        for j in range(sub_block_nums[i]):
            model_path = f"{path_base}/layer_{i}/sub_block_{j}"
            file_list = read_files(model_path)

            get_FE_register(file_list, npu_graph, total_sub_block_nums, npu_graph.dma_read_list)
            get_BE_register(file_list, npu_graph, total_sub_block_nums)
            get_WGT_register(file_list, npu_graph)

            file_list_res = compare_file(tem_file_list, file_list)  # 两个列表找不同，返回不同的数据，并使前者=后者
            # 更新head_list，存储寄存器的输入
            get_block_head(file_list_res, npu_graph, total_sub_block_nums, head_list, npu_graph.dma_read_list)
            res_list.append(file_list_res)
            # res_list.append(compare_file(tem_file_list, file_list))
            tem_file_list = copy.deepcopy(file_list)
            total_sub_block_nums += 1
    assert (total_sub_block_nums == len(res_list))
    res_list = transfer_register_format(res_list)
    top_head = get_top_head(npu_graph)
    finally_path = f'{top_info_path}/register'
    gather_register(res_list, top_head, head_list, finally_path)

    getZipDir(f"{top_info_path}/", f"{top_info_path}.zip")
    pass



@_register_ir_transformation_rule(TransformRule.CALCULATING_THE_WEIGHT)
def calculating_the_weight(npu_graph):
    WeightBaseAddress = [0]
    sum = 0
    for i in range(len(npu_graph.WeightTensors) - 1):
        WeightBaseAddress.append(sum + len(npu_graph.WeightTensors[i]))
        sum += len(npu_graph.WeightTensors[i])
    npu_graph.WeightBaseAddress = WeightBaseAddress


codegen_transform = [
                     TransformRule.CREATE_TOP_INFO,
                     ]
