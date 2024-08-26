from json import JSONEncoder
import numpy as np
import math


def OneD2ThreeD(data, h, w, c):
    output = np.zeros([h, w, c], dtype=np.int8)
    for h_idx in range(h):
        for w_idx in range(w):
            for c_idx in range(c):
                output[h_idx, w_idx, c_idx] = data[c_idx + w_idx * c + h_idx * w * c]
    return output


def ThreeD2OneD(data):
    h, w, c = data.shape
    output = np.zeros([c * w * h], dtype=np.int8)

    for h_idx in range(h):
        for w_idx in range(w):
            for c_idx in range(c):
                output[c_idx + w_idx * c + h_idx * w * c] = data[h_idx, w_idx, c_idx]
    return output


def OneD2FourD(data, n, h, w, c):
    output = np.zeros([n, h, w, c], dtype=np.int8)

    for n_idx in range(n):
        for h_idx in range(h):
            for w_idx in range(w):
                for c_idx in range(c):
                    output[n_idx, h_idx, w_idx, c_idx] \
                        = data[c_idx + w_idx * c + h_idx * w * c + n_idx * h * w * c]
    return output


def FourD2OneD(data):
    n, h, w, c = data.shape
    output = np.zeros([n * c * w * h], dtype=np.int8)

    for n_idx in range(n):
        for h_idx in range(h):
            for w_idx in range(w):
                for c_idx in range(c):
                    output[c_idx + w_idx * c + h_idx * w * c + n_idx * h * w * c] \
                        = data[n_idx, h_idx, w_idx, c_idx]
    return output


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return obj.__name__
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return JSONEncoder.default(self, obj)


def intToBin(number, index, feature=True):
    # index为该数据位宽,number为待转换数据,
    # feature为True则进行十进制转二进制(补码)，为False则进行二进制转十进制。
    if (feature == True):
        if (number >= 0):
            b = bin(number)
            b = '0' * (index + 2 - len(b)) + b
        else:
            b = 2 ** (index) + number
            b = bin(b)
            b = '1' * (index + 2 - len(b)) + b
        b = b.replace("0b", '')
        b = b.replace('-', '')

        return b
    elif (feature == False):
        i = int(str(number), 2)
        if (i >= 2 ** (index - 1)):
            i = -(2 ** index - i)
            return i
        else:
            return i


def Array2Txt_hwc(x, in_bit, out_bit, path):
    num = int(out_bit / in_bit)
    digit = int(in_bit / 4)
    assert (len(x.shape) == 1), "except 1 dim not {} dim".format(len(x.shape))
    cricle = math.ceil(x.shape[0] / num)
    f = open(path, "w")
    for c in range(cricle):
        string = ""
        for n in range(num):
            try:
                com_bin = str(hex(int(str(intToBin(x[c * num + n], \
                                                   in_bit, feature=True)), 2))).split("x")[-1]

                if len(com_bin) != digit:
                    com_bin = "0" * (digit - len(com_bin)) + com_bin
                string += com_bin
            except:
                string += "0" * digit
        f.write(string + "\n")
    f.close()


def Array2Txt_hwc_hex(x, in_bit=8, out_bit=32, path=""):
    nums_per_line = int(out_bit / in_bit)
    digit = int(in_bit / 4)
    cricle = math.ceil(x.shape[0] / nums_per_line)
    # path = os.path.dirname(__file__)
    f = open(path, "w")
    for c in range(cricle):
        string = ""
        for n in range(nums_per_line):
            try:
                com_bin = str(hex(int(str(intToBin(x[c * nums_per_line + n], in_bit, feature=True)), 2))).split("x")[-1]
                if len(com_bin) != digit:
                    com_bin = "0" * (digit - len(com_bin)) + com_bin
                string += com_bin
            except:
                string += "0" * digit
        f.write(string + "\n")
    f.close()


def Array2Txt_hwc_decima(x, in_bit=8, out_bit=32, path=""):
    nums_per_line = int(out_bit / in_bit)
    cricle = math.ceil(x.shape[0] / nums_per_line)
    # path = os.path.dirname(__file__)
    f = open(path, "w")
    for c in range(cricle):
        string = ""
        for n in range(nums_per_line):
            try:
                com_bin = str(x[c * nums_per_line + n])

                string += com_bin + ","
            except:
                pass
        f.write(string + "\n")
    f.close()


def Parameter2Txt_decima(names, para, path):
    f = open(path, "w")
    for name in names:
        string = ""
        string += f"{name}: {para[name]}\n"
        f.write(string)
    f.close()


def get_ratio_r(input_shape, output_shape):
    in_height = input_shape[0]
    out_height = output_shape[0]
    in_width = input_shape[1]
    out_width = output_shape[1]
    ratio_h = (int)(((1 << 10) * in_height + out_height / 2) / out_height)
    ratio_w = (int)(((1 << 10) * in_width + out_width / 2) / out_width)

    return ratio_w, ratio_h


def get_pad_val(input_size, output_size, filter, stride, padding):
    if len(output_size) == 4:
        output_size = [output_size[1], output_size[2], output_size[3]]
    if len(input_size) == 4:
        input_size = [input_size[1], input_size[2], input_size[3]]
    if padding == 'VALID':
        padding_values_width = 0
        padding_values_height = 0
        return padding_values_width, padding_values_height
    else:
        padding_values_height = max(0, (output_size[0] - 1) * stride + filter - input_size[0])
        padding_values_height = int(padding_values_height / 2)

        padding_values_width = max(0, (output_size[1] - 1) * stride + filter - input_size[1])
        padding_values_width = int(padding_values_width / 2)

        return padding_values_width, padding_values_height


def toPolling(x, polling_bit=8 * 8, in_bit=8, bank_num=4):
    channls_per_polling = int(polling_bit / in_bit) * bank_num  # 32 channls
    nums_per_polling = int(channls_per_polling / bank_num)  # 8 nums
    channls_per_loop = int(x.shape[2] / channls_per_polling)
    bank = [[], [], [], []]
    y = np.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=np.int8)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            n = 0
            for k in range(bank_num):
                for z in range(channls_per_loop):
                    # y[i][j][n:n+nums_per_polling] = \
                    # x[i][j][z*channls_per_polling+k*nums_per_polling:z*channls_per_polling+(k+1)*nums_per_polling]
                    # n+=nums_per_polling
                    bank[k].append(x[i][j][z * channls_per_polling + k * nums_per_polling:z * channls_per_polling + (
                            k + 1) * nums_per_polling])
                    n += nums_per_polling

    return np.array(bank[0]), np.array(bank[1]), np.array(bank[2]), np.array(bank[3])


def Array2Txt_hwc_hex_waddr(x, addr, in_bit=8, out_bit=32, path=""):
    dict = {}
    nums_per_line = int(out_bit / in_bit)
    digit = int(in_bit / 4)
    cricle = math.ceil(x.shape[0] / nums_per_line)
    # path = os.path.dirname(__file__)

    for c in range(cricle):
        string = addr[c] + " "
        for n in range(nums_per_line):
            try:
                com_bin = str(hex(int(str(intToBin(x[c * nums_per_line + n], in_bit, feature=True)), 2))).split("x")[-1]
                if len(com_bin) != digit:
                    com_bin = "0" * (digit - len(com_bin)) + com_bin
                string += com_bin
            except:
                string += "0" * digit
        dict[string[:12]] = string[12:]

    return dict


if __name__ == "__main__":
    a = np.array([i for i in range(1, 129)]).reshape(1, 1, 128)
    print(a.reshape([-1]))
    y = toPolling(a, polling_bit=64)
    print(y[0])
