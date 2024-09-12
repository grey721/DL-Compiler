
from tool.utils import *


share_mem_dict = {0: [81920, 98304, 114688, 131072], 1: [147456, 163840, 180224, 196608],
                  2: [212992, 229376, 245760, 262144]}
zhaofang_share_mem_dict = {0: [81920, 98304, 114688, 131072], 1: [147456, 163840, 180224, 196608],
                           2: [212992, 229376, 245760, 262144]}

vpu_addr = ['00009000', '00009004', '00009008', '0000900c', '00009010', '00009014', '00009018', '0000901c', '00009020',
            '00009024', '00009028', '0000902c', '00009030', '00009034', '00009038', '0000903c', '00009040', '00009044',
            '00009048', '0000904c', '00009050', '00009054', '00009058', '0000905c', '00009060', '00009064', '00009068',
            '0000906c', '00009070', '00009074', '00009078', '0000907c', '00009080', '00009084', '00009088', '0000908c',
            '00009090', '00009094', '00009098', '0000909c', '000090a0', '000090a4', '000090a8', '000090ac', '000090b0',
            '000090b8', '000090b4', '000090bc', '000090c0', '000090c4', '000090c8', '000090cc', '000090d0', '000090d4',
            '000090d8', '000090dc', '000090e0', '000090e4', '000090e8', '000090ec', '000090f0', '000090f4', '000090f8',
            '000090fc', '00009150', '00009154', '00009158', '0000915c', '00009160', '00009164', '00009168', '0000916c',
            '00009170', '00009174', '00009178', '0000917c', '00009180', '00009184', '00009188', '0000918c', '00009190',
            '00009194', '00009198', '0000919c', '000091a0', '000091a4', '000091a8', '000091ac', '000091b0', '000091b4',
            '000091b8', '000091bc', '000091c0', '000091c4', '000091c8', '000091cc', '000091d0', '000091d4', '000091d8',
            '000091dc', '000091e0', '000091e4', '000091e8', '000091ec', '000091f0', '000091f4', '000091f8', '00009250',
            '00009254', '00009258', '0000925c', '00009260', '00009264', '00009268', '0000926c', '00009270', '00009274',
            '00009278', '0000927c', '00009280', '00009284', '00009288', '0000928c', '00009290', '00009294', '00009298',
            '0000929c', '000092a0', '000092a4', '000092a8', '000092ac', '000092b0', '000092b4', '000092b8', '000092bc',
            '000092c0', '000092c4', '000092c8', '000092cc', '000092d0', '000092d4', '000092d8', '000092dc', '000092e0',
            '000092e4', '000092e8', '000092ec', '000092f0', '000092f4', '000092f8', '00009450', '00009454', '00009458',
            '0000945c', '00009460', '00009464', '00009468', '0000946c', '00009470', '00009474', '00009478', '0000947c',
            '00009480', '00009484', '00009488', '0000948c', '00009490', '00009494', '00009498', '0000949c', '000094a0',
            '000094a4', '000094a8', '000094ac', '000094b0', '000094b4', '000094b8', '000094bc', '000094c0', '000094c4',
            '000094c8', '000094cc', '000094d0', '000094d4', '000094d8', '000094dc', '000094e0', '000094e4', '000094e8',
            '000094ec', '000094f0', '000094f4', '000094f8', '00009850', '00009854', '00009858', '0000985c', '00009860',
            '00009864', '00009868', '0000986c', '00009870', '00009874', '00009878', '0000987c', '00009880', '00009884',
            '00009888', '0000988c', '00009890', '00009894', '00009898', '0000989c', '000098a0', '000098a4', '000098a8',
            '000098ac', '000098b0', '000098b4', '000098b8', '000098bc', '000098c0', '000098c4', '000098c8', '000098cc',
            '000098d0', '000098d4', '000098d8', '000098dc', '000098e0', '000098e4', '000098e8', '000098ec', '000098f0',
            '000098f4',
            '000098f8']  # cim_addr = ['00006800', '00006804', '00006808', '0000680c', '00006810', '00006814', '00006818', '0000681c', '00006820', '00006824', '00006828', '0000682c', '00006830', '00006904', '00006944']
cim_addr = ['00006800', '00006804', '00006808', '0000680c', '00006810', '00006814', '00006818', '0000681c', '00006820',
            '00006824', '00006828', '0000682c', '00006830']
pre_addr = ['00006000', '00006004', '00006008', '0000600c', '00006010', '00006014', '00006018', '00006100', '00006104',
            '00006108', '0000610c', '00006110', '00006114', '00006118', '0000611c', '00006120', '00006124', '00006128',
            '0000612c', '00006130', '00006134', '00006138', '00006200', '00006204', '00006208', '0000620c', '00006210',
            '00006214', '00006218', '0000621c', '00006220', '00006224', '00006228', '0000622c', '00006230', '00006234',
            '00006238', '00006300', '00006304', '00006308', '0000630c', '00006310', '00006314', '00006318', '0000631c',
            '00006320', '00006324', '00006328', '0000632c', '00006330', '00006334', '00006338', '00006400', '00006404',
            '00006408', '0000640c', '00006410', '00006414', '00006418', '0000641c', '00006420', '00006424', '00006428',
            '0000642c', '00006430', '00006434', '00006438']


def pre_writeTxt(res_txt, path, with_addr=True):
    f = open(path, "w")
    n = 0
    string = ''
    for i in res_txt:
        for j in i:
            if with_addr:
                string += "32'h" + f'{pre_addr[n]} ' + j + '\n'
            else:
                string += j + '\n'
            n += 1
    f.write(string)
    f.close()


def cim_cluster_writeTxt(res_txt, path, with_addr=True):
    f = open(path, "w")
    n = 0
    string = ''
    assert (len(res_txt) == len(cim_addr))
    for i in res_txt:
        if with_addr:
            string += "32'h" + f'{cim_addr[n]} ' + i + '\n'
        else:
            string += i + '\n'
        n += 1

    f.write(string)
    f.close()


def vpu_writeTxt(res_txt, path):
    f = open(path, 'w')
    string = ''
    for i in range(len(res_txt) - 64):
        string += "32'h" + vpu_addr[i] + ' ' + res_txt[i] + '\n'

    for i in range(16):
        addr = 37120 + i * 4
        addr = intToBin(addr, 32, feature=True)
        addr = binTohex(addr, 32)
        string += f"32'h{addr}" + ' ' + res_txt[i + len(res_txt) - 64] + '\n'

    for i in range(16):
        addr = 37376 + i * 4
        addr = intToBin(addr, 32, feature=True)
        addr = binTohex(addr, 32)
        string += f"32'h{addr}" + ' ' + res_txt[i + len(res_txt) - 64 + 16] + '\n'

    for i in range(16):
        addr = 37888 + i * 4
        addr = intToBin(addr, 32, feature=True)
        addr = binTohex(addr, 32)
        string += f"32'h{addr}" + ' ' + res_txt[i + len(res_txt) - 64 + 32] + '\n'

    for i in range(16):
        addr = 38912 + i * 4
        addr = intToBin(addr, 32, feature=True)
        addr = binTohex(addr, 32)
        string += f"32'h{addr}" + ' ' + res_txt[i + len(res_txt) - 64 + 48] + '\n'

    f.write(string)
    f.close()


def intToBin(number, index, feature=True):
    # index is the bit width of the data and number is the data to be converted.
    # If feature is True, decimal to binary (complement code) is performed;
    # if feature is False, binary to decimal is performed.
    assert (isinstance(number, int)
            or isinstance(number, np.int32)
            or isinstance(number, np.int64)
            or isinstance(number, np.int8)), 'the type of number must be int'
    if feature:
        # Decimal to binary (signed, using two's complement)
        if number >= 0:
            binary = bin(number)[2:].zfill(index)
        else:
            # Calculate two's complement
            mask = (1 << index) - 1
            binary = bin(~number & mask)[2:].zfill(index)
        return binary
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
