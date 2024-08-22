import numpy as np


def norm(x):
    return np.array(x, dtype=np.float64)


def QuantizeMultiplier(x):
    assert x.all() > 0., "real_multiplier err"
    q, shift = np.frexp(x)  # 返回m和e的指数 q属于[0.5, 1)
    q_fixed = np.round(q * (1 << 31)).astype(np.int32)  # int32 mulhigh  这样做的目的是将尾数放大到一个可以用32位整数表示的范围内

    if (q_fixed == (1 << 31)).sum() > 0:  # q_fixed 四舍五入后溢出，q减小一倍，右移加一，q * 2 ^ shift 的结果保持不变
        shift[q_fixed == (1 << 31)] += 1
        q_fixed[q_fixed == (1 << 31)] /= 2
    if (shift < -31).sum() > 0:  # x 的绝对值小于 2 ^ −31
        q_fixed[shift < -31] = 0
        shift[shift < -31] = 0

    return q_fixed, shift


def Quantize(scale, zero_point, f):
    """
        according to activation function to calculate Value range
    """
    tmp = f / scale
    q = zero_point + tmp
    return q


def get_activation_range(output_scale, zero_point, activation_function):
    """
        activation function, support relu, relu6, 
    """
    # support int8
    q_max = 127
    q_min = -128

    if activation_function is None:
        pass
    elif activation_function == "RELU":
        tmp_q = Quantize(output_scale, zero_point, 0.0)
        q_min = max(q_min, tmp_q)
    elif activation_function == "RELU6":
        tmp_q = Quantize(output_scale, zero_point, 0.0)
        q_min = max(q_min, tmp_q)

        tmp_q = Quantize(output_scale, zero_point, 6.0)
        q_max = min(q_min, tmp_q)
    elif activation_function == "SIGMOID":
        tmp_q = Quantize(output_scale,zero_point, 0.0)
        q_min = max(q_min, tmp_q)

        tmp_q = Quantize(output_scale, zero_point, 1.0)
        q_max = min(q_min, tmp_q)
    else:  # not support other activation function so far
        pass

    return q_max, q_min
