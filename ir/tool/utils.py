def within_n_base_2(n: int, var: int):
    """
    将不为1的整数var补偿为，n以内最接近的以2为底的数
    """
    if var != 1:
        compensation = var % n
        if compensation and n & (n-1) == 0:  # 判断是否以2为底
            var += (1 << compensation.bit_length()) - compensation
    return var
