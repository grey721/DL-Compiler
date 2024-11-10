def within_n_base_2(n, var):
    """
    将var补偿为，n以内最接近的以2为底的数
    """
    compensation = var % n
    if compensation and var != 1:
        if n & (n-1) == 0:
            var += (1 << compensation.bit_length()) - compensation
    return var
