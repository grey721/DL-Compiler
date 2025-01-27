def within_n_base_2(n: int, var: int):
    """
    将不为1的整数var补偿为，n以内最接近的以2为底的数
    """
    compensation = var % n
    if compensation & (compensation - 1):  # 判断是否以2为底
        var += (1 << compensation.bit_length()) - compensation
    return var


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x > pivot]  # 大于枢轴的元素
    middle = [x for x in arr if x == pivot]  # 等于枢轴的元素
    right = [x for x in arr if x < pivot]  # 小于枢轴的元素
    return quick_sort(left) + middle + quick_sort(right)


def generate_binary_number(n):
    return (1 << n) - 1
