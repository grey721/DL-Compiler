import numpy as np


def count2str(count):
    if count == int(count):
        count = int(count)
    if count > 0:
        return f"+{count}"
    else:
        return f"{count}"


class Variable:
    def __init__(self, symbol, params=(0, 1)):
        self.params = np.array(params)
        self.symbol = symbol

    def __add__(self, other):
        r = Variable(self.symbol)
        if isinstance(other, Variable):
            if other.symbol == self.symbol:
                len1, len2 = len(self.params), len(other.params)
                if len1 == len2:
                    r.params = self.params + other.params
                elif len1 > len2:
                    r.params = self.params.copy()
                    r.params[:len2] += other.params
                else:
                    # 如果 vec2 更长，填充 vec1
                    r.params = other.params.copy()
                    r.params[:len1] += self.params

            else:
                r = Formula()
                r.symbol = (self.symbol, other.symbol)
                r.params = np.zeros((len(self.params), len(other.params)))
                r.params[:, 0:1] += self.params[:, np.newaxis]
                r.params[0:1, :] += other.params
                return r

        elif isinstance(other, (int, float)):
            r.params = self.params.copy()
            r.params[0] += other

        else:
            raise TypeError

        return r

    def __sub__(self, other):
        r = Variable(self.symbol)
        if isinstance(other, Variable):
            if other.symbol == self.symbol:
                len1, len2 = len(self.params), len(other.params)
                if len1 == len2:
                    r.params = self.params - other.params
                elif len1 > len2:
                    r.params = self.params.copy()
                    r.params[:len2] -= other.params
                else:
                    # 如果 vec2 更长，填充 vec1
                    r.params = -1 * other.params
                    r.params[:len1] += self.params

            else:
                r = Formula()
                r.symbol = (self.symbol, other.symbol)
                r.params = np.zeros((len(self.params), len(other.params)))
                r.params[:, 0:1] += self.params[:, np.newaxis]
                r.params[0:1, :] -= other.params
                return r

        elif isinstance(other, (int, float)):
            r.params = self.params.copy()
            r.params[0] -= other

        else:
            raise TypeError

        return r

    def __mul__(self, other):
        r = Variable(self.symbol)
        if isinstance(other, Variable):
            if other.symbol == self.symbol:
                m = np.outer(self.params, other.params)
                shape = m.shape
                result = np.zeros(shape[0] + shape[1] - 1)

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        result[i + j] += m[i][j]
                result = np.trim_zeros(result, "b")  # "b"表示从末尾开始
                r.params = result

            else:
                r = Formula()
                r.symbol = (self.symbol, other.symbol)
                r.params = np.zeros((len(self.params), len(other.params)))
                for i in range(self.params.shape[0]):
                    r.params[i: i+1, 0: other.params.shape[0]] += other.params * self.params[i]

                return r

        elif isinstance(other, (int, float)):
            r.params = other * self.params

        else:
            raise TypeError

        return r

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        r = Variable(self.symbol)
        if isinstance(other, Variable):
            raise NotImplementedError
            pass

        elif isinstance(other, (int, float)):
            r.params = self.params / other

        else:
            raise TypeError

        return r

    def __pow__(self, power):
        if power > 0:
            if int(power) == power:
                power = int(power)
            if isinstance(power, int):
                power = bin(power)[:1:-1]
                temp = self  # 底数
                result = 1
                for i in power:
                    if i == "1":
                        result *= temp
                    temp *= temp
                return result
            else:
                raise TypeError(f"ErrorType{type(power)}")
        elif power == 0:
            return 1

    def __str__(self):

        formula = []
        for idx, i in enumerate(self.params[1:], 1):
            if i == 0:
                continue

            if i == 1:
                temp = f"+{self.symbol}"
            else:
                temp = f"{count2str(i)}{self.symbol}"

            if idx > 1:
                temp += f"^{idx}"

            formula.insert(0, temp)

        if self.params[0]:
            formula.append(f"{count2str(self.params[0])}")

        formula = "".join(formula)
        if formula[0] == "+":
            formula = formula[1:]

        return formula

    def __repr__(self):
        formula = []
        for idx, i in enumerate(self.params[1:], 1):
            if i == 0:
                continue

            if i == 1:
                temp = f"+{self.symbol}"
            else:
                temp = f"{count2str(i)}{self.symbol}"

            if idx > 1:
                temp += f"^{idx}"

            formula.insert(0, temp)

        if self.params[0]:
            formula.append(f"{count2str(self.params[0])}")

        formula = "".join(formula)
        if formula[0] == "+":
            formula = formula[1:]

        return formula


class Formula:
    def __init__(self):
        self.symbol = None
        self.params = np.array([[0, 1], [1, 0]])

    def get_mapping(self, other):
        """
        mapping中的位置是self.symbol中对应位置的符号，内容是在other中的索引

        """

        mapping = []
        for s in self.symbol:
            if s in other.symbol:
                mapping.append(other.symbol.index(s))
            else:
                mapping.append(None)

        return mapping

    def __add__(self, other):
        r = Formula()
        if isinstance(other, Formula):
            r.symbol = self.symbol
            flag = True
            for idx, s in enumerate(other.symbol):
                if s not in self.symbol:
                    flag = False
                    break

            if flag:  # 符号都相同,少于或等于当前符号
                mapping = self.get_mapping(other)
                other_dims = len(other.params.shape)
                other_p = other.params
                for idx, i in enumerate(mapping):
                    if i is None:
                        mapping[idx] = other_dims
                        other_dims += 1
                        other_p = np.expand_dims(other_p, axis=-1)
                other_p = other_p.transpose(mapping)  # 对齐符号顺序

                self_shape = self.params.shape
                other_shape = other_p.shape
                #
                if self_shape == other_shape:
                    r.params = self.params + other_p
                else:
                    # 切片，取各维度最大值
                    self_slice = []  # 广播，适应维度
                    other_slice = []
                    if len(self_shape) >= len(other_shape):
                        shape = list(self_shape)
                    else:
                        shape = list(other_shape)
                    for idx, (self_d, other_d) in enumerate(zip(self_shape, other_shape)):
                        self_slice.append(slice(0, self_d))
                        other_slice.append(slice(0, other_d))
                        if other_d > self_d:
                            shape[idx] = other_d

                    r.params = np.zeros(shape)
                    r.params[*self_slice] += self.params
                    r.params[*other_slice] += other_p

            else:  # 也是表达式，但有不同的符号
                r = other + self

        elif isinstance(other, (int, float)):
            r.params = self.params.copy()
            r.symbol = self.symbol
            # 动态生成索引元组
            index_tuple = (0,) * r.params.ndim
            r.params[index_tuple] += other  # (0,0,0,0....)

        elif isinstance(other, Variable):
            self_shape = self.params.shape
            other_len = other.params.shape[0]
            if other.symbol in self.symbol:  # 是已有的符号
                r.symbol = self.symbol
                sym_idx = self.symbol.index(other.symbol)

                if self_shape[sym_idx] < other_len:
                    r_shape = [*self_shape]
                    r_shape[sym_idx] = other_len
                    r.params = np.zeros(r_shape)
                    self_slice = [slice(0, i) for i in self_shape]
                    r.params[*self_slice] += self.params
                else:
                    r.params = self.params.copy()

                other_slice = [0] * len(self_shape)
                other_slice[sym_idx] = slice(0, other_len)
                r.params[*other_slice] += other.params

            else:  # 新符号
                r.symbol = (other.symbol, *self.symbol)
                self_slice = [0] * len(self_shape)

                r.params = np.zeros((other_len, *self_shape))

                r.params[:, *self_slice] += other.params
                r.params[0, ...] += self.params

        else:
            raise TypeError

        return r

    def __sub__(self, other):
        r = Formula()
        if isinstance(other, Formula):
            r.symbol = self.symbol

            flag = True
            for idx, s in enumerate(other.symbol):
                if s not in self.symbol:
                    flag = False
                    break

            if flag:  # 符号都相同,少于或等于当前符号
                mapping = self.get_mapping(other)
                other_dims = len(other.params.shape)
                other_p = other.params
                for idx, i in enumerate(mapping):
                    if i is None:
                        mapping[idx] = other_dims
                        other_dims += 1
                        other_p = np.expand_dims(other_p, axis=-1)
                other_p = other_p.transpose(mapping)  # 对齐符号顺序

                self_shape = self.params.shape
                other_shape = other_p.shape
                #
                if self_shape == other_shape:
                    r.params = self.params - other_p  # 减后面
                else:
                    # 切片，取各维度最大值
                    self_slice = []  # 广播，适应维度
                    other_slice = []
                    if len(self_shape) >= len(other_shape):
                        shape = list(self_shape)
                    else:
                        shape = list(other_shape)
                    for idx, (self_d, other_d) in enumerate(zip(self_shape, other_shape)):
                        self_slice.append(slice(0, self_d))
                        other_slice.append(slice(0, other_d))
                        if other_d > self_d:
                            shape[idx] = other_d

                    r.params = np.zeros(shape)
                    r.params[*self_slice] += self.params
                    r.params[*other_slice] -= other_p  # 减后面

            else:  # 也是表达式，但有不同的符号
                r = other - self
                r *= -1

        elif isinstance(other, (int, float)):
            r.params = self.params.copy()
            r.symbol = self.symbol
            # 动态生成索引元组
            index_tuple = (0,) * r.params.ndim
            r.params[index_tuple] -= other  # (0,0,0,0....)  # 减后面

        elif isinstance(other, Variable):
            self_shape = self.params.shape
            other_len = other.params.shape[0]
            if other.symbol in self.symbol:  # 是已有的符号
                r.symbol = self.symbol
                sym_idx = self.symbol.index(other.symbol)

                if self_shape[sym_idx] < other_len:
                    r_shape = [*self_shape]
                    r_shape[sym_idx] = other_len
                    r.params = np.zeros(r_shape)
                    self_slice = [slice(0, i) for i in self_shape]
                    r.params[*self_slice] += self.params
                else:
                    r.params = self.params.copy()

                other_slice = [0] * len(self_shape)
                other_slice[sym_idx] = slice(0, other_len)
                r.params[*other_slice] -= other.params  # 减后面

            else:  # 新符号
                r.symbol = (other.symbol, *self.symbol)
                self_slice = [0] * len(self_shape)

                r.params = np.zeros((other_len, *self_shape))

                r.params[:, *self_slice] -= other.params
                r.params[0, ...] += self.params

        else:
            raise TypeError

        return r

    def __mul__(self, other):
        r = Formula()
        if isinstance(other, Formula):
            r.symbol = self.symbol
            flag = True
            for idx, s in enumerate(other.symbol):
                if s not in self.symbol:
                    flag = False
                    break

            if flag:  # 符号都相同,少于或等于当前符号
                mapping = self.get_mapping(other)
                other_dims = len(other.params.shape)
                other_p = other.params
                for idx, i in enumerate(mapping):
                    if i is None:
                        mapping[idx] = other_dims
                        other_dims += 1
                        other_p = np.expand_dims(other_p, axis=-1)
                other_p = other_p.transpose(mapping)  # 对齐符号顺序

                self_shape = self.params.shape
                other_shape = other_p.shape

                n_shape = list(self_shape)
                for idx, i in enumerate(other_shape):
                    n_shape[idx] += i - 1
                r.params = np.zeros(n_shape)

                for index in np.ndindex(other_shape):
                    if other_p[index] != 0:
                        kernel = other_p[index] * self.params
                        kernel_slice = [slice(i, i + self_shape[d]) for d, i in enumerate(index)]
                        r.params[*kernel_slice] += kernel

            else:
                r = other * self
        elif isinstance(other, (int, float)):
            r.params = self.params.copy()
            r.symbol = self.symbol
            r.params *= other
        elif isinstance(other, Variable):
            self_shape = self.params.shape
            other_len = other.params.shape[0]
            if other.symbol in self.symbol:  # 是已有的符号
                r.symbol = self.symbol
                sym_idx = self.symbol.index(other.symbol)

                r_shape = [*self_shape]
                offset = other_len - 1
                r_shape[sym_idx] += offset

                r.params = np.zeros(r_shape)
                n_slice = [slice(0, None)] * len(self_shape)
                n_slice[sym_idx] = slice(offset, None)

                r.params[*n_slice] += self.params

            else:  # 新符号
                r.symbol = (other.symbol, *self.symbol)
                r.params = np.zeros((other_len, *self_shape))
                for idx, i in enumerate(other.params):
                    if i:
                        r.params[idx, ...] += self.params

        else:
            raise TypeError

        return r

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            r = Variable(self.symbol)
            r.params = self.params / other
            return r
        else:
            raise TypeError

    def __pow__(self, power):
        if power > 0:
            if int(power) == power:
                power = int(power)
            if isinstance(power, int):
                power = bin(power)[:1:-1]
                temp = self  # 底数
                result = 1
                for i in power:
                    if i == "1":
                        result *= temp
                    temp *= temp
                return result
            else:
                raise TypeError
        elif power == 0:
            return 1


if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    a = x + y
    a = a * z
    print(a.symbol)
    print(a.params)
    # print("+")
    # b = y - x
    # print(b.symbol)
    # print(b.params)

    # c = a * x
    # print(c.symbol)
    # print(c.params)
