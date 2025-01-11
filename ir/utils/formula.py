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
                raise NotImplementedError

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
                raise NotImplementedError

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
                raise NotImplementedError

        elif isinstance(other, (int, float)):
            r.params = other * self.params

        else:
            raise TypeError

        return r

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
        result = 1
        if power == int(power):
            power = int(power)
            for i in range(power):
                result = self * result
            return result
        else:
            raise TypeError

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
