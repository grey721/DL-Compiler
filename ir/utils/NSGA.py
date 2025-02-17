import math
import random
import matplotlib.pyplot as plt
import numpy as np

from backend.chip.Ada300 import *
from ir.graph.Graph_IR import *
from ir.utils.utils import *


def random_sub_range(low, high, exclude=(), random_order=False):
    assert low != high, f"{low} == {high}"
    num1 = np.random.randint(low, high)
    num2 = np.random.randint(low, high)
    while (num2 == num1) or (num1 in exclude) or (num2 in exclude):
        num1 = np.random.randint(low, high)
        num2 = np.random.randint(low, high)

    if random_order:
        return num1, num2

    if num1 > num2:
        return num2, num1
    else:
        return num1, num2


class Task:
    def __init__(self, sequence, shape):
        self.sequence = sequence
        self.n_cim = shape[1]
        self.row = shape[2]
        self.col = shape[3]
        self.bitmap = [0] * shape[2]

    def select_all(self):
        for i in range(self.row):
            self.bitmap[i] = generate_binary_number(self.col)

    def randomly_draws(self, row=None, col=None):
        assert any(self.bitmap)
        high = 1 << self.col
        if row is not None:
            sub = slice(row, row + 1)
        else:
            sub = slice(0, None)
        if col is not None:
            mask = 1 << col
        else:
            mask = generate_binary_number(self.col)

        result = []
        while (not any(result)) and any(self.bitmap[sub]):
            result = []
            for i, mapping in enumerate(self.bitmap[sub]):
                if mapping:
                    selection = random.randint(0, high)
                    output = mapping & mask & selection
                    result.append(output)
                    # 从该副本中删去
                    if row:
                        assert self.bitmap[row+i] >= output
                        self.bitmap[row+i] -= output
                    else:
                        assert self.bitmap[i] >= output
                        self.bitmap[i] -= output
                else:
                    result.append(0)

        return result if result else [0]

    def add_task(self, t, row=None):
        try:
            if row is None:
                for i, other in enumerate(t):
                    self.bitmap[i] += other
            else:
                self.bitmap[row] += t[0]
        except IndexError:
            print(t)
            print(len(t))
            print(len(self.bitmap))
            raise IndexError

    def destroy(self, num_split):
        # 生成 n 个随机小数
        random_numbers = np.random.rand(num_split)

        # 计算这些随机小数的总和
        total_sum = np.sum(random_numbers)
        # 将每个随机小数除以总和，得到新的小数
        random_numbers = np.round(random_numbers / total_sum * self.row)

        bias = self.row - np.sum(random_numbers)
        idx = np.random.randint(0, num_split)
        random_numbers[idx] += bias
        while random_numbers[idx] < 0:
            bias = random_numbers[idx]
            random_numbers[idx] = 0
            idx = np.random.randint(0, num_split)
            random_numbers[idx] += bias

        result = []
        pre_row = 0
        for layer in random_numbers:
            temp = []
            next_row = int(pre_row + layer)
            temp.extend([0] * pre_row)
            temp.extend(self.bitmap[slice(pre_row, next_row)])
            temp.extend([0] * (self.row - next_row))
            pre_row = next_row
            result.append(temp)
            assert len(result[-1]) == self.row, f"{len(result[-1])} != {self.row}, {random_numbers}"
        return result


class NSGA:
    def __init__(self, npu_graph: GraphIR,
                 chip,
                 n_individuals=50,
                 n_generations=100,
                 # p_crossover=0.7,
                 p_mutation=0.4):

        self.n_generations = n_generations
        self.n_pop = n_individuals
        self.n_obj = 2  # 优化目标数
        self.pop = np.zeros((self.n_obj, n_individuals * 2))  # 后一半放父代， 前一半放子代
        self.gene = None
        # chip info
        self.n_core_per_chip = chip.num_core
        self.n_cim_per_core = chip.num_cim
        self.cim_h = chip.CIM.H
        self.cim_w = chip.CIM.W

        self.p_mutation = p_mutation

        # 基因初始化
        mvm_op_idx = npu_graph.mvm_op_idx
        n_gene = len(mvm_op_idx)
        self.n_gene = n_gene
        self.pattern = [(npu_graph.AllOps[idx].NpuOpConvOp.WeightValue.shape[0],
                         npu_graph.AllOps[idx].NpuOpConvOp.WeightValue.shape[1],
                         npu_graph.AllOps[idx].NpuOpConvOp.OutputShape[0].H,
                         npu_graph.AllOps[idx].NpuOpConvOp.OutputShape[0].W) for idx in mvm_op_idx]

        self.sequence_boundary = npu_graph.num_mvm_op_unit
        self.gene = [[]]
        sequence = 0
        for shape in self.pattern:
            self.gene[0].append([])
            for _ in range(shape[0]):
                self.gene[0][-1].append([])
                task = Task(sequence, shape)
                task.select_all()
                self.gene[0][-1][-1].append(task)
                sequence += 1
        self.gene = self.gene * n_individuals * 2

        # 生成目标值
        self.evaluate()
        # 保存到后一半
        self.select([[idx for idx in range(self.n_pop)]])
        self.evolve()

    def evolve(self):
        # init
        n_individuals = self.n_pop * 2

        # evolve
        print("Begin")
        for _ in range(self.n_generations):
            # print(f"第{_}代")
            # 选这一代的父代
            rank = self.get_rank(n_individuals)
            self.select(rank)
            for i in range(0, self.n_pop, 2):
                # 生成当代
                parent_idx = self.n_pop + i
                child1, child2 = self.crossover(parent_idx, parent_idx+1)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                self.gene[i] = child1
                self.gene[i+1] = child2

            # 生成目标值
            self.evaluate()

        # 可视化
        # self.normalize()
        rank = self.get_rank(n_individuals)  # n_individuals    self.n_pop
        self.plot_2d_rank(rank)
        for ranke in rank:
            for idx in ranke:
                print(f"第{idx}个个体")
                print(f"芯片数：{self.pop[0][idx]}")
                print(f"浪费率：{self.pop[1][idx]}")
                # for op in self.gene[idx]:
                #     for c in range(len(op)):
                #         print(f"第{c}层，复制{len(op[c])}份")
        # self.plot_3d_rank(rank, only_f0=False)

    def mutate(self, indivi):
        individual = indivi[:]
        if np.random.rand() < self.p_mutation:
            position = np.random.randint(0, self.n_gene)  # 算子位置
            c_position = np.random.randint(0, len(individual[position]))
            num_t = len(individual[position][c_position])
            if num_t > 1:
                kind = np.random.randint(0, 4)
            else:  # 共4种变异函数
                kind = np.random.randint(0, 2)

            if kind == 0:
                # 增加复制数
                n_task = Task(np.random.randint(self.sequence_boundary), self.pattern[position])
                delete_list = []
                for idx, copy in enumerate(individual[position][c_position]):
                    n_task.add_task(copy.randomly_draws())
                    if not any(copy.bitmap):
                        delete_list.append(idx)
                if delete_list:
                    for idx in delete_list[::-1]:
                        del individual[position][c_position][idx]
                individual[position][c_position].append(n_task)

            elif kind == 1:
                # 交换顺序
                if np.random.rand() < 0.05:
                    other_position = np.random.randint(0, self.n_gene)
                    oc_position = np.random.randint(0, len(individual[other_position]))
                    num_ot = len(individual[other_position][oc_position])
                    ot_idx = np.random.randint(0, num_ot)

                    t_idx = np.random.randint(0, num_t)
                    individual[position][c_position][t_idx].sequence = individual[other_position][oc_position][ot_idx].sequence
                else:
                    t_idx = np.random.randint(0, num_t)
                    individual[position][c_position][t_idx].sequence = np.random.randint(0, self.sequence_boundary)

            elif kind == 2:
                # num_t > 1:
                # 减少复制数
                t_idx = np.random.randint(0, num_t)
                split = individual[position][c_position][t_idx].destroy(num_t-1)
                del individual[position][c_position][t_idx]
                for i, j in zip(individual[position][c_position], split):
                    i.add_task(j)

            elif kind == 3:
                # num_t > 1
                # 传递数据
                t1, t2 = random_sub_range(0, num_t, random_order=True)
                if np.random.rand() < 0.5:
                    n_row = np.random.randint(0, individual[position][c_position][t2].row)
                    while individual[position][c_position][t1].bitmap[n_row] == 0:
                        n_row = np.random.randint(0, individual[position][c_position][t2].row)
                    addition = individual[position][c_position][t1].randomly_draws(row=n_row)
                    individual[position][c_position][t2].add_task(addition, row=n_row)
                else:
                    addition = individual[position][c_position][t1].randomly_draws()
                    individual[position][c_position][t2].add_task(addition)

                if not any(individual[position][c_position][t1].bitmap):
                    del individual[position][c_position][t1]

        return individual

    def crossover(self, parent1, parent2):
        start, end = random_sub_range(0, self.n_gene)
        child1 = self.gene[parent1][:]
        child2 = self.gene[parent2][:]
        child1[start: end] = self.gene[parent2][start: end]
        child2[start: end] = self.gene[parent1][start: end]
        return child1, child2

    def evaluate(self):

        for idx, g in enumerate(self.gene[0:self.n_pop]):
            chip = []
            ag = 0
            for op in g:
                for c_layer in op:
                    for copy in c_layer:
                        # 找到插入位置
                        for i in range(len(chip)):
                            if copy.sequence < chip[i].sequence:
                                chip.insert(i, copy)
                                ag += copy.n_cim
                                break
                        else:
                            chip.append(copy)
                            ag += copy.n_cim
            num_chip = ag / self.n_cim_per_core / self.n_core_per_chip
            self.pop[0][idx] = num_chip
            self.pop[1][idx] = math.ceil(num_chip) - num_chip

    def select(self, rank, p=4):
        s = []
        for n, fn in enumerate(rank):
            if len(s) + len(fn) <= self.n_pop:
                s.extend(fn)
            elif len(s) + len(fn) == self.n_pop:
                break
            else:
                # 挑选加入s
                n_refer = math.comb(self.n_obj + p - 1, p)
                # TODO temp
                t = 0
                while len(s) < self.n_pop:
                    s.extend(rank[n + t])
                    t += 1
                if len(s) > self.n_pop:
                    t = len(s) - self.n_pop
                    for _ in range(t):
                        del s[-1]
                break
        # 保存父代到后一半
        self.pop[:, self.n_pop:None] = self.pop[:, s]
        for idx, i_idx in enumerate(s, self.n_pop):
            self.gene[idx] = self.gene[i_idx]

    def get_rank(self, n_individuals):
        dominate_set = [[] for _ in range(n_individuals)]
        dominated_count = [0] * n_individuals
        current_rank = []
        for i_idx in range(n_individuals):
            for j_idx in range(n_individuals):
                if i_idx == j_idx:
                    continue
                elif np.all(self.pop[:, j_idx] < self.pop[:, i_idx]):
                    # j 支配 i
                    dominated_count[i_idx] += 1
                    dominate_set[j_idx].append(i_idx)

            if dominated_count[i_idx] == 0:
                current_rank.append(i_idx)

        rank = []
        while current_rank:
            rank.append(current_rank)
            current_rank = []
            for i_idx in rank[-1]:
                for dot in dominate_set[i_idx]:
                    dominated_count[dot] -= 1
                    if dominated_count[dot] == 0:
                        current_rank.append(dot)
                    elif dominated_count[dot] < 0:
                        raise ValueError

        return rank

    def normalize(self):
        row_min_values = np.min(self.pop, axis=1)
        # [:, np.newaxis]不会复制数据，而Array.reshape会复制数据
        f = self.pop - row_min_values[:, np.newaxis]
        self.pop = f
        # f = f / np.linalg.norm(f, axis=0)
        print(np.argmin(f, axis=1))
        return np.argmin(f, axis=1)

    def plot_2d_rank(self, rank, only_f0=False):
        print(f"非支配等级为F1的个数:{len(rank[0])}")
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta', 'cyan', 'black', 'lime', 'brown', 'pink',
                  'navy', 'gold']
        if only_f0:
            x = self.pop[0, rank[0]]
            y = self.pop[1, rank[0]]
            plt.scatter(x, y, color=colors[0])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        else:
            for n, fn in enumerate(rank):
                x = self.pop[0, fn]
                y = self.pop[1, fn]
                print(fn, colors[n])
                plt.scatter(x, y, color=colors[n])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

    def plot_3d_rank(self, rank, only_f0=False):
        print(f"非支配等级为F1的个数:{len(rank[0])}")

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta', 'cyan', 'black', 'lime', 'brown', 'pink',
                  'navy', 'gold']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if only_f0:
            x = self.pop[0, rank[0]]
            y = self.pop[1, rank[0]]
            z = self.pop[2, rank[0]]

            ax.scatter(x, y, z, c=colors[0])

        else:
            for n, fn in enumerate(rank):
                x = self.pop[0, rank[n]]
                y = self.pop[1, rank[n]]
                z = self.pop[2, rank[n]]
                ax.scatter(x, y, z, c=colors[n])
                print(fn, colors[n])
        ax.set_xlabel('Delay')
        ax.set_ylabel('1/T')
        ax.set_zlabel('Power')
        plt.show()

    def get_distance(self, i1, i2):
        return np.sqrt(np.sum((self.pop[:, i1] - self.pop[:, i2]) ** 2))

    def get_modulus(self, idx):
        return np.sqrt(np.sum(self.pop[idx] ** 2))


if __name__ == "__main__":
    import pickle
    with open('output/yolov5s_1/npu_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    # a = NSGA(graph, 16, )

    a = [0, 1]
    a = NSGA(graph, Ada300)
