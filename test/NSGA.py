import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ir.graph.Graph_IR import *
import numpy as np
import math


def random_sub_range(low, high, exclude=()):
    num1 = np.random.randint(low, high)
    num2 = np.random.randint(low, high)
    while (num2 == num1) or (num1 in exclude) or (num2 in exclude):
        num1 = np.random.randint(low, high)
        num2 = np.random.randint(low, high)

    if num1 > num2:
        return num2, num1
    else:
        return num1, num2


def count_binary_ones(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def find_hyperplane_intercepts(points):
    """
    计算n-1维超平面在n维空间中的截距。

    参数:
    points -- 一个形状为(n, n)的NumPy数组，其中每行代表一个点的坐标。

    返回:
    intercepts -- 一个数组，包含每个轴上的截距。
    """
    # 确保点的数量等于维度
    assert points.shape[0] == points.shape[1], "点的数量必须等于维度"

    # 构建矩阵A和向量b
    A = points[:, :-1]
    b = points[:, -1]

    # 使用最小二乘法求解法向量a
    a, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 计算常数d
    d = np.mean(np.dot(a, points.T))

    # 计算截距
    intercepts = d / a

    return intercepts


class NSGA:
    def __init__(self, npu_graph: GraphIR, n_core,
                 n_individuals=50,
                 n_generations=2000,
                 # p_crossover=0.7,
                 p_mutation=0.4):

        self.n_generations = n_generations
        self.n_pop = n_individuals
        self.n_obj = 3
        self.pop = np.zeros((self.n_obj, n_individuals * 2))  # 后一半放父代， 前一半放子代
        self.gene = None

        self.bound = (1, 1 << n_core)

        # self.p_crossover = p_crossover
        self.p_mutation = p_mutation

        # 基因初始化
        n_gene = len(npu_graph.AllOps)
        self.n_gene = n_gene
        self.gene = np.random.randint(self.bound[0], self.bound[1], size=(self.n_pop * 2, n_gene))
        # 生成目标值
        self.evaluate()
        # 保存到后一半
        self.select([[idx for idx in range(self.n_pop)]])
        # 生成当代
        self.crossover()
        self.mutate()
        # 生成目标值
        self.evaluate()

        self.evolve()

    def evolve(self):
        # init
        n_individuals = self.n_pop * 2

        # evolve
        for _ in range(self.n_generations):
            # 选这一代的父代
            rank = self.get_rank(n_individuals)
            self.select(rank)
            # self.plot_2d_rank(rank, only_f0=True)
            # 生成当代
            self.crossover()
            self.mutate()
            # 生成目标值
            self.evaluate()

        # 可视化
        # self.normalize()
        rank = self.get_rank(n_individuals)  # n_individuals    self.n_pop
        self.plot_3d_rank(rank, only_f0=False)

    def mutate(self):
        for g_idx in range(self.n_pop):
            if np.random.rand() < self.p_mutation:
                position = np.random.randint(0, self.n_gene)
                new = np.random.randint(self.bound[0], self.bound[1]) & self.gene[g_idx][position]
                if new == 0:
                    new = 1

                self.gene[g_idx][position] = new

    def crossover(self):
        for g_idx in range(self.n_pop):
            p_idx = np.random.randint(0, self.n_pop)
            while p_idx == g_idx:
                p_idx = np.random.randint(0, self.n_pop)

            start, end = random_sub_range(0, self.n_gene)
            temp = self.gene[g_idx, start: end]
            self.gene[g_idx, start: end] = self.gene[p_idx, start: end]
            self.gene[p_idx, start: end] = temp

    def evaluate(self):

        def delay(x):
            time = 0
            for i in x:
                n_copy = count_binary_ones(i)
                time += 1 / n_copy + n_copy * 0.01
            return time

        def throughput(x):
            output = 0
            for n, i in enumerate(x):
                output += count_binary_ones(i) * n
            return 100000 / output

        def power(x):
            load_num = 0
            for i in x:
                load_num += count_binary_ones(i)
            return load_num

        for idx, g in enumerate(self.gene[0:self.n_pop, :]):
            self.pop[0][idx] = delay(g)
            self.pop[1][idx] = throughput(g)
            self.pop[2][idx] = power(g)

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
                while len(s) < 50:
                    s.extend(rank[n+t])
                    t += 1
                if len(s) > 50:
                    t = len(s) - 50
                    for _ in range(t):
                        del s[-1]
                break
        # 保存父代到后一半
        self.pop[:, self.n_pop:None] = self.pop[:, s]
        self.gene[self.n_pop:None, :] = self.gene[s, :]
        # 复制父代基因到前一半，准备生成子代基因
        self.gene[0:self.n_pop, :] = self.gene[self.n_pop:None, :]

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
    with open('output/yolov5s/npu_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    a = NSGA(graph, 16)

