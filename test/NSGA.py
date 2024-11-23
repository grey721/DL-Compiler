import matplotlib.pyplot as plt
import numpy as np
import math


class NSGA:
    def __init__(self,
                 n_objective=2,
                 n_individuals=50,
                 n_generations=1000,
                 p_crossover=0.7,
                 p_mutation=0.4):

        self.n_generations = n_generations
        self.n_pop = n_individuals
        self.n_obj = n_objective
        self.pop = np.zeros((n_objective, n_individuals * 2))  # 后一半放父代， 前一半放子代
        self.gene = None
        self.init()

        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        pass

    def init(self):
        self.gene = np.random.rand(self.n_pop*2, self.n_obj)
        self.evaluate()

    def evolve(self):
        # init
        n_individuals = self.n_pop * 2
        rank = self.get_rank(self.n_pop)
        self.select(rank)

        # evolve
        for _ in range(self.n_generations):
            # 生成当代
            self.mutate()
            self.crossover()
            # 生成目标值
            self.evaluate()

            # 选这下一代的父代
            rank = self.get_rank(n_individuals)
            self.select(rank)

        # 可视化
        rank = self.get_rank(n_individuals)
        print(len(rank[0]))
        self.plot_2d_rank(rank, only_f0=True)

    def mutate(self):
        self.gene[0:self.n_pop, :] = self.gene[self.n_pop:None, :]
        for g_idx in range(self.n_pop):
            if np.random.rand() < self.p_mutation:
                position = np.random.randint(0, 2)
                new = np.random.uniform(-1, 1) + self.gene[g_idx][position]
                if new > 1:
                    new /= 2
                elif new < 0:
                    new *= -1
                self.gene[g_idx][position] = new
        pass

    def crossover(self):
        for c_idx in range(self.n_pop):
            if np.random.rand() < self.p_crossover:
                p_idx = int(np.round(self.n_pop * np.random.rand()))
                if p_idx == c_idx and c_idx == self.n_pop-1:
                    p_idx -= 1
                elif p_idx == c_idx and c_idx == 0:
                    p_idx -= 1
                elif p_idx == c_idx:
                    p_idx += 1

                if np.random.randint(0, 2):
                    temp = self.gene[c_idx][0]
                    self.gene[c_idx][0] = self.gene[p_idx][0]
                    self.gene[p_idx][0] = temp
                else:
                    temp = self.gene[c_idx][1]
                    self.gene[c_idx][1] = self.gene[p_idx][1]
                    self.gene[p_idx][1] = temp

    def evaluate(self):

        def z1(x):
            n = len(x)
            return (1 - np.exp(-sum((x - 1 / np.sqrt(n)) ** 2))) * 10

        def z2(x):
            n = len(x)
            return (1 - np.exp(-sum((x + 1 / np.sqrt(n)) ** 2))) * 10

        for idx, g in enumerate(self.gene[:self.n_pop, :]):
            self.pop[0][idx] = z1(g)
            self.pop[1][idx] = z2(g)

    def select(self, rank, p=4):
        s = []
        for n, fn in enumerate(rank):
            if len(s) + len(fn) <= self.n_pop:
                s.extend(fn)
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
        self.pop[:, self.n_pop:None] = self.pop[:, s]
        self.gene[:self.n_pop, :] = self.gene[s, :]

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
        pre_rank_idx = 0
        while current_rank:
            rank.append(current_rank)
            current_rank = []
            for i_idx in rank[pre_rank_idx]:
                for dot in dominate_set[i_idx]:
                    dominated_count[dot] -= 1
                    if dominated_count[dot] == 0:
                        current_rank.append(dot)
                    elif dominated_count[dot] < 0:
                        raise ValueError

            pre_rank_idx += 1

        return rank

    def plot_2d_rank(self, rank, only_f0=False):
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

    def get_distance(self, i1, i2):
        return np.sqrt(np.sum((self.pop[:, i1] - self.pop[:, i2]) ** 2))


if __name__ == "__main__":
    a = NSGA()
    a.evolve()
