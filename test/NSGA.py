import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class Individual:
    def __init__(self):
        self.Rank = 0
        self.objective = None
        self.gene = None  # [x ,y ...]
        self.DominationSet = set()
        self.DominatedCount = None
        self.CrowdingDistance = None

    def __lt__(self, other):
        if isinstance(other, Individual):
            return all(i < j for i, j in zip(self.objective, other.objective))
        else:
            return False

    def __repr__(self):
        return (f"F{self.Rank}\n"
                f"Obj:{self.objective.tolist()}\n"
                f"Gene:{self.gene.tolist()}\n"
                f"DominatedCount:{self.DominatedCount}\n"
                f"DominationSet:{self.DominationSet}")

    def random_value(self, n_gene):
        gene = []
        for i in range(n_gene):
            gene.append(np.random.rand())
        self.gene = np.array(gene)

    def evaluate(self, fitness_funcs):
        objective = [f(self.gene) for f in fitness_funcs]
        self.objective = np.array(objective)
        return self.objective


class FuncsType(Enum):
    OBJ = 0
    mutate = 1
    crossover = 2
    p_init = 3


class NSGA2:
    def __init__(self,
                 n_populations=50,
                 n_generations=10000,
                 p_crossover=0.7,
                 p_mutation=0.4):

        self.n_pop = n_populations
        self.n_gen = n_generations

        self.p_crossover = p_crossover
        self.p_mutation = p_mutation

        self.funcs = {
            FuncsType.OBJ: [],
            FuncsType.mutate: [],
        }

        self.pop = self.initialize_population()

    def non_dominated_sorting(self):
        current_rank = []
        for idx in range(len(self.pop)):
            dominated_count = 0
            for jdx in range(len(self.pop)):
                if idx == jdx:
                    continue
                elif self.pop[jdx] < self.pop[idx]:
                    dominated_count += 1
                    self.pop[jdx].DominationSet.add(idx)

            self.pop[idx].DominatedCount = dominated_count

            if dominated_count == 0:
                self.pop[idx].Rank = 1
                current_rank.append(idx)

        rank = []
        pre_rank_idx = 0
        while current_rank:
            rank.append(current_rank)
            current_rank = []
            for n_idx in rank[pre_rank_idx]:
                for dot in self.pop[n_idx].DominationSet:
                    self.pop[dot].Rank += 1
                    if self.pop[dot].Rank == self.pop[dot].DominatedCount:
                        self.pop[dot].Rank = pre_rank_idx + 2
                        current_rank.append(dot)
                    elif self.pop[dot].Rank >= self.pop[dot].DominatedCount:
                        raise ValueError

            pre_rank_idx += 1

        return rank

    def select(self, offspring):
        s_pop = []
        for i in offspring:
            if self.n_pop >= len(s_pop) + len(i):
                s_pop.extend(i)
            else:
                # 判断i的拥挤距离并加入s_pop
                break
        return s_pop

    def evolve(self):
        if len(self.funcs[FuncsType.OBJ]) == 0:
            print("No objective function")
            return

        self.evaluate_population()
        rank = self.non_dominated_sorting()

        self.plot_2d_rank(rank)

    def mutate(self, pop):
        for idx in range(len(pop)):
            for g_idx in pop[idx].gene:
                if np.random.rand() < self.p_mutation:
                    pop[idx].gene[g_idx] = 1 - pop[idx].gene[g_idx]
        return pop

    def crossover(self):
        pass

    def initialize_population(self) -> list[Individual]:
        pop = []
        for _ in range(self.n_pop):
            pop.append(Individual())
            pop[-1].random_value(2)
        return pop

    def evaluate_population(self):
        for idx in range(len(self.pop)):
            self.pop[idx].evaluate(self.funcs[FuncsType.OBJ])

    def plot_2d_rank(self, rank):
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta', 'cyan', 'black', 'lime', 'brown', 'pink',
                  'navy', 'gold']

        for f in range(len(rank)):
            x = []
            y = []
            print(rank[f], colors[f])
            for i in rank[f]:
                x.append(self.pop[i].objective[0])
                y.append(self.pop[i].objective[1])

            plt.scatter(x, y, color=colors[f])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def register_funcs(self, f_type):
        def callback(impl):
            self.funcs[f_type].append(impl)
        return callback


if __name__ == "__main__":
    al = NSGA2()

    @al.register_funcs(FuncsType.OBJ)
    def z1(x):
        n = len(x)
        return 1 - np.exp(-sum((x-1/np.sqrt(n))**2))

    @al.register_funcs(FuncsType.OBJ)
    def z2(x):
        n = len(x)
        return 1 - np.exp(-sum((x+1/np.sqrt(n))**2))

    al.evolve()
