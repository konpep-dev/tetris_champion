import numpy as np
import random
import copy
import time

INPUT_SIZE = 21
HIDDEN = [20, 16]
OUTPUT_SIZE = 1

POP_SIZE = 40
ELITE = 4
MUTATION_RATE = 0.25
MUTATION_STRENGTH = 0.2

class NeuralNet:
    def __init__(self):
        self.layers = []
        sizes = [INPUT_SIZE] + HIDDEN + [OUTPUT_SIZE]
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.layers.append({'w': w, 'b': b})

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = x @ layer['w'] + layer['b']
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x[0]

    def copy(self):
        return copy.deepcopy(self)

    def mutate(self, rate=None, strength=None):
        r = rate if rate is not None else MUTATION_RATE
        s = strength if strength is not None else MUTATION_STRENGTH
        for layer in self.layers:
            mask_w = np.random.random(layer['w'].shape) < r
            mask_b = np.random.random(layer['b'].shape) < r
            layer['w'] += np.random.randn(*layer['w'].shape) * s * mask_w
            layer['b'] += np.random.randn(*layer['b'].shape) * s * mask_b

    def save(self, path, gen=0, best_fit=0.0):
        data = {'version': 2, 'gen': gen, 'best_fit': best_fit}
        for i, layer in enumerate(self.layers):
            data[f'w{i}'] = layer['w']
            data[f'b{i}'] = layer['b']
        np.savez_compressed(path, **data)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        ver = int(data.get('version', 1))
        sizes = [INPUT_SIZE] + HIDDEN + [OUTPUT_SIZE]
        expected = sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))
        total = 0
        i = 0
        while f'w{i}' in data:
            total += data[f'w{i}'].size + data[f'b{i}'].size
            i += 1
        if ver < 2 or total != expected:
            print(f"[brain] model version {ver} incompatible with current (v2, {expected} params), starting fresh")
            return NeuralNet(), 0, 0.0
        net = object.__new__(NeuralNet)
        net.layers = []
        i = 0
        while f'w{i}' in data:
            net.layers.append({'w': data[f'w{i}'].copy(), 'b': data[f'b{i}'].copy()})
            i += 1
        gen = int(data.get('gen', 0))
        best_fit = float(data.get('best_fit', 0.0))
        return net, gen, best_fit

def crossover(a, b):
    child = NeuralNet()
    for i in range(len(a.layers)):
        mask = np.random.random(a.layers[i]['w'].shape) < 0.5
        child.layers[i]['w'] = np.where(mask, a.layers[i]['w'], b.layers[i]['w'])
        mask_b = np.random.random(a.layers[i]['b'].shape) < 0.5
        child.layers[i]['b'] = np.where(mask_b, a.layers[i]['b'], b.layers[i]['b'])
    return child

def tournament_select(pop, n=3):
    best = None
    best_f = -1e9
    for _ in range(n):
        i = random.randint(0, len(pop) - 1)
        if pop[i][0] > best_f:
            best_f = pop[i][0]
            best = pop[i][1]
    return best

class GeneticAlgorithm:
    def __init__(self):
        self.population = [NeuralNet() for _ in range(POP_SIZE)]
        self.fitness = [0.0] * POP_SIZE
        self.generation = 0
        self.best_fitness = 0.0
        self.best_net = None
        self.total_games = 0
        self.history = []
        self.gen_scores = []
        self.gen_steps = []
        self.gen_lines = []
        self.start_time = time.time()

    def record_fitness(self, idx, score, steps, lines):
        self.fitness[idx] = score * 100 + steps + lines * 1000
        self.gen_scores.append(score)
        self.gen_steps.append(steps)
        self.gen_lines.append(lines)
        self.total_games += 1

    def is_generation_done(self):
        return self.total_games >= POP_SIZE

    def evolve(self):
        paired = list(zip(self.fitness, self.population))
        paired.sort(key=lambda x: x[0], reverse=True)
        self.population, self.fitness = [p[1] for p in paired], [p[0] for p in paired]

        if self.fitness[0] > self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.best_net = self.population[0].copy()

        entry = {
            'gen': self.generation,
            'best_score': int(max(self.gen_scores) if self.gen_scores else 0),
            'avg_score': float(np.mean(self.gen_scores)) if self.gen_scores else 0,
            'best_steps': int(max(self.gen_steps) if self.gen_steps else 0),
            'avg_steps': float(np.mean(self.gen_steps)) if self.gen_steps else 0,
            'best_lines': int(max(self.gen_lines) if self.gen_lines else 0),
            'avg_lines': float(np.mean(self.gen_lines)) if self.gen_lines else 0,
            'best_fit': float(self.fitness[0]),
            'avg_fit': float(np.mean([f for f in self.fitness if f > 0])) if any(f > 0 for f in self.fitness) else 0,
        }
        self.history.append(entry)

        progress = self.generation / max(1, self.generation + 1)
        mr = MUTATION_RATE * (1.0 - 0.5 * progress)
        ms = MUTATION_STRENGTH * (1.0 - 0.3 * progress)

        new_pop = []
        for i in range(ELITE):
            new_pop.append(self.population[i].copy())
        while len(new_pop) < POP_SIZE:
            a = tournament_select(paired, 4)
            b = tournament_select(paired, 4)
            child = crossover(a, b)
            child.mutate(mr, ms)
            new_pop.append(child)
        self.population = new_pop
        self.fitness = [0.0] * POP_SIZE
        self.total_games = 0
        self.gen_scores = []
        self.gen_steps = []
        self.gen_lines = []
        self.generation += 1

    def save_best(self, path):
        if self.best_net is not None:
            self.best_net.save(path, self.generation, self.best_fitness)

    def load_best(self, path):
        net, gen, fit = NeuralNet.load(path)
        self.best_net = net
        self.generation = gen
        self.best_fitness = fit
        for i in range(min(POP_SIZE, 3)):
            self.population[i] = net.copy()

    def current_stats(self):
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        fit = self.fitness
        if self.gen_scores:
            return {
                'gen': self.generation,
                'best_fit': int(self.best_fitness) if self.best_fitness else 0,
                'curr_fit': int(max(fit)) if any(f > 0 for f in fit) else 0,
                'elapsed': f'{m}m {s}s',
                'progress': f'{self.total_games}/{POP_SIZE}',
                'best_score': int(max(self.gen_scores)),
                'avg_score': int(np.mean(self.gen_scores)),
                'best_steps': int(max(self.gen_steps)),
            }
        elif self.history:
            h = self.history[-1]
            return {
                'gen': self.generation,
                'best_fit': int(self.best_fitness) if self.best_fitness else 0,
                'curr_fit': int(max(fit)) if any(f > 0 for f in fit) else 0,
                'elapsed': f'{m}m {s}s',
                'progress': f'{self.total_games}/{POP_SIZE}',
                'best_score': h['best_score'],
                'avg_score': h['avg_score'],
                'best_steps': h['best_steps'],
            }
        else:
            return {
                'gen': self.generation,
                'best_fit': 0,
                'curr_fit': 0,
                'elapsed': f'{m}m {s}s',
                'progress': f'{self.total_games}/{POP_SIZE}',
                'best_score': 0,
                'avg_score': 0,
                'best_steps': 0,
            }
