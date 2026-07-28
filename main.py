import pygame
import sys
import os
import numpy as np
from game import TetrisGame
from brain import GeneticAlgorithm, NeuralNet, POP_SIZE
from visuals import Renderer
from tracker import Tracker

SAVE_PATH = "tetris_champion.npz"

class Agent:
    def __init__(self, brain, idx):
        self.brain = brain
        self.game = TetrisGame()
        self.idx = idx
        self.done = False
        self.recorded = False

    def reset(self, brain):
        self.brain = brain
        self.game = TetrisGame()
        self.done = False
        self.recorded = False

    def step(self):
        if self.done:
            return
        placements = self.game.get_placements()
        if not placements:
            self.game.game_over = True
            self.done = True
            return
        scores = np.array([self.brain.forward(feat) for _, _, feat in placements])
        rot, col, _ = placements[np.argmax(scores)]
        self.game.execute_ai(rot, col)
        if self.game.game_over:
            self.done = True

class Sim:
    def __init__(self):
        self.ga = GeneticAlgorithm()
        self.tracker = Tracker()
        self.n = 4
        self.agents = [Agent(self.ga.population[i], i) for i in range(self.n)]
        self.next_idx = self.n
        self.paused = False
        self.speed = 2

    def replace_agent(self, agent):
        if self.next_idx >= POP_SIZE:
            return False
        agent.reset(self.ga.population[self.next_idx])
        agent.idx = self.next_idx
        self.next_idx += 1
        return True

    def update(self):
        for _ in range(self.speed):
            for a in self.agents:
                a.step()
        for a in self.agents:
            if a.done and not a.recorded:
                self.ga.record_fitness(a.idx, a.game.score, a.game.steps, a.game.lines)
                a.recorded = True
                self.replace_agent(a)
        if self.ga.is_generation_done():
            self.ga.evolve()
            self.next_idx = self.n
            for i in range(self.n):
                if i < POP_SIZE:
                    self.agents[i].reset(self.ga.population[i])
                    self.agents[i].idx = i
            last = self.ga.history[-1]
            self.tracker.log(last)
            if self.ga.generation % 5 == 0:
                self.ga.save_best(SAVE_PATH)

def main():
    sim = Sim()

    if os.path.exists(SAVE_PATH):
        sim.ga.load_best(SAVE_PATH)
        sim.next_idx = sim.n
        for i in range(sim.n):
            if i < POP_SIZE:
                sim.agents[i].reset(sim.ga.population[i])
                sim.agents[i].idx = i

    vis = Renderer(sim.n)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_UP:
                    sim.speed = min(20, sim.speed + 1)
                elif event.key == pygame.K_DOWN:
                    sim.speed = max(1, sim.speed - 1)
                elif event.key == pygame.K_s:
                    sim.ga.save_best(SAVE_PATH)
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not sim.paused:
            sim.update()
        stats = sim.ga.current_stats()
        vis.render(sim.agents, stats, sim.speed)

    sim.tracker.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
