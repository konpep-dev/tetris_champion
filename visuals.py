import pygame
import numpy as np
from game import TetrisGame, PIECE_COLORS

CELL = 18
BOARD_W = TetrisGame.WIDTH * CELL
BOARD_H = TetrisGame.HEIGHT * CELL
GAP = 10
MARGIN = 15
NN_W = 340

class Renderer:
    def __init__(self, n_games=4):
        cols = 2
        rows = (n_games + cols - 1) // cols
        self.game_area_w = cols * (BOARD_W + GAP) - GAP
        self.game_area_h = rows * (BOARD_H + GAP) - GAP
        self.stats_h = 290
        self.nn_h = self.game_area_h - self.stats_h - GAP

        self.win_w = MARGIN * 2 + self.game_area_w + GAP + NN_W
        self.win_h = MARGIN * 2 + self.game_area_h

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Tetris — Neural Network Evolution")
        self.font = pygame.font.SysFont('consolas', 12)
        self.font_big = pygame.font.SysFont('consolas', 14)
        self.font_sm = pygame.font.SysFont('consolas', 9)
        self.clock = pygame.time.Clock()
        self.cols = cols

        self.bg = (12, 12, 22)
        self.panel = (22, 22, 36)
        self.brd = (55, 55, 80)
        self.grid = (35, 35, 55)
        self.text = (195, 195, 215)
        self.acc = (0, 210, 255)
        self.acc2 = (255, 200, 50)
        self.ghost = (55, 55, 78)

    def draw_board(self, game, x, y, label):
        if not game.game_over and game.current_pos is not None:
            try:
                gpos, gshape = game.ghost_pos()
                for r in range(gshape.shape[0]):
                    for c in range(gshape.shape[1]):
                        if gshape[r, c]:
                            br, bc = gpos[0] + r, gpos[1] + c
                            if 0 <= br < game.HEIGHT:
                                px = x + bc * CELL
                                py = y + br * CELL
                                pygame.draw.rect(self.screen, self.ghost, (px+2, py+2, CELL-4, CELL-4))
            except:
                pass

        for r in range(game.HEIGHT):
            for c in range(game.WIDTH):
                px = x + c * CELL
                py = y + r * CELL
                if game.board[r, c]:
                    col = tuple(game.board_colors[r, c])
                    pygame.draw.rect(self.screen, col, (px+1, py+1, CELL-2, CELL-2))
                    hl = tuple(min(255, v+50) for v in col)
                    pygame.draw.rect(self.screen, hl, (px+1, py+1, CELL-6, 2))
                    pygame.draw.rect(self.screen, hl, (px+1, py+1, 2, CELL-6))
                else:
                    pygame.draw.rect(self.screen, self.grid, (px+1, py+1, CELL-2, CELL-2))

        if not game.game_over and game.current_pos is not None:
            shape = game.rotate_shape(game.current_name, game.current_rot)
            col = PIECE_COLORS[game.current_name]
            for r in range(shape.shape[0]):
                for c in range(shape.shape[1]):
                    if shape[r, c]:
                        br, bc = game.current_pos[0] + r, game.current_pos[1] + c
                        if 0 <= br < game.HEIGHT:
                            px = x + bc * CELL
                            py = y + br * CELL
                            pygame.draw.rect(self.screen, col, (px+1, py+1, CELL-2, CELL-2))
                            hl = tuple(min(255, v+50) for v in col)
                            pygame.draw.rect(self.screen, hl, (px+1, py+1, CELL-6, 2))
                            pygame.draw.rect(self.screen, hl, (px+1, py+1, 2, CELL-6))

        pygame.draw.rect(self.screen, self.brd, (x, y, BOARD_W, BOARD_H), 2)
        txt = self.font.render(label, True, self.text)
        self.screen.blit(txt, (x, y - 16))

    def draw_nn(self, net, x, y, w, h, title="Neural Network"):
        pygame.draw.rect(self.screen, self.panel, (x, y, w, h))
        pygame.draw.rect(self.screen, self.brd, (x, y, w, h), 1)
        t = self.font_big.render(title, True, self.acc)
        self.screen.blit(t, (x + 10, y + 8))

        sizes = [21, 20, 16, 1]
        n_lyr = len(sizes)
        pad_x, pad_y = 20, 28
        dw = w - pad_x * 2
        dh = h - pad_y * 2
        max_n = max(sizes)
        lx = [pad_x + i * dw // (n_lyr - 1) for i in range(n_lyr)]

        for li in range(n_lyr - 1):
            for i in range(sizes[li]):
                for j in range(sizes[li + 1]):
                    wgt = net.layers[li]['w'][i, j]
                    if abs(wgt) < 0.05:
                        continue
                    alpha = min(1.0, abs(wgt) * 1.8)
                    x1 = x + lx[li]
                    y1 = y + pad_y + (i + 0.5) * (dh - 8) // max_n + 4
                    x2 = x + lx[li + 1]
                    y2 = y + pad_y + (j + 0.5) * (dh - 8) // max_n + 4
                    if wgt > 0:
                        c = (0, int(alpha * 200), int(alpha * 255))
                    else:
                        c = (int(alpha * 255), int(alpha * 80), int(alpha * 80))
                    pygame.draw.line(self.screen, c, (int(x1), int(y1)), (int(x2), int(y2)), 1)

        for li in range(n_lyr):
            for nd in range(sizes[li]):
                cx = x + lx[li]
                cy = y + pad_y + (nd + 0.5) * (dh - 8) // max_n + 4
                r = 4
                pygame.draw.circle(self.screen, (90, 90, 130), (int(cx), int(cy)), r)
                pygame.draw.circle(self.screen, (150, 150, 190), (int(cx), int(cy)), r, 1)

        labels = ["Input\n21", "H1\n20", "H2\n16", "Out\n1"]
        for li in range(n_lyr):
            txt = self.font_sm.render(labels[li], True, (140, 140, 170))
            cx = x + lx[li]
            self.screen.blit(txt, (cx - txt.get_width() // 2, y + h - 16))

    def draw_stats(self, stats, games, x, y, w, speed):
        pygame.draw.rect(self.screen, self.panel, (x, y, w, self.stats_h))
        pygame.draw.rect(self.screen, self.brd, (x, y, w, self.stats_h), 1)

        s = stats
        lines = [
            f"Generation:    {s['gen']}",
            f"Best Fit Ever: {s['best_fit']}",
            f"Curr Gen Best: {s['curr_fit']}",
            f"Best Score:    {s['best_score']}",
            f"Avg Score:     {s['avg_score']}",
            f"Best Steps:    {s['best_steps']}",
            f"Elapsed:       {s['elapsed']}",
            f"Progress:      {s['progress']}",
            "",
        ]
        for i, g in enumerate(games):
            status = "DONE" if g.game_over else "PLAY"
            lines.append(f"#{i+1}: Sc:{int(g.score):4d}  St:{g.steps:3d}  Ln:{g.lines:3d}  [{status}]")
        lines.append("")
        lines.append(f"Speed: {speed}x  S:save  SPC:pause  UP/DN  ESC:exit")

        yy = y + 8
        for line in lines:
            if line.startswith("#") and len(line) < 5:
                pass
            if "PLAY" in line and "[" in line:
                col = self.acc2
            elif "DONE" in line:
                col = (100, 100, 120)
            elif line.startswith("Generation"):
                col = self.acc
            elif "Speed" in line or "SPC" in line:
                col = (90, 90, 120)
            else:
                col = self.text
            txt = self.font.render(line, True, col)
            self.screen.blit(txt, (x + 10, yy))
            yy += 16

    def render(self, agents, stats, speed=2):
        self.screen.fill(self.bg)

        ox = MARGIN
        oy = MARGIN
        nn_x = ox + self.game_area_w + GAP
        nn_y = oy
        nn_h = self.game_area_h - self.stats_h - GAP
        sx = nn_x
        sy = nn_y + nn_h + GAP

        for idx, agent in enumerate(agents):
            r = idx // self.cols
            c = idx % self.cols
            gx = ox + c * (BOARD_W + GAP)
            gy = oy + r * (BOARD_H + GAP)
            lbl = f"#{idx+1}  Score:{int(agent.game.score):4d}  Steps:{agent.game.steps}"
            if agent.game.game_over:
                lbl += "  DONE"
            self.draw_board(agent.game, gx, gy, lbl)

        if agents and agents[0].brain:
            self.draw_nn(agents[0].brain, nn_x, nn_y, NN_W, nn_h, "Network #1")

        self.draw_stats(stats, [a.game for a in agents], sx, sy, NN_W, speed)
        pygame.display.flip()
        self.clock.tick(30)
