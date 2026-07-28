import pygame
import sys
import numpy as np
from game import TetrisGame, PIECE_COLORS, SHAPES
from brain import NeuralNet

CELL = 30
BOARD_W = TetrisGame.WIDTH * CELL
BOARD_H = TetrisGame.HEIGHT * CELL
MARGIN = 20
INFO_W = 280
WIN_W = MARGIN * 2 + BOARD_W + INFO_W
WIN_H = MARGIN * 2 + BOARD_H

BG = (12, 12, 22); PANEL = (22, 22, 36); BORDER = (55, 55, 80)
GRID = (35, 35, 55); TEXT = (195, 195, 215); GHOST = (55, 55, 78)

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Tetris AI — Normal Speed")
font = pygame.font.SysFont('consolas', 13)
font_big = pygame.font.SysFont('consolas', 16)
clock = pygame.time.Clock()

net, gen, _ = NeuralNet.load("tetris_champion.npz")
g = TetrisGame()
pause = False
target_rot = None
target_col = None
thinking = False
dropping = False

def decide():
    global target_rot, target_col, thinking
    placements = g.get_placements()
    if not placements:
        g.game_over = True
        return
    scores = [net.forward(feat) for _, _, feat in placements]
    rot, col, _ = placements[np.argmax(scores)]
    target_rot = rot
    target_col = col
    thinking = False

def step_ai():
    global target_rot, target_col, thinking, dropping
    if thinking:
        decide()
        return
    if target_rot is None or target_col is None:
        thinking = True
        return
    if g.current_rot != target_rot:
        g.rotate()
        return
    if g.current_pos[1] < target_col:
        g.move_right()
        return
    if g.current_pos[1] > target_col:
        g.move_left()
        return
    if not dropping:
        dropping = True
    if g.move_down():
        return
    dropping = False
    target_rot = None
    target_col = None
    thinking = True

def draw_board():
    for r in range(g.HEIGHT):
        for c in range(g.WIDTH):
            px = MARGIN + c * CELL; py = MARGIN + r * CELL
            if g.board[r, c]:
                col = tuple(g.board_colors[r, c])
                pygame.draw.rect(screen, col, (px+1, py+1, CELL-2, CELL-2))
                hl = tuple(min(255, v+50) for v in col)
                pygame.draw.rect(screen, hl, (px+1, py+1, CELL-6, 2))
                pygame.draw.rect(screen, hl, (px+1, py+1, 2, CELL-6))
            else:
                pygame.draw.rect(screen, GRID, (px+1, py+1, CELL-2, CELL-2))
    if g.game_over or g.current_pos is None:
        pygame.draw.rect(screen, BORDER, (MARGIN, MARGIN, BOARD_W, BOARD_H), 2)
        return
    try:
        gpos, gshape = g.ghost_pos()
        for r in range(gshape.shape[0]):
            for c in range(gshape.shape[1]):
                if gshape[r, c]:
                    br, bc = gpos[0] + r, gpos[1] + c
                    if 0 <= br < g.HEIGHT:
                        px = MARGIN + bc * CELL; py = MARGIN + br * CELL
                        pygame.draw.rect(screen, GHOST, (px+2, py+2, CELL-4, CELL-4))
    except:
        pass
    shape = g.rotate_shape(g.current_name, g.current_rot)
    col = PIECE_COLORS[g.current_name]
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r, c]:
                br, bc = g.current_pos[0] + r, g.current_pos[1] + c
                if 0 <= br < g.HEIGHT:
                    px = MARGIN + bc * CELL; py = MARGIN + br * CELL
                    pygame.draw.rect(screen, col, (px+1, py+1, CELL-2, CELL-2))
                    hl = tuple(min(255, v+50) for v in col)
                    pygame.draw.rect(screen, hl, (px+1, py+1, CELL-6, 2))
                    pygame.draw.rect(screen, hl, (px+1, py+1, 2, CELL-6))
    pygame.draw.rect(screen, BORDER, (MARGIN, MARGIN, BOARD_W, BOARD_H), 2)

def draw_info():
    ix = MARGIN + BOARD_W + 20; iy = MARGIN + 10; iw = INFO_W - 20
    pygame.draw.rect(screen, PANEL, (ix-10, iy-10, iw+20, WIN_H - MARGIN*2))
    pygame.draw.rect(screen, BORDER, (ix-10, iy-10, iw+20, WIN_H - MARGIN*2), 1)
    info = [
        ("Champion AI", (0, 210, 255)), ("", TEXT),
        (f"Generation: {gen}", TEXT), ("", TEXT),
        (f"Score: {g.score}", (255, 220, 50)),
        (f"Lines: {g.lines}", TEXT), (f"Pieces: {g.steps}", TEXT),
        ("", TEXT), ("Next:", TEXT),
    ]
    yy = iy
    for text, c in info:
        if text: screen.blit(font.render(text, True, c), (ix, yy)); yy += 18
    if g.next_name:
        shape = SHAPES[g.next_name]; c = PIECE_COLORS[g.next_name]; cs = 20
        offx, offy = ix + 20, yy
        for r in range(shape.shape[0]):
            for cc in range(shape.shape[1]):
                if shape[r, cc]:
                    pygame.draw.rect(screen, c, (offx + cc*cs, offy + r*cs, cs-2, cs-2))
    yy = WIN_H - MARGIN - 55
    for ctrl in ["SPC: pause", "ESC: exit", "Natural speed (1 action/frame)"]:
        screen.blit(font.render(ctrl, True, (100, 100, 130)), (ix, yy)); yy += 16

running = True
thinking = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            pause = not pause
    if not pause and not g.game_over:
        step_ai()
    screen.fill(BG); draw_board(); draw_info()
    if g.game_over:
        txt = font_big.render("GAME OVER", True, (255, 60, 60))
        tw = txt.get_width()
        screen.blit(txt, (MARGIN + BOARD_W//2 - tw//2, MARGIN + BOARD_H//2 - 20))
        screen.blit(font.render("Press ESC to exit", True, (150, 150, 180)),
                    (MARGIN + BOARD_W//2 - 70, MARGIN + BOARD_H//2 + 10))
    pygame.display.flip()
    clock.tick(30)
pygame.quit(); sys.exit()
