import pygame
import sys
import numpy as np
from game import TetrisGame, PIECE_COLORS, SHAPES
from brain import NeuralNet

CELL = 24
BOARD_W = TetrisGame.WIDTH * CELL
BOARD_H = TetrisGame.HEIGHT * CELL
GAP = 40
MARGIN = 20
BAR_H = 70

WIN_W = MARGIN * 2 + BOARD_W * 2 + GAP
WIN_H = MARGIN * 2 + BOARD_H + BAR_H

BG = (12, 12, 22); BORDER = (55, 55, 80)
GRID = (35, 35, 55); GHOST = (55, 55, 78)

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Tetris — 1v1  Player vs AI")
font = pygame.font.SysFont('consolas', 14)
font_big = pygame.font.SysFont('consolas', 18)
clock = pygame.time.Clock()

net, gen, _ = NeuralNet.load("tetris_champion.npz")

player = TetrisGame()
ai = TetrisGame()

ai_target_rot = None
ai_target_col = None
ai_thinking = True
ai_dropping = False

def ai_decide():
    global ai_target_rot, ai_target_col, ai_thinking
    placements = ai.get_placements()
    if not placements:
        ai.game_over = True
        return
    scores = [net.forward(feat) for _, _, feat in placements]
    rot, col, _ = placements[np.argmax(scores)]
    ai_target_rot = rot
    ai_target_col = col
    ai_thinking = False

def ai_step():
    global ai_target_rot, ai_target_col, ai_thinking, ai_dropping
    if ai_thinking:
        ai_decide()
        return
    if ai_target_rot is None or ai_target_col is None:
        ai_thinking = True
        return
    if ai.current_rot != ai_target_rot:
        ai.rotate()
        return
    if ai.current_pos[1] < ai_target_col:
        ai.move_right()
        return
    if ai.current_pos[1] > ai_target_col:
        ai.move_left()
        return
    if not ai_dropping:
        ai_dropping = True
    if ai.move_down():
        return
    ai_dropping = False
    ai_target_rot = None
    ai_target_col = None
    ai_thinking = True

def draw_board(game, x, y, label, colour):
    for r in range(game.HEIGHT):
        for c in range(game.WIDTH):
            px = x + c * CELL; py = y + r * CELL
            if game.board[r, c]:
                col = tuple(game.board_colors[r, c])
                pygame.draw.rect(screen, col, (px+1, py+1, CELL-2, CELL-2))
                hl = tuple(min(255, v+50) for v in col)
                pygame.draw.rect(screen, hl, (px+1, py+1, CELL-6, 2))
                pygame.draw.rect(screen, hl, (px+1, py+1, 2, CELL-6))
            else:
                pygame.draw.rect(screen, GRID, (px+1, py+1, CELL-2, CELL-2))
    if game.game_over or game.current_pos is None:
        pygame.draw.rect(screen, BORDER, (x, y, BOARD_W, BOARD_H), 2)
        return
    try:
        gpos, gshape = game.ghost_pos()
        for r in range(gshape.shape[0]):
            for c in range(gshape.shape[1]):
                if gshape[r, c]:
                    br, bc = gpos[0] + r, gpos[1] + c
                    if 0 <= br < game.HEIGHT:
                        px = x + bc * CELL; py = y + br * CELL
                        pygame.draw.rect(screen, GHOST, (px+2, py+2, CELL-4, CELL-4))
    except:
        pass
    shape = game.rotate_shape(game.current_name, game.current_rot)
    col = PIECE_COLORS[game.current_name]
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r, c]:
                br, bc = game.current_pos[0] + r, game.current_pos[1] + c
                if 0 <= br < game.HEIGHT:
                    px = x + bc * CELL; py = y + br * CELL
                    pygame.draw.rect(screen, col, (px+1, py+1, CELL-2, CELL-2))
                    hl = tuple(min(255, v+50) for v in col)
                    pygame.draw.rect(screen, hl, (px+1, py+1, CELL-6, 2))
                    pygame.draw.rect(screen, hl, (px+1, py+1, 2, CELL-6))
    pygame.draw.rect(screen, BORDER, (x, y, BOARD_W, BOARD_H), 2)
    txt = font_big.render(label, True, colour)
    screen.blit(txt, (x, y - 24))

running = True
pause = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                pause = not pause
            if not pause and not player.game_over:
                if event.key == pygame.K_LEFT:
                    player.move_left()
                elif event.key == pygame.K_RIGHT:
                    player.move_right()
                elif event.key == pygame.K_DOWN:
                    player.move_down()
                elif event.key == pygame.K_UP:
                    player.rotate()
                elif event.key == pygame.K_SPACE:
                    player.hard_drop()

    if not pause:
        if not ai.game_over:
            ai_step()

    px = MARGIN
    py = MARGIN + 30
    ai_x = px + BOARD_W + GAP

    screen.fill(BG)
    draw_board(player, px, py, "PLAYER", (255, 220, 50))
    draw_board(ai, ai_x, py, f"AI (gen {gen})", (0, 210, 255))

    info_y = py + BOARD_H + 8
    sep_x = px + BOARD_W + GAP // 2

    screen.blit(font.render(f"Score: {player.score}  Lines: {player.lines}", True, (255, 220, 50)), (px, info_y))
    screen.blit(font.render(f"Score: {ai.score}  Lines: {ai.lines}", True, (0, 210, 255)), (ai_x, info_y))
    screen.blit(font.render("P: pause   SPACE: hard drop   ESC: exit", True, (80, 80, 110)), (px, info_y + 22))

    if player.game_over:
        txt = font_big.render("GAME OVER", True, (255, 60, 60))
        screen.blit(txt, (px + BOARD_W//2 - txt.get_width()//2, py + BOARD_H//2 - 10))
        txt2 = font.render("You lost!", True, (200, 200, 200))
        screen.blit(txt2, (px + BOARD_W//2 - txt2.get_width()//2, py + BOARD_H//2 + 20))
    if ai.game_over:
        txt = font_big.render("GAME OVER", True, (255, 60, 60))
        screen.blit(txt, (ai_x + BOARD_W//2 - txt.get_width()//2, py + BOARD_H//2 - 10))
        if player.game_over:
            pass
        elif not player.game_over:
            txt2 = font_big.render("YOU WIN!", True, (0, 255, 100))
            screen.blit(txt2, (px + BOARD_W//2 - txt2.get_width()//2, py + BOARD_H//2 + 50))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
