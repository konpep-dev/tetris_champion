import numpy as np
import random

SHAPES = {
    'I': np.array([[1,1,1,1]]),
    'O': np.array([[1,1],[1,1]]),
    'T': np.array([[0,1,0],[1,1,1]]),
    'S': np.array([[0,1,1],[1,1,0]]),
    'Z': np.array([[1,1,0],[0,1,1]]),
    'J': np.array([[1,0,0],[1,1,1]]),
    'L': np.array([[0,0,1],[1,1,1]]),
}
PIECE_NAMES = list(SHAPES.keys())
PIECE_COLORS = {
    'I': (0, 255, 255), 'O': (255, 255, 0), 'T': (160, 0, 255),
    'S': (0, 255, 0), 'Z': (255, 0, 0), 'J': (0, 0, 255), 'L': (255, 160, 0)
}

class TetrisGame:
    WIDTH = 10
    HEIGHT = 20

    def __init__(self):
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=int)
        self.board_colors = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=int)
        self.current_name = None
        self.current_shape = None
        self.current_pos = None
        self.current_rot = 0
        self.next_name = None
        self.next_shape = None
        self.score = 0
        self.lines = 0
        self.steps = 0
        self.game_over = False
        self.new_piece()

    def rotate_shape(self, name, rot):
        s = SHAPES[name]
        for _ in range(rot % 4):
            s = np.rot90(s)
        return s

    def new_piece(self):
        if self.next_name is None:
            self.next_name = random.choice(PIECE_NAMES)
            self.next_shape = SHAPES[self.next_name]
        self.current_name = self.next_name
        self.current_shape = self.next_shape
        self.current_rot = 0
        self.next_name = random.choice(PIECE_NAMES)
        self.next_shape = SHAPES[self.next_name]
        shape = self.rotate_shape(self.current_name, 0)
        self.current_pos = [0, self.WIDTH // 2 - shape.shape[1] // 2]
        if self.collides(self.current_pos, shape):
            self.game_over = True

    def collides(self, pos, shape):
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    br = pos[0] + r
                    bc = pos[1] + c
                    if br >= self.HEIGHT or bc < 0 or bc >= self.WIDTH or (br >= 0 and self.board[br, bc]):
                        return True
        return False

    def lock(self):
        shape = self.rotate_shape(self.current_name, self.current_rot)
        color = PIECE_COLORS[self.current_name]
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    br = self.current_pos[0] + r
                    bc = self.current_pos[1] + c
                    if 0 <= br < self.HEIGHT:
                        self.board[br, bc] = 1
                        self.board_colors[br, bc] = color
        self.clear_lines()
        self.new_piece()

    def clear_lines(self):
        new_rows = []
        new_colors = []
        cleared = 0
        for r in range(self.HEIGHT):
            if all(self.board[r, :]):
                cleared += 1
            else:
                new_rows.append(self.board[r, :].copy())
                new_colors.append(self.board_colors[r, :, :].copy())
        if cleared:
            self.lines += cleared
            self.score += [0, 100, 300, 500, 800][cleared]
            empty = np.zeros((cleared, self.WIDTH), dtype=int)
            empty_c = np.zeros((cleared, self.WIDTH, 3), dtype=int)
            self.board = np.vstack((empty, new_rows))
            self.board_colors = np.vstack((empty_c, new_colors))

    def move_left(self):
        shape = self.rotate_shape(self.current_name, self.current_rot)
        new_pos = [self.current_pos[0], self.current_pos[1] - 1]
        if not self.collides(new_pos, shape):
            self.current_pos = new_pos

    def move_right(self):
        shape = self.rotate_shape(self.current_name, self.current_rot)
        new_pos = [self.current_pos[0], self.current_pos[1] + 1]
        if not self.collides(new_pos, shape):
            self.current_pos = new_pos

    def move_down(self):
        shape = self.rotate_shape(self.current_name, self.current_rot)
        new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
        if not self.collides(new_pos, shape):
            self.current_pos = new_pos
            return True
        self.lock()
        return False

    def rotate(self):
        new_rot = (self.current_rot + 1) % 4
        shape = self.rotate_shape(self.current_name, new_rot)
        if not self.collides(self.current_pos, shape):
            self.current_rot = new_rot

    def hard_drop(self, shape=None):
        if shape is None:
            shape = self.rotate_shape(self.current_name, self.current_rot)
        pos = self.current_pos.copy()
        while not self.collides([pos[0] + 1, pos[1]], shape):
            pos[0] += 1
        self.current_pos = pos
        self.lock()

    def ghost_pos(self):
        shape = self.rotate_shape(self.current_name, self.current_rot)
        pos = self.current_pos.copy()
        while not self.collides([pos[0] + 1, pos[1]], shape):
            pos[0] += 1
        return pos, shape

    def simulate_placement(self, piece_name, rotation, col):
        shape = self.rotate_shape(piece_name, rotation)
        if self.collides([0, col], shape):
            return None, 0
        row = 0
        while not self.collides([row + 1, col], shape):
            row += 1
        board = self.board.copy()
        color = PIECE_COLORS[piece_name]
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    br = row + r
                    bc = col + c
                    if 0 <= br < self.HEIGHT:
                        board[br, bc] = 1
        new_rows = []
        cleared = 0
        for r in range(self.HEIGHT):
            if all(board[r, :]):
                cleared += 1
            else:
                new_rows.append(board[r, :].copy())
        if cleared:
            empty = np.zeros((cleared, self.WIDTH), dtype=int)
            board = np.vstack((empty, new_rows))
        return board, cleared

    def get_placements(self):
        placements = []
        for rot in range(4):
            shape = self.rotate_shape(self.current_name, rot)
            for col in range(self.WIDTH - shape.shape[1] + 1):
                result = self.simulate_placement(self.current_name, rot, col)
                if result[0] is None:
                    continue
                board, lines = result
                feat = extract_features(board, lines)
                placements.append((rot, col, feat))
        return placements

    def execute_ai(self, rotation, col):
        self.current_rot = rotation
        shape = self.rotate_shape(self.current_name, rotation)
        if self.collides([0, col], shape):
            self.game_over = True
            return
        row = 0
        while not self.collides([row + 1, col], shape):
            row += 1
        self.current_pos = [row, col]
        self.lock()
        self.steps += 1

def extract_features(board, lines_cleared):
    heights = []
    for c in range(10):
        h = 0
        for r in range(20):
            if board[r, c]:
                h = 20 - r
                break
        heights.append(h)
    holes = 0
    for c in range(10):
        found = False
        for r in range(20):
            if board[r, c]:
                found = True
            elif found:
                holes += 1
    bump = sum(abs(heights[i] - heights[i+1]) for i in range(9))
    total_h = sum(heights)
    max_h = max(heights) if heights else 0
    filled = int(sum(sum(row) for row in board))

    eroded = lines_cleared * 10

    row_trans = 0
    for r in range(20):
        for c in range(9):
            if board[r, c] != board[r, c+1]:
                row_trans += 1

    col_trans = 0
    for c in range(10):
        for r in range(19):
            if board[r, c] != board[r+1, c]:
                col_trans += 1

    deep_wells = 0
    cum_wells = 0
    for c in range(10):
        left = heights[c-1] if c > 0 else 20
        right = heights[c+1] if c < 9 else 20
        well_depth = min(left, right) - heights[c]
        if well_depth > 1:
            deep_wells += 1
            cum_wells += well_depth

    return np.array(
        [h / 20.0 for h in heights] +
        [holes / 100.0, bump / 180.0, total_h / 200.0, max_h / 20.0,
         lines_cleared / 4.0, filled / 200.0,
         eroded / 40.0, row_trans / 190.0, col_trans / 190.0,
         deep_wells / 5.0, cum_wells / 20.0],
        dtype=np.float32
    )
