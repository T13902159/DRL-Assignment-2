import numpy as np
import string

BOARD_SIZE = 19
COLUMN_LABELS = string.ascii_uppercase[:BOARD_SIZE]
COLUMN_LABELS = COLUMN_LABELS.replace("I", "")  # GTP skips 'I'


class Connect6Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.history = []
        self.player = 1  # 1 for black, -1 for white
        self.turn = 0  # Track first move (only one stone)

    def copy(self):
        env = Connect6Env()
        env.board = self.board.copy()
        env.history = self.history.copy()
        env.player = self.player
        env.turn = self.turn
        return env

    def get_valid_moves(self):
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r, c] == 0]

    def is_valid(self, move):
        r, c = move
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == 0

    def play_move(self, moves):
        # Input: list of (r, c) tuples
        for move in moves:
            r, c = move
            self.board[r, c] = self.player
            self.history.append((r, c))
        self.player *= -1
        self.turn += 1

    def play_move_from_gtp(self, move_str):
        # Handle opponent move from GTP
        coords = move_str.strip().split()
        move_list = [self.gtp_to_action(m) for m in coords]
        self.play_move(move_list)

    def check_win(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[row, col]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while True:
                    r += dr * d
                    c += dc * d
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= 6:
                return True
        return False

    def is_terminal(self):
        # Check for win or draw
        for r, c in self.history[-2:]:  # Only need to check last moves
            if self.check_win(r, c):
                return True, self.board[r, c]
        if len(self.history) == BOARD_SIZE * BOARD_SIZE:
            return True, 0
        return False, None

    def action_to_gtp(self, actions):
        return " ".join([f"{COLUMN_LABELS[c]}{BOARD_SIZE - r}" for r, c in actions])

    def gtp_to_action(self, gtp):
        col = COLUMN_LABELS.index(gtp[0].upper())
        row = BOARD_SIZE - int(gtp[1:])
        return (row, col)
