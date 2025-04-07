import numpy as np
import random
# Ensure this points to your environment file
from game_2048_env import Game2048Env
import time


# Movement priorities: right (3), down (1), left (2), up (0)
PREFERRED_MOVES = [3, 1, 2, 0]


def get_legal_moves(env):
    return [action for action in range(4) if env.is_move_legal(action)]


def select_preferred_action(env):
    legal_moves = get_legal_moves(env)
    for move in PREFERRED_MOVES:
        if move in legal_moves:
            return move
    return legal_moves[0] if legal_moves else 0


def print_board(board, score, action=None):
    print(
        f"\nScore: {score} | Action: {['↑', '↓', '←', '→'][action] if action is not None else '-'}")
    print('-' * 21)
    for row in board:
        print(' '.join(f"{val:4}" if val != 0 else '   .' for val in row))
    print('-' * 21)


def run_game():
    env = Game2048Env()
    state = env.reset()
    done = False
    total_moves = 0

    print_board(env.board, env.score)

    while not done:
        action = select_preferred_action(env)
        state, reward, done, _ = env.step(action)
        total_moves += 1
        print_board(env.board, env.score, action)
        time.sleep(0.1)

    print(
        f"\n✅ Game Over! Final Score: {env.score}, Total Moves: {total_moves}")


if __name__ == "__main__":
    run_game()
