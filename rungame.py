import numpy as np
import random
# Ensure this points to your environment file
from game_2048_env import Game2048Env
import time


def triangle_score(board):
    """Return negative distance of max tile from bottom-right (prefer bottom-right alignment)."""
    max_tile = np.max(board)
    positions = np.argwhere(board == max_tile)
    best_score = -float('inf')
    for x, y in positions:
        # Distance from bottom-right
        score = -(abs(3 - x) + abs(3 - y))
        best_score = max(best_score, score)
    return best_score


def evaluate_board(env, action):
    """Simulate a move and return score gain, max tile, triangle score."""
    sim_env = Game2048Env()
    sim_env.board = np.copy(env.board)
    sim_env.score = env.score
    _, new_score, _, _ = sim_env.step(action)
    gain = new_score - env.score
    tile = np.max(sim_env.board)
    tri_score = triangle_score(sim_env.board)
    return gain, tile, tri_score, sim_env.board


def get_action(state, score):
    env = Game2048Env()
    env.board = np.copy(state)
    env.score = score

    options = []
    for action in [3, 1]:  # right, down
        if env.is_move_legal(action):
            gain, tile, tri_score, _ = evaluate_board(env, action)
            options.append((gain, tile, tri_score, action))

    if options:
        # Sort by triangle score, then gain, then tile
        options.sort(key=lambda x: (x[2], x[0], x[1]), reverse=True)

        # Override if one is significantly better in tile or gain
        if len(options) == 2:
            diff_gain = options[0][0] - options[1][0]
            diff_tile = options[0][1] - options[1][1]
            if diff_gain > 128 or diff_tile >= 128:
                # Choose better one even if triangle is worse
                return options[0][3]

        return options[0][3]

    # Fallbacks: left then up
    if env.is_move_legal(2):
        return 2
    if env.is_move_legal(0):
        return 0

    return 0  # Default if nothing is legal


def print_board(board, score, action=None):
    print("=" * 25)
    print(
        f"Action: {['↑', '↓', '←', '→'][action] if action is not None else 'N/A'} | Score: {score}")
    print("-" * 25)
    for row in board:
        print(" ".join(f"{num:4}" if num != 0 else "   ." for num in row))
    print("=" * 25)


if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    done = False
    step = 0

    print_board(env.board, env.score)

    while not done:
        action = get_action(state, env.score)
        state, reward, done, _ = env.step(action)
        print_board(env.board, env.score, action)
        step += 1

    print("Game Over")
    print(f"Final Score: {env.score}")
