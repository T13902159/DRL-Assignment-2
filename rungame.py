import numpy as np
import random
from game_2048_env import Game2048Env


class MCTSNode:
    def __init__(self, board, parent=None, action=None):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == 4  # Since there are 4 possible actions

    def best_child(self):
        if self.visits == 0:
            return random.choice(self.children)
        return max(self.children, key=lambda x: (x.wins / x.visits) if x.visits > 0 else float('inf'))


def simulate_game(env, move_sequence):
    env_copy = Game2048Env()
    env_copy.board = np.copy(env.board)
    env_copy.score = env.score
    for move in move_sequence:
        best_move = get_best_move_based_on_heuristics(env_copy)
        _, reward, done, _ = env_copy.step(best_move)
        if done:
            break
    return env_copy.score


def get_best_move_based_on_heuristics(env):
    legal_moves = []
    for action in [3, 1, 2, 0]:
        if env.is_move_legal(action):
            legal_moves.append(action)
    # Heuristic logic: prioritize actions that maintain large tiles or lead to merging
    # Placeholder for actual heuristic-based move selection
    return random.choice(legal_moves)


def select_move_with_mcts(env, simulations=5000):
    root = MCTSNode(board=np.copy(env.board))

    for _ in range(simulations):
        node = root
        while node.is_fully_expanded():
            node = node.best_child()

        for action in [3, 1, 2, 0]:
            if env.is_move_legal(action):
                child_node = MCTSNode(board=np.copy(
                    env.board), parent=node, action=action)
                node.children.append(child_node)

        best_child = node.best_child()
        score = simulate_game(env, [best_child.action])

        while node:
            node.visits += 1
            node.wins += score
            node = node.parent

    best_action = root.best_child().action
    return best_action


def get_action(state, score):
    env = Game2048Env()
    env.board = np.copy(state)
    env.score = score
    return select_move_with_mcts(env)


if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    done = False
    step = 0

    while not done:
        action = get_action(state, env.score)
        state, reward, done, _ = env.step(action)
        print(f"Step: {step}, Action: {['↑', '↓', '←', '→'][action]}")
        print(state)
        step += 1

    print("Game Over")
    print(f"Final Score: {env.score}")
