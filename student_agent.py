import numpy as np
import random
from game_2048_env import Game2048Env
import math


class UCTNode:
    def __init__(self, state, score, env, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0


def heuristic(env):
    empty_tiles = np.sum(env.board == 0)
    max_tile = np.max(env.board)
    return empty_tiles + math.log2(max_tile + 1)


class UCTMCTS:
    def __init__(self, env, iterations=50, exploration_constant=0.9, rollout_depth=5):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        new_env = Game2048Env()
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                uct_value = float('inf')
            else:
                exploit = child.total_reward / child.visits
                explore = self.c * \
                    math.sqrt(math.log(node.visits + 1) / child.visits)
                uct_value = exploit + explore
            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        total_reward = 0
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            _, reward, done, _ = sim_env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward + heuristic(sim_env)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            _, _, done, _ = sim_env.step(node.action)
            if done:
                return

        if not node.fully_expanded():
            action = node.untried_actions.pop()
            _, _, done, _ = sim_env.step(action)
            new_state = sim_env.board.copy()
            new_score = sim_env.score
            child_node = UCTNode(
                state=new_state, score=new_score, env=sim_env, parent=node, action=action)
            node.children[action] = child_node
            node = child_node

        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / \
                total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def get_action(state, score):
    env = Game2048Env()
    env.board = np.copy(state)
    env.score = score

    uct_mcts = UCTMCTS(env, iterations=100,
                       exploration_constant=0.9, rollout_depth=5)
    root = UCTNode(state=env.board.copy(), score=env.score, env=env)

    for _ in range(uct_mcts.iterations):
        uct_mcts.run_simulation(root)

    best_action, _ = uct_mcts.best_action_distribution(root)

    # Fallback in case best_action is None or illegal
    if best_action is not None and env.is_move_legal(best_action):
        return best_action
    for a in [3, 1, 2, 0]:  # right, down, left, up
        if env.is_move_legal(a):
            return a
    return 0
