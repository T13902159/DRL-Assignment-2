import numpy as np
import random
# Ensure this points to your environment file
from game_2048_env import Game2048Env
import time
import copy
import random
import math
import numpy as np

# UCT Node for MCTS


class UCTNode:
    def __init__(self, state, score, env, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


def heuristic(env):
    empty_tiles = np.sum(env.board == 0)
    max_tile = np.max(env.board)
    return empty_tiles + math.log2(max_tile)


class UCTMCTS:
    def __init__(self, env, iterations=200, exploration_constant=0.9, rollout_depth=20):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                uct_value = float('inf')  # Prioritize unexplored children
            else:
                exploit = child.total_reward / child.visits
                explore = self.c * \
                    math.sqrt(math.log(node.visits) / child.visits)
                uct_value = exploit + explore
            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        total_reward = 0
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            # Use step instead of move
            _, reward, done, _ = sim_env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward + heuristic(sim_env)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            # Use step instead of move
            _, reward, done, _ = sim_env.step(node.action)
            if done:
                break

        # TODO: Expansion: if the node has untried actions, expand one.
        if not node.fully_expanded():
            action = node.untried_actions.pop()
            # Use step instead of move
            _, reward, done, _ = sim_env.step(action)
            new_state = sim_env.board.copy()
            new_score = sim_env.score
            child_node = UCTNode(
                state=new_state, score=new_score, env=sim_env, parent=node, action=action)

            node.children[action] = child_node
            node = child_node  # Continue from expanded node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
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

    # Initialize MCTS object
    uct_mcts = UCTMCTS(env, iterations=200,
                       exploration_constant=0.9, rollout_depth=20)

    print_board(env.board, env.score)

    while not done:
        # Initialize root node
        root = UCTNode(state=env.board.copy(), score=env.score, env=env)

        # Run MCTS simulations
        for _ in range(uct_mcts.iterations):
            uct_mcts.run_simulation(root)

        # Choose the best action based on visit count
        best_action, visit_distribution = uct_mcts.best_action_distribution(
            root)

        # Take the action in the real environment
        state, reward, done, _ = env.step(best_action)
        total_moves += 1
        print_board(env.board, env.score, best_action)
        time.sleep(0.1)

    print(
        f"\n✅ Game Over! Final Score: {env.score}, Total Moves: {total_moves}")


if __name__ == "__main__":
    run_game()
