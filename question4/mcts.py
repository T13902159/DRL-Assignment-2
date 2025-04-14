import math
import random
import time
from collections import defaultdict


class Node:
    def __init__(self, state, parent=None, prior_action=None):
        self.state = state  # Connect6Env object
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.prior_action = prior_action
        self.untried_actions = self.get_possible_actions()

    def get_possible_actions(self):
        legal_moves = self.state.get_valid_moves()
        if self.state.turn == 0:
            return [[move] for move in legal_moves]  # First turn: only 1 stone
        else:
            pairs = []
            for i in range(len(legal_moves)):
                for j in range(i + 1, len(legal_moves)):
                    pairs.append([legal_moves[i], legal_moves[j]])
            return pairs

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.copy()
        next_state.play_move(action)
        child = Node(next_state, parent=self, prior_action=action)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                win_rate = child.wins / child.visits
                ucb = win_rate + c_param * \
                    math.sqrt(math.log(self.visits) / child.visits)
            choices.append((ucb, child))
        return max(choices, key=lambda x: x[0])[1]

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def is_terminal_node(self):
        return self.state.is_terminal()[0]

    def rollout(self):
        current = self.state.copy()
        player = current.player
        while True:
            valid = current.get_valid_moves()
            if current.turn == 0:
                moves = [random.choice(valid)]
            elif len(valid) >= 2:
                moves = random.sample(valid, 2)
            else:
                moves = valid
            current.play_move(moves)
            done, winner = current.is_terminal()
            if done:
                return 1 if winner == player else -1 if winner == -player else 0


class MCTSAgent:
    def __init__(self, simulations=200, time_limit=4.5):
        self.simulations = simulations
        self.time_limit = time_limit
        self.transposition_table = {}

    def get_action(self, env, is_black):
        root = Node(env.copy())
        start_time = time.time()
        sims = 0

        while time.time() - start_time < self.time_limit:
            node = root

            # Selection
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.is_terminal_node() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            result = node.rollout()

            # Backpropagation
            node.backpropagate(result)
            sims += 1

        print(f"Simulations performed: {sims}")
        best = max(root.children, key=lambda c: c.visits)
        return best.prior_action
