import pickle
import numpy as np
import random
import copy
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import pickle
import random


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_actions = [0, 1, 2, 3]  # up, down, left, right

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self):
        """Choose the child with the highest UCB (Upper Confidence Bound) score."""
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            ucb = child.total_reward / \
                (child.visits + 1) + np.sqrt(2 *
                                             np.log(self.visits + 1) / (child.visits + 1))
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child


class MCTS:
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations
        self.state_action_values = {}  # Dictionary to store state-action values

    def select(self, node):
        """Select the best child node using UCB1."""
        while not node.is_fully_expanded():
            action = node.untried_actions.pop()
            next_state, reward, done, _ = self.env.step(action)
            child_node = MCTSNode(next_state, parent=node)
            node.children.append(child_node)
            return child_node
        return node.best_child()

    def simulate(self, node):
        """Simulate a random rollout from the given node."""
        temp_env = copy.deepcopy(
            self.env)  # Create a temporary copy of the environment
        state = node.state
        done = False
        total_reward = 0

        while not done:
            action = random.choice([0, 1, 2, 3])  # Random action
            state, reward, done, _ = temp_env.step(action)
            total_reward += reward

        return total_reward

    def backpropagate(self, node, reward):
        """Backpropagate the reward through the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run(self, state):
        """Run the MCTS search for a given state."""
        root = MCTSNode(state)

        for _ in range(self.iterations):
            node = root
            # Selection: traverse the tree to find a node to expand
            node = self.select(node)
            # Simulation: perform a random simulation and get a reward
            reward = self.simulate(node)
            # Backpropagation: propagate the result back up the tree
            self.backpropagate(node, reward)

        # Store the best action for the current state
        best_child = root.best_child()
        best_action = best_child.parent.untried_actions[0]
        self.state_action_values[tuple(map(tuple, state))] = best_action

        return best_action

    def save_model(self, filename='mcts_model.pkl'):
        """Save the learned state-action values to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.state_action_values, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='mcts_model.pkl'):
        """Load the state-action values from a pickle file."""
        with open(filename, 'rb') as f:
            self.state_action_values = pickle.load(f)
        print(f"Model loaded from {filename}")


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)),
                         mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1,
                                     1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(
            new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(
            new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


# Initialize your environment
env = Game2048Env()

# Train the agent using MCTS
mcts_agent = MCTS(env, iterations=1000)
done = False
state = env.reset()

# Run for a fixed number of steps or episodes
for episode in range(100):  # Train for 100 episodes
    state = env.reset()
    done = False
    while not done:
        action = mcts_agent.run(state)  # Use MCTS to select an action
        state, reward, done, _ = env.step(action)
        env.render()  # Optionally render the game after each step

# Save the learned state-action values to a pickle file
mcts_agent.save_model('mcts_model.pkl')
