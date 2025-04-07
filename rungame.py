import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import pickle
import random

# Make sure to define your Game2048Env class and load_value_function here...
# Define the COLOR_MAP and TEXT_COLOR dictionaries for rendering
COLOR_MAP = {
    0: "#cdc1b4",  # Empty tiles (light grey)
    2: "#eee4da",  # Tile with 2 (light beige)
    4: "#ede0c8",  # Tile with 4 (beige)
    8: "#f2b179",  # Tile with 8 (orange)
    16: "#f59563",  # Tile with 16 (orange-red)
    32: "#f67c5f",  # Tile with 32 (red)
    64: "#f65e3b",  # Tile with 64 (dark red)
    128: "#edcf72",  # Tile with 128 (yellow)
    256: "#edcc61",  # Tile with 256 (light yellow)
    512: "#edc850",  # Tile with 512 (yellow-orange)
    1024: "#edc53f",  # Tile with 1024 (dark yellow)
    2048: "#edc22e",  # Tile with 2048 (gold)
}

TEXT_COLOR = {
    0: "#776e65",  # Text color for empty tiles (dark grey)
    2: "#776e65",  # Text color for 2 (dark grey)
    4: "#776e65",  # Text color for 4 (dark grey)
    8: "#f9f6f2",  # Text color for 8 (light grey)
    16: "#f9f6f2",  # Text color for 16 (light grey)
    32: "#f9f6f2",  # Text color for 32 (light grey)
    64: "#f9f6f2",  # Text color for 64 (light grey)
    128: "#f9f6f2",  # Text color for 128 (light grey)
    256: "#f9f6f2",  # Text color for 256 (light grey)
    512: "#f9f6f2",  # Text color for 512 (light grey)
    1024: "#f9f6f2",  # Text color for 1024 (light grey)
    2048: "#f9f6f2",  # Text color for 2048 (light grey)
}


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


def load_value_function(filename='value.pkl'):
    """Load the value function from a pickle file."""
    with open(filename, 'rb') as f:
        value_function = pickle.load(f)
    return value_function


def get_action(state, score):
    import numpy as np


class StudentAgent:
    def __init__(self, env):
        self.env = env
        self.actions = ["up", "down", "left", "right"]

    def check_move_valid(self, action):
        """Check if a move is valid."""
        # Create a copy of the current board state
        temp_board = self.env.board.copy()
        moved = False

        if action == 0:  # Move up
            moved = self.env.move_up()
        elif action == 1:  # Move down
            moved = self.env.move_down()
        elif action == 2:  # Move left
            moved = self.env.move_left()
        elif action == 3:  # Move right
            moved = self.env.move_right()

        # Return if the move was valid (it changed the board)
        return moved

    def get_action(self, state, score):
        """
        Get action based on the current state and score using the defined strategy:
        1. Try right, down
        2. Use left if right/down are invalid
        3. Use up if left is also invalid
        """
        # Check for right (action 3) move first
        if self.check_move_valid(3):  # Right
            return 3

        # Check for down (action 1) move second
        if self.check_move_valid(1):  # Down
            return 1

        # Check for left (action 2) move if right and down are not valid
        if self.check_move_valid(2):  # Left
            return 2

        # Check for up (action 0) if none of the above are valid
        if self.check_move_valid(0):  # Up
            return 0

        # If no move is valid, return a random action (just in case)
        return random.choice([0, 1, 2, 3])


def run_game():
    # Create the environment
    env = Game2048Env()

    # Reset the environment to start a new episode
    state = env.reset()
    done = False

    # Keep track of the total score
    total_score = 0

    # Run the game until it ends
    while not done:
        # Get the action from the agent
        action = get_action(state, env.score)

        # Take the action and observe the result
        next_state, reward, done, info = env.step(action)

        # Update the total score
        total_score = env.score

        # Render the current state of the board (optional, will show a plot)
        env.render(action=action)

        # Set the state for the next iteration
        state = next_state

    print(f"Game Over! Final score: {total_score}")


# Run the game
run_game()
