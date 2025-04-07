# train_2048.py

import numpy as np
import pickle
import random
from collections import defaultdict

BOARD_SIZE = 4
ALPHA = 0.001
GAMMA = 0.99
EPISODES = 100_000

# Define 4 tuples (rows + columns)
TUPLES = [
    [(i, j) for j in range(4)] for i in range(4)
] + [
    [(j, i) for j in range(4)] for i in range(4)
]


def get_ntuple_features(board):
    features = []
    for tup in TUPLES:
        feature = tuple(board[i][j] for i, j in tup)
        features.append(feature)
    return features


class NtupleValueFunction:
    def __init__(self):
        self.weights = defaultdict(float)

    def evaluate(self, board):
        value = 0
        for f in get_ntuple_features(board):
            value += self.weights[f]
        return value

    def update(self, board, target, alpha=ALPHA):
        for f in get_ntuple_features(board):
            self.weights[f] += alpha * (target - self.weights[f])


def swipe_left(board):
    new_board = np.zeros_like(board)
    reward = 0
    for i in range(4):
        line = board[i][board[i] != 0]
        new_line = []
        j = 0
        while j < len(line):
            if j + 1 < len(line) and line[j] == line[j + 1]:
                new_line.append(line[j] * 2)
                reward += line[j] * 2
                j += 2
            else:
                new_line.append(line[j])
                j += 1
        new_board[i][:len(new_line)] = new_line
    return new_board, reward


def move(board, direction):
    board = np.array(board)
    if direction == 0:  # Up
        board = np.rot90(board, -1)
        board, r = swipe_left(board)
        board = np.rot90(board)
    elif direction == 1:  # Down
        board = np.rot90(board, 1)
        board, r = swipe_left(board)
        board = np.rot90(board, -1)
    elif direction == 2:  # Left
        board, r = swipe_left(board)
    elif direction == 3:  # Right
        board = np.fliplr(board)
        board, r = swipe_left(board)
        board = np.fliplr(board)
    return board, r


def get_legal_actions(board):
    legal = []
    for a in range(4):
        new_board, _ = move(board, a)
        if not np.array_equal(board, new_board):
            legal.append(a)
    return legal


def add_tile(board):
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return board
    i, j = random.choice(empty)
    board[i][j] = 2 if random.random() < 0.9 else 4
    return board


def train():
    vf = NtupleValueFunction()

    for episode in range(EPISODES):
        board = np.zeros((4, 4), dtype=int)
        board = add_tile(add_tile(board))

        while True:
            legal = get_legal_actions(board)
            if not legal:
                break

            action = random.choice(legal)
            next_board, reward = move(board, action)
            next_board = add_tile(next_board)

            v_current = vf.evaluate(board)
            v_next = vf.evaluate(next_board)
            target = reward + GAMMA * v_next

            vf.update(board, target)
            board = next_board

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}")

    with open("value.pkl", "wb") as f:
        pickle.dump(vf.weights, f)
    print("Saved value function to value.pkl")


if __name__ == "__main__":
    train()
