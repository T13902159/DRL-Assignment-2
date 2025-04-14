
import sys
import time
from mcts import MCTSAgent
from connect6_env import Connect6Env


class Connect6GTP:
    def __init__(self):
        self.env = Connect6Env()
        self.agent = MCTSAgent(
            simulations=300, time_limit=4.5)  # Adjust as needed
        self.name = "MCTSConnect6"
        self.version = "1.0"

    def gtp_loop(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line.startswith("name"):
                print(f"= {self.name}\n")
            elif line.startswith("version"):
                print(f"= {self.version}\n")
            elif line.startswith("clear_board"):
                self.env.reset()
                print("=\n")
            elif line.startswith("boardsize"):
                print("=\n")  # Assume always 19
            elif line.startswith("play"):
                _, move = line.split()
                self.env.play_move_from_gtp(move)
                print("=\n")
            elif line.startswith("genmove"):
                _, color = line.split()
                is_black = (color.lower() == "b")
                action = self.agent.get_action(self.env, is_black)
                self.env.play_move(action)
                move_str = self.env.action_to_gtp(action)
                print(f"= {move_str}\n")
            elif line.startswith("quit"):
                print("=\n")
                break
            else:
                print("= \n")  # Acknowledge unknown commands


if __name__ == "__main__":
    gtp_bot = Connect6GTP()
    gtp_bot.gtp_loop()
