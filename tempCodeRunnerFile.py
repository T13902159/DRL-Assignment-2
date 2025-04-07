def evaluate_board(env, action):
    """Simulate a move and return score gain and max tile."""
    sim_env = Game2048Env()
    sim_env.board = np.copy(env.board)
    sim_env.score = env.score
    _, new_score, _, _ = sim_env.step(action)
    score_gain = new_score - env.score
    max_tile = np.max(sim_env.board)
    return score_gain, max_tile


def get_action(state, score):
    env = Game2048Env()
    env.board = np.copy(state)
    env.score = score

    preferred = [3, 1]  # right, down
    evaluations = []

    for action in preferred:
        if env.is_move_legal(action):
            score_gain, max_tile = evaluate_board(env, action)
            evaluations.append((score_gain, max_tile, action))
        else:
            evaluations.append((-1, -1, action))

    # Sort by score gain, then max tile
    evaluations.sort(reverse=True)

    for _, _, best_action in evaluations:
        if env.is_move_legal(best_action):
            return best_action

    # Fallbacks
    if env.is_move_legal(2):  # left
        return 2
    if env.is_move_legal(0):  # up
        return 0

    return 0  # No legal moves (shouldn't happen)


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
