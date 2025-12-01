import pickle
import numpy as np

def win_check(board):

    for i in range(3):
        if np.all(board[i, :] == 1):
            return 1
        if np.all(board[i, :] == 2):
            return 2
        if np.all(board[:, i] == 1):
            return 1
        if np.all(board[:, i] == 2):
            return 2
    
    if np.all(np.diag(board) == 1):
        return 1
    if np.all(np.diag(board) == 2):
        return 2
    
    if np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    if np.all(np.diag(np.fliplr(board)) == 2):
        return 2
    
    return None

def load_agent(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Agent file '{filename}' not found.")
        return None

bot_agent = load_agent('player_1_qagent_brain.pkl')

if bot_agent:
    bot_agent.epsilon = 0.0
    print("Agent loaded successfully and set to optimal play mode.")

def print_board(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    print("\n--- Current Board ---")
    for r in range(3):
        row_display = [symbols[board[r, c]] for c in range(3)]
        print(f" {row_display[0]} | {row_display[1]} | {row_display[2]} ")
        if r < 2:
            print("---+---+---")
    print("---------------------\n")

def play_game(bot):
    board = np.zeros((3, 3), dtype = int)
    current_player_id = 1
    while win_check(board) is None and 0 in board:
        print_board(board)

        available_moves = list(zip(*np.where(board == 0)))

        if current_player_id == 1:
            print("It's the bot's turn.")
            action = bot.choose_action(board, available_moves)
            board[action] = 1
            print(f"Bot plays at: {action}")
            current_player_id = 2

        else:
            print("Your turn.")
            while True:
                try:
                    move_str = input("Enter your move (row,col): ")
                    r, c = map(int, move_str.split(','))
                    action = (r, c)

                    if action in available_moves:
                        board[action] = 2
                        current_player_id = 1
                        break
                    else:
                        print("Invalid move.")
                except ValueError:
                    print("Invalid format.")
    print_board(board)
    winner = win_check(board)
    if winner == 1:
        print("The bot wins.")
    elif winner == 2:
        print("You win.")
    else:
        print("It's a Cat's game.")

if bot_agent:
    play_game(bot_agent)