import pickle
import numpy as np

def win_check(board):

    for i in range(3):
        
        # Check whether any rows contain all 1s or 2s
        if np.all(board[i, :] == 1):
            return 1
        if np.all(board[i, :] == 2):
            return 2
        
        # Check whether any columns contain all 1s or 2s
        if np.all(board[:, i] == 1):
            return 1
        if np.all(board[:, i] == 2):
            return 2
    
    # Check whether the diagonal contains all 1s or 2s
    if np.all(np.diag(board) == 1):
        return 1
    if np.all(np.diag(board) == 2):
        return 2
    
    # Check whether the antidiagonal contains all 1s or 2s
    if np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    if np.all(np.diag(np.fliplr(board)) == 2):
        return 2
    
    return None

# Define a function to load the agent (or return an error if the file is not found)
def load_agent(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Agent file '{filename}' not found.")
        return None

# Initialize a variable in which to store the loaded agent
bot_agent = load_agent('player_1_qagent_brain.pkl')

if bot_agent:

    # Set episolon equal to zero to suppress exploration
    bot_agent.epsilon = 0.0
    print("Agent loaded successfully and set to optimal play mode.")

# Define a function to display the board in a user-friendly way
def print_board(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    print("\n--- Current Board ---")
    for r in range(3):
        row_display = [symbols[board[r, c]] for c in range(3)]
        print(f" {row_display[0]} | {row_display[1]} | {row_display[2]} ")
        if r < 2:
            print("---+---+---")
    print("---------------------\n")

# Define a function to allow the user to play against the trained agent
def play_game(bot):

    # Initialize an empty board
    board = np.zeros((3, 3), dtype = int)

    # Set the robot to play first
    current_player_id = 1

    # Perform this loop until a win condition is satisfied or the board is full
    while win_check(board) is None and 0 in board:

        print_board(board)

        # Initialize a list of legal moves
        available_moves = list(zip(*np.where(board == 0)))

        # Play the bot's turn using the loaded agent and its associated choose_action logic from the QAgent class
        if current_player_id == 1:
            print("It's the bot's turn.")
            action = bot.choose_action(board, available_moves)
            board[action] = 1
            print(f"Bot plays at: {action}")

            # Toggle the turn player
            current_player_id = 2

        else:
            print("Your turn.")
            while True:
                try:
                    
                    # Prompt the user to input a move
                    player_move = input("Enter your move (row,column): ")
                    row, column = map(int, player_move.split(','))
                    action = (row, column)

                    # If the move is legal, replace the item in board corresponding to the indices of action with a 2 and toggle the turn player
                    if action in available_moves:
                        board[action] = 2
                        current_player_id = 1
                        break
                    else:
                        print("Invalid move.")
                except ValueError:
                    print("Invalid format.")

    # Once a win condition is satisfied, print the board and announce the winner                
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