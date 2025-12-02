# Tic-Tac-Toe

import numpy as np
import random
import pickle

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

# Create a class to build a q table which evaluates and stores the maximum quality of each potential action for a given board state
class QAgent:

    def __init__(self, player_id, epsilon=0.2, alpha=0.3, gamma=0.9):

        # Initialize player_id (1 or 2) to distinguish between players while training
        self.player_id = player_id

        # Initialize a q_table to store the quality of each potential action for a given board state
        self.q_table = {}

        # Initialize epsilon to establish an exploration rate (the percentage of randomized moves)
        self.epsilon = epsilon

        # Initialize alpha, the learning rate, which determineds to what extent newly-acquired information overrides old information
        self.alpha = alpha

        # Initialize gamma, the discount factor, which determines the importance of future rewards compared to immediate rewards
        self.gamma = gamma

    # Define a function to compress the NumPy array representing the board state into a string to facilitate state storage
    def get_state_id(self,board):
        return "".join(map(str, board.flatten()))
    
    # Define a function to choose the next action by selecting the highest quality available action for the given board state (based on q_table)
    def choose_action(self, board, available_moves):

        # Convert the current board state into a string
        state = self.get_state_id(board)

        # If the board state is a new board state, add it to q_table with a default quality of 0 for all available moves
        if state not in self.q_table:
            self.q_table[state] = {move: 0.0 for move in available_moves}

        # Determine whether an exploratory move will be made by comparing a random value between 0 and 1 to epsilon
        if random.uniform(0,1) < self.epsilon:

            # If an exploratory move is to be made, select a move at random from the available moves
            return random.choice(available_moves)
        
        # Initialize best_value as negative infinity to ensure that the first score the agent checks will be larger than best_value
        best_value = -float('inf')

        # Initialize an empty list of best moves
        best_moves = []

        # Loop over the moves and q values in q_table for the current board state
        for move, value in self.q_table[state].items():

            # If the q value is greater than the current best_value, replace best_value and update best_moves
            if value > best_value:
                best_value = value
                best_moves = [move]

            # If the q value is equal to the current best_value, append the move to best_moves
            elif value == best_value:
                best_moves.append(move)
        
        # Randomly select a move from best_moves
        return random.choice(best_moves)
    
    # Define a function to update the Bellman Equation
    def learn(self, board, action, reward, next_board, game_over):

        # Convert the current board state into a string
        state = self.get_state_id(board)

        # Initialize a variable to store the next board state based on the selected move
        next_state = self.get_state_id(next_board)

        # Initialize a variable to store the state and q value of the current board state
        current_q = self.q_table.get(state, {}).get(action, 0.0)

        # If game_over is True, set max_future_q equal to 0
        if game_over:
            max_future_q = 0

        else:
            # If next_state is not in q_table, set max_future_q equal to 0
            if next_state not in self.q_table:
                max_future_q = 0
            
            # If next_state is in q_table, set max_future_q equal to the maximum associated q value 
            else:
                max_future_q = max(self.q_table[next_state].values())

        # Calculate new_q using the Bellman equation
        new_q = current_q + self.alpha * (reward + (self.gamma * max_future_q) - current_q)

        # If the state is not in q_table, add it 
        if state not in self.q_table:
            self.q_table[state] = {}

        # Update the q value with new_q based on the result of the Bellman Equation
        self.q_table[state][action] = new_q

# Initialize objects, player_1 and player_2, from the QAgent class with corresponding player_id
player_1 = QAgent(player_id=1, epsilon = 0.9)
player_2 = QAgent(player_id=2, epsilon = 0.9)

# Establish the training parameters
episodes = 100000
min_epsilon = 0.05
epsilon_decay = 0.99995
win_reward = 10
loss_reward = -10
draw_reward = 2

# Iterate episodes times
for i in range(episodes):

    # Initialize a 3x3 NumPy array of zeros to represent the tic-tac-toe board
    board = np.zeros((3, 3), dtype = int)
    
    # Initialize a boolean to enable the computer to play the game within a while loop until a win condition is satisfied
    game_over = False

    # Initialize a variable to track the current player to enable proper reward assignment
    current_player = player_1

    # Initialize variables to store the last states and actions of each player with which to populate the Bellman Equation and, subsequently, q_table
    player_1_last_state = None
    player_1_last_action = None
    player_2_last_state = None
    player_2_last_action = None

    while not game_over:

        # Initialize a list that contains the zipped indices corresponding to each 0 in the board
        available_moves = list(zip(*np.where(board == 0)))

        if not available_moves:
            winner = None
            game_over = True
            break

        # Call the choose_action function method on the current_player instance and store it in the variable action
        action = current_player.choose_action(board, available_moves)

        # Initialize a variable prev_board_copy to store a copy of the current board state
        prev_board_copy = board.copy()

        # Replace the item corresponding to the indices of action with the id of the current player (1 or 2)
        board[action] = current_player.player_id

        # Call win_check to determine whether the most recent move resulted in a win for either player
        winner_id = win_check(board)
        if winner_id == 1:
            winner = player_1
        elif winner_id == 2:
            winner = player_2
        else:
            winner = None
    
        # Assign game_over to true if either player is assigned to winner or if the board is full
        game_over = winner is not None or 0 not in board

        if not game_over:
            # If it is player_1's turn and player 2 made a move in the previous turn, update the q_table of player 2's QAgent instance using the new board as the next state
            if current_player == player_1 and player_2_last_state is not None:
                player_2.learn(player_2_last_state, player_2_last_action, 0, board, False)
            # If it is player_2's turn and player 1 made a move in the previous turn, update the q_table of player_1's QAgent instance using the new board as the next state
            elif current_player == player_2 and player_1_last_state is not None:
                player_1.learn(player_1_last_state, player_1_last_action, 0, board, False)
        
        # Update last_state and last_action of the current turn player based on the selected action and the new board
        if current_player == player_1:
            player_1_last_state, player_1_last_action = prev_board_copy, action
            current_player = player_2
        else:
            player_2_last_state, player_2_last_action = prev_board_copy, action
            current_player = player_1
    
    # As the agents learn, decrease the rate of exploration (but keep it above the minimum threshold, min_epsilon)
    player_1.epsilon = max(min_epsilon, player_1.epsilon * epsilon_decay)
    player_2.epsilon = max(min_epsilon, player_2.epsilon * epsilon_decay)    

    # If the game just ended, update the player_1 and player_2 instances based on the result, the training parameters, and updates to the q_table (from the learn function)
    if winner == player_1:

        player_1.learn(player_1_last_state, player_1_last_action, win_reward, board, True)
        player_2.learn(player_2_last_state, player_2_last_action, loss_reward, board, True)

    elif winner == player_2:
        player_2.learn(player_2_last_state, player_2_last_action, win_reward, board, True)
        player_1.learn(player_1_last_state, player_1_last_action, loss_reward, board, True)
        
    elif winner is None:
        player_1.learn(player_1_last_state, player_1_last_action, draw_reward, board, True)
        player_2.learn(player_2_last_state, player_2_last_action, draw_reward, board, True)

    # Every 10000 episodes, print a status update including the current episode, the epsilon of player_1 to four digits, and the size of q_table
    if i % 10000 == 0:
        print(f"Episode {i}: P1 Epsilon = {player_1.epsilon:.4f}, Q-Table Size = {len(player_1.q_table)}")

# Define a function used to save an agent once training is complete
def save_agent(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)

# Save the player_1 and player_2 agents using save_agent
save_agent(player_1, 'player_1_qagent_brain.pkl')
save_agent(player_2, 'player_2_qagent_brain.pkl')

print("\nTraining Complete! Agent saved.")