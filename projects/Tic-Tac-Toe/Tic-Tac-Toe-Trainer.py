# Tic-Tac-Toe

import numpy as np
import random
import pickle

def win_check(board):

    # Initialize booleans to track whether either player has satisfied a win condition
    player_1_winner = False
    player_2_winner = False

    # Initialize empty lists to store the lengths of the sets of the rows and columns of board
    set_lengths_rows = []
    set_lengths_columns = []

    # Calculate the length of the set of the diagonal and antidiagonal of board
    set_length_diag = len(set(np.diag(board)))
    set_length_antidiag = len(set(np.diag(np.fliplr(board))))

    # Calculate the sum of the diagonal and antidiagonal of board
    sum_diag = sum(np.diag(board))
    sum_antidiag = sum(np.diag(np.fliplr(board)))
    
    for i in board:

        # Append the length of the set of each row to set_lengths_rows
        set_lengths_rows.append(len(set(i)))

        # Create an array comprised of the sums of the items in the rows of board
        sum_rows = np.sum(board, axis=1)

        # Create an array comprised of the sums of the items in the columns of the board
        sum_columns = np.sum(board, axis=0)

    for i in board.T:
         
         # Append the length of the set of each column to set_lengths_columns
         set_lengths_columns.append(len(set(i)))

    # Convert sum_rows and sum_columns to lists
    sum_rows_list = sum_rows.tolist()
    sum_columns_list = sum_columns.tolist()

    # If there is only one unique entry in the diagonal, declare player 1 or 2 the winner if the sum of the diagonals is 3 or 6 respectively
    if set_length_diag == 1:
        if sum_diag == 3:
            player_1_winner = True
            return player_1_winner, player_2_winner
        elif sum_diag == 6:
            player_2_winner = True
            return player_1_winner, player_2_winner
        else:
            pass
    
    # If there is only one unique entry in the antidiagonal, declare player 1 or 2 the winner if the sum of the antidiagonals is 3 or 6 respectively
    if set_length_antidiag == 1:

        if sum_antidiag == 3:
            player_1_winner = True
            return player_1_winner, player_2_winner
        
        elif sum_antidiag == 6:
            player_2_winner = True
            return player_1_winner, player_2_winner
        
        # Otherwise, proceed to the next check
        else:
            pass

    # Iterate over the rows
    for i in range(len(set_lengths_rows)):

        # If there is only one unique entry in the current row, declare player 1 or 2 the winner if the sum of the row items is 3 or 6 respectively
        if set_lengths_rows[i] == 1 and sum_rows_list[i] == 3:
            player_1_winner = True
            return player_1_winner, player_2_winner
        
        elif set_lengths_rows[i] == 1 and sum_rows_list[i] == 6:
            player_2_winner = True
            return player_1_winner, player_2_winner
        
        # Otherwise, proceed to the next check
        else:
            continue

    # Iterate over the columns
    for i in range(len(set_lengths_columns)):

        # If there is only one unique entry in the current column, declare player 1 or 2 the winner if the sum of the column items is 3 or 6 respectively
        if set_lengths_columns[i] == 1 and sum_columns_list[i] == 3:
            player_1_winner = True
            return player_1_winner, player_2_winner
        
        elif set_lengths_columns[i] == 1 and sum_columns_list[i] == 6:
            player_2_winner = True
            return player_1_winner, player_2_winner
        
        # Otherwise, exit win_check and return player_1_winner and player_2_winner
        else:
            continue

    return player_1_winner, player_2_winner

# Create a class to build a q table which evaluates and stores the maximum quality of each potential action for a given board state
class QAgent:

    # Initialize the variables required to run the class
    def __init__(self, player_id, epsilon=0.2, alpha=0.3, gamma=0.9):

        # Initialize player_id (1 or 2) to distinguish between players while training
        self.player_id = player_id

        # Initialize a q_table in which to store the quality of each potential action for a given board state
        self.q_table = {}

        # Initialize epsilon to establish an exploration rate (the percentage of randomized moves)
        self.epsilon = epsilon

        # Initialize alpha, the learning rate
        self.alpha = alpha

        # Initialize gamma, the discount factor
        self.gamma = gamma

    # Define a function to compress the NumPy array representing the board state into a string to facilitate state storage
    def get_state_id(self,board):
        return "".join(map(str, board.flatten()))
    
    # Define a function to choose an action by selecting the highest quality action for the given board state
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

        # Loop over both the moves and q values in the q_table
        for move, value in self.q_table[state].items():

            # If the q value is greater than the current best_value, replace it and update best_moves
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
        player_1 = QAgent(player_id=1)
        player_2 = QAgent(player_id=2)

        # Establish the number of training runs
        episodes = 10000

        # Iterate episodes times
        for i in range(episodes):

            # Initialize a 3x3 NumPy array of zeros to represent the tic-tac-toe board
            board = np.zeros((3, 3), dtype = int)
            
            # Initialize a boolean to enable the computer to play the game within a while loop until a win condition is satisfied
            game_over = False

            # Initialize a variable to track the current player to enable proper reward assignment
            current_player = player_1

            # Initialize variables in which to store the last states and actions of each player with which to populate the Bellman Equation and, subsequently, q_table
            player_1_last_state = None
            player_1_last_action = None
            player_2_last_state = None
            player_2_last_action = None

            # Until a win condition is satisfied or the board is full...
            while not game_over:

                # Initialize a list that contains the zipped indices corresponding to each 0 in the board
                available_moves = list(zip(*np.where(board == 0)))

                # Call the choose_action function method on the current_player instance and store it in the variable action
                action = current_player.choose_action(board, available_moves)

                # Initialize a variable prev_board_copy to store a copy of the current board state
                prev_board_copy = board.copy()

                # Replace the item corresponding to the indices of action with the id of the current player (1 or 2)
                board[action] = current_player.player_id

                # Call win_check to determine whether the most recent move resulted in a win for either player
                player_1_winner, player_2_winner = win_check(board)

                # Assign player_1 or player_2 to winner if the move did result in a win, otherwise assign None to winner
                if player_1_winner == True:
                    winner = player_1
                elif player_2_winner == True:
                    winner = player_2
                else:
                    winner = None

                # Assign game_over to true if either player is assigned to winner or if the board is full
                game_over = winner is not None or 0 not in board

                if current_player == player_1:
                    # If player_1 just moved, update last_state and last_action for player_1
                    player_1_last_state = prev_board_copy
                    player_1_last_action = action

                    if game_over:

                        # If the game just ended, update the player_1 and player_2 instances based on the outputs of the learn function
                        if winner == player_1:

                            # Reward the winner with +10
                            player_1.learn(player_1_last_state, action, 10, board, True)

                            # Reward the loser with -10
                            player_2.learn(player_2_last_state, player_2_last_action, -10, board, True)

                        # Reward both players with +2 if the result is a tie
                        elif winner is None:
                            player_1.learn(player_1_last_state, action, 2, board, True)
                            player_2.learn(player_2_last_state, player_2_last_action, 2, board, True)

                else:

                    # If player_2 just moved, update last_state and last_action for player_2
                    player_2_last_state = prev_board_copy
                    player_2_last_action = action

                    # If the game is not over, update the player_1 instance based on the outputs of the learn function, but do not give a reward
                    if not game_over:
                        player_1.learn(player_1_last_state, player_1_last_action, 0, board, False)

                    # If the game just ended, update the player_1 and player_2 instances based on the outputs of the learn function
                    if game_over:
                        if winner == player_2:
                            # Reward the winner with +10
                            player_2.learn(player_2_last_state, action, 10, board, True)
                            # Reward the loser with -10
                            player_1.learn(player_1_last_state, player_1_last_action, -10, board, True)
                        elif winner is None:
                            # Reward both players with +2 if the result is a tie
                            player_1.learn(player_1_last_state, player_1_last_action, 2, board, True)
                            player_2.learn(player_2_last_state, action, 2, board, True)

                # Switch current_player             
                current_player = player_2 if current_player == player_1 else player_1