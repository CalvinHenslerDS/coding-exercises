# Tic-Tac-Toe

import numpy as np
import random
import pickle

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