# Tic-Tac-Toe

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random



class DQN(nn.Module):
    """Deep Q-Network for evaluating Tic-Tac-Toe board states.
    
    This network takes a 3x3 board as input (flattened to 9 units)
    and outputs a Q-Value for each possible action on the board.
    The architecture consists of two hidden layers with ReLU activations.
    
    Attributes:
    net (nn.Sequential): The neural network responsible for computing Q-values.
    """

    def __init__(self):
        """Initialize the DQN architecture."""

        # Initialize nn.Module internals
        super().__init__()

        # Create a feed-forward neural network
            # Two hidden layers (with 64 ReLU units)
        # and outputs:
            # 
        self.net = nn.Sequential(

            # Flattened 3x3 board
            nn.Flatten(),

            # Two hidden layers with 64 ReLU units
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 9)

        )

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the board state,
                shaped as (batch_size, 3, 3).
                
        Returns:
            torch.Tensor: Q-values for each of the 9 board positions.
        """
        return self.net(x)
        
class DQNAgent:
    """Deep Q-Learning agent for playing Tic-Tac-Toe.
    
    This agent maintains a policy network, a targeting network, an experience
    replay buffer, and implements epsilon-greedy action selection, Bellman updates,
    and periodic target-network synchronization.
    
    Attributes:
        player_id (int): Identifier of the player (1 or 2).
        epsilon (float): Exploration probability for epsilon-greedy policy.
        gamma (float): Discount factor for future rewards.
        memory (deque): Experience replay buffer containing transitions.
        batch_size (int): Number of samples used per training step.
        model (DQN): The primary Q-network updated during training.
        target_model (DQN): The target Q-network used for stable training.
        optim (torch.optim.Optimizer): Optimizer for the Q-network.
        loss_fn (nn.Module): Loss function used for Q-value updates.
    """
    def __init__(self, player_id, epsilon=0.2, gamma=0.9, lr=1e-3):
        """Initialize the DQNAgent with hyperparameters and neural networks.
        
        Args:
            player_id (int): ID representing the player (1 or 2).
            epsilon (float, optional): Exploration rate. Defaults to 0.2
            gamma (float, optional): Discount factor. Defaults to 0.9.
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
        """
        self.player_id = player_id
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = deque(maxlen = 50000)
        self.batch_size = 64
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optim = optim.Adam(self.model.parameters(), lr = lr)
        self.loss_fn = nn.MSELoss()

    # Define a function to compress the NumPy array representing the board state into a string to facilitate state storage
    def get_state_tensor(self, board):
        """Convert a NumPy board array into a standardized PyTorch tensor.
        
        Player 1 is encoded as +1, and player 2 is encoded as -1. This
        normalization helps the neural network learn symmetrically and
        reduces redundant patterns.
        
        Args:
            board (np.ndarray): 3x3 Tic-Tac-Toe board.
            
        Returns:
            torch.Tensor: Tensor of shape (1, 3, 3) containing normalized board values.
        """

        enc = board.copy()
        enc[enc == 2] = -1
        return torch.tensor(enc, dtype = torch.float32).unsqueeze(0)
    
    # Define a function to choose the next action by selecting the highest quality available action for the given board state (based on q_table)
    def choose_action(self, board, available_moves):
        """Select an action using an epsilon-greedy policy.
        
        With probability epsilon, a random action is selected for exploration.
        Otherwise, the action with the highest predicted Q-value is chosen,
        but only among legal moves.
        
        Args:
            board (np.ndarray): Current Tic-Tac-Toe board (3x3).
            available_moves (list[tuple[int, int]]): List of legal (row, col) moves.
            
        Returns:
            tuple[int, int]: The chosen move as (row, col).
        """

        # Choose a random move epsilon % of the time
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        # Get the predicted Q-values for the current board state
        state = self.get_state_tensor(board)
        qvals = self.model(state)[0].detach().numpy()

        # Mask illegal moves by assigning large, negative Q-values
        mask = np.full(9, -1e9)

        # Assign the proper, predicted Q-values to legal moves
        for (row, column) in available_moves:
            index = row * 3 + column
            mask[index] = qvals[index]

        # Pick the highest Q-value move from the flattened index and return the row, column index
        best_index = np.argmax(mask)
        return (best_index // 3, best_index % 3)
    
    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the experience replay buffer.
        
        Args:
            state (np.array): The previous board state.
            action (tuple[int, int]): Action taken (row, col).
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Resulting board state.
            done (bool): Whether the episode terminated after this transition.
        """

        # Store a (state, action, reward, next_state, done) transition in the replay buffer as a tuple
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def replay(self):
        """Perform one training step using a batch of replayed experiences.
        
        If enough experiences exist in memory, a minibatch is sampled and
        used to compute target Q-values using the Bellman equation. The
        policy network is then updated using gradient descent.
        
        Returns:
            None: This method updates model weights in place.
        """

        # If the memory exceeds the batch size, randomly sample experiences
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in batch:
            s = self.get_state_tensor(state).view(-1,9)
            ns = self.get_state_tensor(next_state).view(-1,9)

            target = self.model(s).detach()
            qvals_next = self.target_model(ns).detach()

            if done:
                target[0][action[0]*3 + action[1]] = reward
            else:
                target[0][action[0]*3 + action[1]] = reward + self.gamma * torch.max(qvals_next)

            states.append(s)
            targets.append(target)

        states = torch.vstack(states)
        targets = torch.vstack(targets)

        preds = self.model(states)
        loss = self.loss_fn(preds, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_target(self):
        """Synchronize the target network with the current policy network.
        
        Copies the parameters of the policy network to the target network.
        This is essential for training stability in DQN.
        """
        self.target_model.load_state_dict(self.model.state_dict())

def win_check(board):
    """Check the Tic-Tac-Toe board for a winner.
    
    Evaluates all rows, columns, the main diagonal, and the anti-diagonal
    for a line of three identical, non-zero markers.
    
    Args:
        board (np.ndarray): A 3x3 array representing the game board.
        
    Returns:
        int | None: The ID of the winning player (1 or 2) or None if
            no winner exists yet.
    """
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

# Initialize objects, player_1 and player_2, from the QAgent class with corresponding player_id
player_1 = DQNAgent(player_id=1, epsilon = 0.9)
player_2 = DQNAgent(player_id=2, epsilon = 0.9)

# Establish the training parameters
episodes = 50000
min_epsilon = 0.05
epsilon_decay = 0.9999
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
        game_over = winner is not None or len(available_moves) == 0

        if not game_over:
            # If it is player_1's turn and player 2 made a move in the previous turn, update the q_table of player 2's QAgent instance using the new board as the next state
            if current_player == player_1 and player_2_last_state is not None:
                player_2.remember(player_2_last_state, player_2_last_action, 0, board, False)
            # If it is player_2's turn and player 1 made a move in the previous turn, update the q_table of player_1's QAgent instance using the new board as the next state
            if current_player == player_2 and player_1_last_state is not None:
                player_1.remember(player_1_last_state, player_1_last_action, 0, board, False)
        
        # Update last_state and last_action of the current turn player based on the selected action and the new board
        if current_player == player_1:
            player_1_last_state, player_1_last_action = prev_board_copy, action
            current_player = player_2
        else:
            player_2_last_state, player_2_last_action = prev_board_copy, action
            current_player = player_1
    
    # If the game just ended, update the player_1 and player_2 instances based on the result, the training parameters, and updates to the q_table (from the learn function)
    if winner == player_1:

        player_1.remember(player_1_last_state, player_1_last_action, win_reward, board, True)
        player_2.remember(player_2_last_state, player_2_last_action, loss_reward, board, True)

    elif winner == player_2:
        player_2.remember(player_2_last_state, player_2_last_action, win_reward, board, True)
        player_1.remember(player_1_last_state, player_1_last_action, loss_reward, board, True)
        
    elif winner is None:
        player_1.remember(player_1_last_state, player_1_last_action, draw_reward, board, True)
        player_2.remember(player_2_last_state, player_2_last_action, draw_reward, board, True)

    player_1.replay()
    player_2.replay()

    # As the agents learn, decrease the rate of exploration (but keep it above the minimum threshold, min_epsilon)
    player_1.epsilon = max(min_epsilon, player_1.epsilon * epsilon_decay)
    player_2.epsilon = max(min_epsilon, player_2.epsilon * epsilon_decay)    

    

    # Every 10000 episodes, print a status update including the current episode, the epsilon of player_1 to four digits, and the size of q_table
    if i % 1000 == 0:
        player_1.update_target()
        player_2.update_target()
        print(f"Episode {i}: Epsilon = {player_1.epsilon:.4f}")


print("\nTraining Complete!")

player_1_path = 'player_1_dqn.pth'
player_2_path = 'player_2_dqn.pth'
torch.save(player_1.model.state_dict(), player_1_path)
print(f"Player 1 model saved to {player_1_path}")
torch.save(player_2.model.state_dict(), player_2_path)
print(f"Player 2 model saved to {player_2_path}")