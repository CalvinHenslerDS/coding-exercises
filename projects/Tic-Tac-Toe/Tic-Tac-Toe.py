# Tic-Tac-Toe

import numpy as np
import random

board = np.zeros((3,3))

print(board)

board_2 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

def coin_flip():

    heads_tails = input("Select Heads or Tails:")
    flip_result = random.randint(1,2)
    computer_first = None

    if heads_tails == "Heads":
        heads_tails = 1
    else:
        heads_tails = 2

    if flip_result == 1:
        print("The coin landed on Heads.")
    else:
        print("The coin landed on Tails.")

    if flip_result == heads_tails:
        computer_first = False
        print("You go first!")
    else:
        computer_first = True
        print("The computer goes first.")

    return computer_first

def random_index_from_mask(board):

    # Generate a mask from the input board containing all items equal to zero
    mask = (board == 0)

    # Generate a tuple of 1D arrays that makeup the indices of the items in the mask
    candidate_indices = np.where(mask)

    #print(candidate_indices)
    #print(candidate_indices[0])

    if len(candidate_indices[0]) == 0:
        print ("The board is full. Exiting random_index_from_mask.")
        return None
    
    random_index_in_candidates = np.random.choice(len(candidate_indices[0]))
    print(random_index_in_candidates)

    return tuple(index[random_index_in_candidates] for index in candidate_indices)

def make_move(board, computer_first):

    if random_index_from_mask(board) == None:
        print("The board is full. Exiting make_move.")
        return None
    elif computer_first == True:
        affected_square = random_index_from_mask(board)
        board[affected_square] = 1
    else:
        player_move_coordinates = input("Select an empty square to place an 'O' in: (format: row#,column#)")
        player_move_coordinates_int_list = list(map(int, player_move_coordinates.split(',')))
        board[player_move_coordinates_int_list[0], player_move_coordinates_int_list[1]] = 2
    computer_first = not computer_first
    return board, computer_first



def win_check(board):
#print(len(set(board[0])))
#print(board[0])
#print(board[:,0])
    computer_winner = False
    challenger_winner = False
    set_lengths_rows = []
    set_lengths_columns = []

    set_length_diag = len(set(np.diag(board)))
    sum_diag = sum(np.diag(board))
    print(set_length_diag)
    print(sum_diag)
    set_length_antidiag = len(set(np.diag(np.fliplr(board))))
    sum_antidiag = sum(np.diag(np.fliplr(board)))
    # print(set_length_antidiag)
    # print(sum_antidiag)
    
    for i in board:
        set_lengths_rows.append(len(set(i)))
        sum_rows = sum(board)
    
    sum_rows_list = sum_rows.tolist()

    # print(set_lengths_rows)
    # print(sum_rows)
    # print(sum_rows_list)

    for i in board.T:
        set_lengths_columns.append(len(set(i)))
        sum_columns = sum(board.T)

    sum_columns_list = sum_columns.tolist()

    # print(set_lengths_columns)
    # print(sum_columns)
    # print(sum_columns_list)

    if set_length_diag == 1:
        if sum_diag == 3:
            computer_winner = True
            print("The computer won along the diagonal")
            return computer_winner, challenger_winner
        elif sum_diag == 6:
            challenger_winner = True
            print("The challenger won along the diagonal")
            return computer_winner, challenger_winner
        else:
            pass
    
    if set_length_antidiag == 1:
        if sum_antidiag == 3:
            computer_winner = True
            print("The computer won along the antidiagonal")
            return computer_winner, challenger_winner
        elif sum_antidiag == 6:
            challenger_winner = True
            print("The challenger won along the antidiagonal")
            return computer_winner, challenger_winner
        else:
            pass

    for i in set_lengths_columns:
        if set_lengths_columns == 1 and sum_columns_list == 3:
            computer_winner = True
            print("The computer won along column %i" % i)
            return computer_winner, challenger_winner
        elif set_lengths_columns == 1 and sum_columns_list == 6:
            challenger_winner = True
            print("The challenger won along column %i" % i)
            return computer_winner, challenger_winner
        else:
            continue

    for i in set_lengths_rows:
        if set_lengths_rows == 1 and sum_rows_list == 3:
            computer_winner = True
            print("The computer won along column %i" % i)
            return computer_winner, challenger_winner
        elif set_lengths_rows == 1 and sum_rows_list == 6:
            challenger_winner = True
            print("The challenger won along column %i" % i)
            return computer_winner, challenger_winner
        else:
            return computer_winner, challenger_winner
    


    
            

win_check(board_2)
print(board_2)
#for i in set_lengths_rows:
#    if i == 1:

set_length_diag = len(set(np.diag(board)))
print(np.diag(board))
print(set_length_diag)
print(set_lengths_rows)
print(sum_rows)
print(set_lengths_columns)
print(sum_columns)
print(board)

#def win_check(board):

    # if len(set(board[0])) == 1:
    #     if sum(board[0]) == 3:
    #         print("The computer wins along the top row!")
    #     elif sum(board[0]) == 6:
    #         print("The challenger wins along the top row!")
    #     else:
    #         pass
    # elif len(set(board[1])) == 1:
    #     if sum(board[1]) == 3:
    #         print("The computer wins along the top row!")
    #     elif sum(board[1]) == 6:
    #         print("The challenger wins along the top row!")
    #     else:
    #         pass
    # elif len(set(board[2])) == 1:
    #     if sum(board[2]) == 3:
    #         print("The computer wins along the top row!")
    #     elif sum(board[2]) == 6:
    #         print("The challenger wins along the top row!")
    #     else:
    #         pass
    # elif len(set(board[:,0])) == 1:
    #     if sum(board[:,0]) == 3:
    #         print("The computer wins along the top row!")
    #     elif sum(board[:,0]) == 6:
    #         print("The challenger wins along the top row!")
    #     else:
    #         pass
    # elif len(set(board[:,1])) == 1:
    #     if sum(board[:,1]) == 3:
    #         print("The computer wins along the top row!")
    #     elif sum(board[:,1]) == 6:
    #         print("The challenger wins along the top row!")
    #     else:
    #         pass

# def play_game():
#     board = np.zeros((3,3))
#     computer_first = coin_flip()
    
#     win_check_list_2 = (board[1,0], board[1,1], board)
#     for i in range(10):
#         if make_move == None:
#             print("The board is full. Exiting play_game.")
#             return None
#         else:
#             board, computer_first = make_move(board, computer_first)
#             print(board)
#     return board

# play_game()


'''
    if computer_turn == True:

        if len(candidate_positions) > 0:
            random_candidate_position = np.random.choice(candidate_positions)
'''


'''
    if len(candidate_positions) > 0:
        random_candidate_position = np.random.choice(candidate_positions)
'''        



'''
def play_tic_tac_toe():

    computer_first = coin_flip()

    if computer_first == True:
        


play_tic_tac_toe()
'''