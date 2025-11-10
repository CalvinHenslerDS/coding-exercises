# w3resource

# Python Basic Part -I


# Exercise 17:

# Problem Statement:

# Write a Python program to test whether a number is within 100 of 1000 or 2000.


# Solution Attempt:


# Define a function that takes in the input as an argument and checks its proximity to 1000 and 2000, printing proximity statements as appropriate
def check_proximity(number):
    if number >= 1000 - 100 and number <= 1000 + 100:
        print("The input is within 100 of 1000 or 2000")
    elif number >= 2000 - 100 and number <= 2000 + 100:
        print("The input is within 100 of 1000 or 2000")
    else:
        print("The input is not within 100 of 1000 or 2000")
    return

# Prompt user to input a number
number = float(input("Input a number: "))

# Call the function
check_proximity(number)


# Alternate solution:


# Define a function that takes in the input as an argument and uses the absolute value function to check whether the input is within 100 of 1000 or 2000
def check_proximity_2(number_2):
    return abs(1000 - number_2) <= 100 or abs(2000 - number_2) <= 100

# Prompt user to input a number
number_2 = float(input("Input a number: "))

# Print the appropriate result using the output of the function
if check_proximity_2(number_2) == True:
    print("The input is within 100 of 1000 or 2000")
else:
    print("The input is not within 100 of 1000 or 2000")