# w3resource

# Python Basic Part -I


# Exercise 16:

# Problem Statement:

# Write a Python program to calculate the difference between a given number and 17. If the number is greater than 17, return twice the absolute difference.


# Solution Attempt:

# Prompt user to input a number
number = float(input("Input a number: "))

# Check whether the input is greater than 17, and perform the correct operation based on the result
if number > 17:
    result = 2 * abs(17 - number)
else:
    result = 17 - number

# Print the result
print(result)


# Lessons Learned:

# Other:
# Do not use built-in function names as variable names
# Return cannot be used outside of functions