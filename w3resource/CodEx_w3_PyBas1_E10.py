# w3resource

# Python Basic Part -I


# Exercise 10:

# Problem Statement:

# Write a Python program that accepts an integer (n) and computes the value of n+nn+nnn.
# Sample value of n is 5
# Expected Result : 615

# Solution Attempt:

# Prompt the user to provide an integer
n = int(input("Input an integer: "))

# Define nn and nnn
nn = int("%s%s" % (n, n))
nnn = int("%s%s%s" % (n, n, n))

calculated_value = n + nn + nnn

# Print the calculated value
print(calculated_value)

# Lessons Learned:

# Syntax:
# int(): Built-in function used to convert a value into an integer

# Other:
# input() returns a string, so numerical inputs must be converted before they can be used in calculations