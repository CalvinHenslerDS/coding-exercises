# w3resource

# Python Basic Part -I


# Exercise 6:

# Problem Statement:

# Write a Python program that accepts a sequence of comma-separated numbers from the user and generates a list and a tuple of those numbers.


# Solution Attempt:

# Prompt the user to input a sequence of numbers separated by a comma
values = input("Input a sequence of numbers separated by commas: ")

# Split the values string into a list using comma as the delimiter
values_list = values.split(",")

# Convert the list into a tuple
values_tuple = tuple(values_list)

# Print the list and the tuple
print("Here's the list:\n\t",values_list,"\nHere's the tuple:\n\t",values_tuple)

# Lessons Learned:

# Syntax: 
# .split(): Method that divides a string into a list of substrings based on a specified delimiter
# tuple(): Built-in constructor used to create a new typle (either empty or from an iterable)

# Vocabulary:
# Tuple: An ordered, immutable collection of items
# Item: a single element within a collection or sequence