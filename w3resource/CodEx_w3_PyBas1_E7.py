# w3resource

# Python Basic Part -I


# Exercise 7:

# Problem Statement:

# Write a Python program that accepts a filename from the user and prints the extension of the file.


# Solution Attempt:

# Prompt the user to input a file name
file_name = input("Input a file name (including the extension): ")

# Split the file_name string into a list using period as the delimiter
file_extension = file_name.split(".")

# Print the last item in the file_extension list
print("The extension of the file is: ",repr(file_extension[-1]))

# Lessons Learned:

#Syntax:
# .split: Method used to divide a string into a list or array of substrings based on a specified delimiter
# repr(): Built-in function that returns an official string representation of an object