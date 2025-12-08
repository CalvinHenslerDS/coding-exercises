# w3resource

# Python Basic Part -I


# Exercise 51:


# Problem Statement:

# Write a Python program to determine the profiling of Python programs.


# Solution Attempt:

import cProfile

# Define an example function
def sum():
    print(1 + 2)

# Use cProfile to profile the sum function
cProfile.run('sum()')


# Lessons Learned:

# Modules:
# cProfile: Built-in module designed for profile the performance of Python code