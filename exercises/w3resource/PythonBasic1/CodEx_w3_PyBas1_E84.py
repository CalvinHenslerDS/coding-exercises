# w3resource

# Python Basic Part -I


# Exercise 83:


# Problem Statement:

# Write a Python program to count the number of occurrences of a specific character in a string.


# Solution Attempt:

str = "not all those who wander are lost"

all_a = ""

count = 0

for char in str:
    if char == "a":
        count += 1

print("There are %s 'a's in the string." % count)