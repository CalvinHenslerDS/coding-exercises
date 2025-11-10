# w3resource

# Python Basic Part -I


# Exercise 15:

# Problem Statement:

# Write a Python program to get the volume of a sphere with radius six.


# Solution Attempt:

# Import the math module
import math

# Prompt user to input a radius
radius = float(input("Input a radius: "))

# Calculate the volume of the sphere
volume = (4 / 3) * math.pi * radius ** 3

# Print the difference
print("The volume of a sphere with radius %s is %s" % (radius,volume))