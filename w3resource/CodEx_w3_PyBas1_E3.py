# w3resource

# Python Basic Part -I


# Exercise 3:

# Problem Statement:

# Write a Python program to display the current date and time.


# Solution Attempt:

# Import the datetime module
import datetime

# Set a variable 'now' equal to the current date and time
now = datetime.datetime.now()

# Get the current date and time as a formatted string and print
print("The current data and time is: ",now.strftime("%m/%d/%Y %H:%M:%S"))

# Lessons Learned:

# Modules:
# datetime: Provides classes for working with dates and times

# Syntax:
# .strftime(): Method that returns a string representing date and time using a date, time, or datetime object