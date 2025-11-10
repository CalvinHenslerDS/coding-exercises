# w3resource

# Python Basic Part -I


# Exercise 14:

# Problem Statement:

# Write a Python program to calculate the number of days between two dates.


# Solution Attempt:

# Import date fromn datetime module
from datetime import date

# Define two dates
date1 = date(2001, 9, 11)
date2 = date.today()

# Calculate the number of days between them
difference = date2 - date1

# Print the difference
print(difference.days)


# Lessons Learned:

# Syntax:
# date.today(): Retrieves today's date and outputs it in the format YYYY-MM-DD
# .days: An attribute used to extract a whole number of days from a timedelta object