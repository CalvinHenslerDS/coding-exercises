# w3resource

# Python Basic Part -I


# Exercise 12:

# Problem Statement:

# Write a Python program that prints the calendar for a given month and year.
# Note : Use 'calendar' module.


# Solution Attempt:

# Import the calendar module
import calendar

# Prompt the user to input a year and month
year = int(input("Input a year (####): "))
month = int(input("Input a month (##): "))

# Print the calendar for the specified year and month
print(calendar.month(year,month))


# Lessons Learned:

# Modules:
# calendar: Provides functions and classes for working with calendars

# Syntax:
# calendar.month(): Retrieves the calendar for a given year and month using the calendar module