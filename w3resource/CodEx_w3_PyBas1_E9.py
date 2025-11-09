# w3resource

# Python Basic Part -I


# Exercise 9:

# Problem Statement:

# Write a Python program to display the examination schedule. (extract the date from exam_st_date).
# exam_st_date = (11, 12, 2014)
# Sample Output : The examination will start from : 11 / 12 / 2014


# Solution Attempt:

# Create the tuple for the exam date
exam_st_date = (11, 12, 2014)

# Print the date in the desired format, extracting the relevant information from the tuple
print("The exam will start on: %s/%s/%s" % (exam_st_date))


# Lessons Learned:

#Efficiency:
# When reporting all elements in a tuple, no indices are required