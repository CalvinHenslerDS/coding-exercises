# w3resource

# Python Basic Part -I


# Exercise 83:


# Problem Statement:

# Test whether all numbers of a list is greater than a certain number.


# Solution Attempt:

import random

list = [2, 3, 4, 5]
number = 3.2
list2 = []

for i in list:
    if i > number:
        list2.append(i)
    else:
        continue

if len(list2) == 0:
    print("no items larger")
else:
    print(list2)
    print("these items are larger")