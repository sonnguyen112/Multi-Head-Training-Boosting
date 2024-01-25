import copy
a = [1, 2, 3]
b = copy.deepcopy(a)
a[2] = 0
print(b)