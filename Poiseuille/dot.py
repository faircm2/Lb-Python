import numpy as np

# Define c and u
c = np.array([
    [0, 0],   # i=0 
    [1, 0],   # i=1
    [0, 1],   # i=2
    [-1, 0],  # i=3
    [0, -1],  # i=4
    [1, 1],   # i=5
    [-1, 1],  # i=6
    [-1, -1], # i=7
    [1, -1]   # i=8
])

u = np.array([1, 1])

# Dot product of each row in c with u
dot_c_u = np.dot(c, u)

# Dot product of u with u
dot_u_u = np.dot(u, u)

print("dot_c_u: {0}".format(dot_c_u))
print("dot_u_u: {0}".format(dot_u_u))