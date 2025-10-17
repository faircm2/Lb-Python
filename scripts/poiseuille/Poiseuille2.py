import numpy as np

# Define the discrete velocities
discrete_velocities = np.array([
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

# Select the desired index i (0 to 8)
i = 1  # Example index; change this as needed

# Extract c_i as a 1D array
c_i = discrete_velocities[i]  # Shape (2,)

# Define u_avg as a (101, 2) array filled with zeros
u_avg = np.zeros((101, 2))
print(u_avg)


print("u_avg2:")
u_avg2 = np.zeros((1, 2))
u_avg2 = np.array([1,2])
print(u_avg2)
print("arr:")
arr1 = u_avg2**2
print(arr1)
arr2 = np.multiply(u_avg2,u_avg2)
print(arr2)
arr3 = np.dot(u_avg2,u_avg2.T)
print(arr3)

u_avg3 = np.array([[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0.]])

print("np.einsum('ij,ij->i', u_avg3, u_avg3):")
arr3 = np.einsum('ij,ij->i', u_avg3, u_avg3)
print(arr3)
#print("dot_product_u_avg:")
#dot_product_u_avg = u_avg**2
# Print the result
#print(dot_product_u_avg)
#print(dot_product_u_avg.shape)