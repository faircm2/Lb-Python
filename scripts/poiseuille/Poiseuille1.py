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

# Now perform the dot product
# Since u_avg is (101, 2) and c_i is (2,), the dot product is valid
# The result will be of shape (101,)
result = np.dot(u_avg, c_i)  # Or you can use u_avg @ c_i

# Print the result
print(result.shape)

import numpy as np

# Example: u_avg is a (101, 2) array, filled with your actual data.
u_avg = np.random.rand(101, 2)  # Replace this with your actual 'u_avg'

# Compute the dot product of each row of u_avg with itself
dot_product_u_avg = np.einsum('ij,ij->i', u_avg, u_avg)

# Print the result
print(dot_product_u_avg)
print(dot_product_u_avg.shape)


print("dot_product_u_avg:")
dot_product_u_avg = np.multiply(u_avg, u_avg)

# Print the result
print(dot_product_u_avg)
print(dot_product_u_avg.shape)


print("dot:")
dot_product_u_avg = u_avg* u_avg
# Print the result
print(dot_product_u_avg)
print(dot_product_u_avg.shape)