import numpy as np

# Define a sample u_avg array with three vectors (representing three "nodes")
u_avg = np.array([
    [-3.57E+06, 5.95E-18],
    [-3.57142857e-02, 5.94762335e-18],
    [-3.57142857e-02, 5.94762335e-18]
])

# Calculate the squared magnitude for each node using np.einsum
u_avg_squared = np.einsum('ij,ij->i', u_avg, u_avg)

# Print the result
print("u_avg_squared for each node:", u_avg_squared)