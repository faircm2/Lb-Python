import numpy as np

# Data
discrete_velocities = np.array([
    [0, 0],    # i=0
    [1, 0],    # i=1
    [0, 1],    # i=2
    [-1, 0],   # i=3
    [0, -1],   # i=4
    [1, 1],    # i=5
    [-1, 1],   # i=6
    [-1, -1],  # i=7
    [1, -1]    # i=8
])

_ltc = np.array([
    0.457777778,
    0.111111111,
    0,
    0.114444444,
    0.027777778,
    0,
    0,
    0.027777778,
    0.738888889
])

c = np.array([
    0.01459389,	0
])

u = np.array([
    1, 0
])

h = np.dot(c, u)
print(h)

u_avg = np.array([
    5.06E-02, 0
])

dot_product_u_avg = np.einsum('k,k->', u_avg, u_avg)


discrete_velocities_reshaped = discrete_velocities[:, :, np.newaxis, np.newaxis]  # Shape: (9, 2, 1, 1)
# Expand _ltc to shape (9, 2, 101, 10003) using np.expand_dims
_ltc_expanded = np.expand_dims(_ltc, axis=1)
# Now replicate the values along the new dimension (axis=1) to match shape (9, 2, 101, 10003)
_ltc_expanded = np.repeat(_ltc_expanded, 2, axis=1)

# Step 3: Perform element-wise multiplication (now both arrays are compatible)
weighted_vectors = discrete_velocities_reshaped * _ltc_expanded  # Shape: (9, 2, 101, 10003)

# Step 4: Sum along axis 0 to aggregate the velocities
result = np.sum(weighted_vectors, axis=0)  # Shape: (2, 101, 10003)

# Print the result shape
print(result.shape)  # This should print (2, 101, 10003)