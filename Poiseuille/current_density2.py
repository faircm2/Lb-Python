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

# Step 1: Reshape _ltc to (9, 1, 1, 1) and then broadcast to (9, 2, 101, 10003)
_ltс_expanded = _ltc[:, np.newaxis, np.newaxis, np.newaxis]  # Shape: (9, 1, 1, 1)
_ltс_expanded = _ltс_expanded * np.ones((1, 2, 101, 10003))  # Broadcast to (9, 2, 101, 10003)

# Step 2: Reshape discrete_velocities to (9, 2, 1, 1) so that it can be broadcast with _ltc
discrete_velocities_reshaped = discrete_velocities[:, :, np.newaxis, np.newaxis]  # Shape: (9, 2, 1, 1)

# Step 3: Perform element-wise multiplication (now both arrays are compatible)
weighted_vectors = discrete_velocities_reshaped * _ltс_expanded  # Shape: (9, 2, 101, 10003)

# Step 4: Sum along axis 0 to aggregate the velocities
result = np.sum(weighted_vectors, axis=0)  # Shape: (2, 101, 10003)

# Print the result shape
print(result.shape)  # This should print (2, 101, 10003)