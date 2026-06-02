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

Ny = 100
Nx = 10*Ny
_ltc = np.random.rand(9, Ny, Nx+2)

__ltc = np.array([
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

weighted_vectors1 =  np.einsum('ki,i->k', discrete_velocities.T, __ltc)
weighted_vectors2 =  np.einsum('ki,ijl->kjl', discrete_velocities.T, _ltc)
#_u_ckl_dot = np.einsum('k,kij->ij', c_i, _u_ckl)

# Reshape _ltc to match the rows of discrete_velocities and multiply element-wise
weighted_vectors = discrete_velocities * _ltc[:, None]

# Sum the resulting weighted vectors along the rows (axis=0) to get [vx, vy]
result = np.sum(weighted_vectors, axis=0)

print(result)  # Expected output: [0.707777778, -0.794444444]