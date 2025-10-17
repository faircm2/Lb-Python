import numpy as np

# Dimensions
Nx, Ny = 2, 2  # For simplicity

# Create a combined array with shape (9, 2)
combined = np.array([[0.44444444, 0.44444444],
                     [0.11111111, 0.11111111],
                     [0.11111111, 0.11111111],
                     [0.11111111, 0.11111111],
                     [0.11111111, 0.11111111],
                     [0.02777778, 0.02777778],
                     [0.02777778, 0.02777778],
                     [0.02777778, 0.02777778],
                     [0.02777778, 0.02777778]], dtype=object)

# Create the lattice with shape (Nx+3, Ny+1, 9)
lattice = np.empty((Nx + 3, Ny + 1, 9), dtype=object)

# Assign combined to every position in the third dimension of lattice
#lattice[:, :, :] = combined

a = np.arange(4)
a.shape = (2, 2)
print("a:")
print(a)

b = np.array([[8], [9]])
print("b:")
print(b)
c = a + b
print("c:")
print(c)

det1 = np.array([[0.44444444, 0.44444444]])
Ny= 100
Nx= 100000
lattice = np.zeros((Nx+3, Ny+1, 9,2), dtype=object)

print("det1.shape: {0}".format(det1.shape))
print("lattice.shape: {0}".format(lattice.shape))

lattice[0,0,0] = det1

combined = np.array([[0.44444444, 0.44444444],
    [0.11111111, 0.11111111],
    [0.11111111, 0.11111111],
    [0.11111111, 0.11111111],
    [0.11111111, 0.11111111],
    [0.02777778, 0.02777778],
    [0.02777778, 0.02777778],
    [0.02777778, 0.02777778],
    [0.02777778, 0.02777778]])

#for i in range(0,Nx+3):
#    for j in range(0,Ny+1):
#        lattice[i,j] = combined

#lattice[:, :, :, :] = combined        

print("combined.shape: {0}".format(combined.shape))
print("combined: {0}".format(combined))
print(lattice)

discrete_velocities = np.array([[0, 0],   # i=0
                      [1, 0],   # i=1
                      [0, 1],   # i=2
                      [-1, 0],  # i=3
                      [0, -1],  # i=4
                      [1, 1],   # i=5
                      [-1, 1],  # i=6
                      [-1, -1], # i=7
                      [1, -1]]) # i=8
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
u_avg = np.array([0,0])
rho = 1
Cs = 0.5773502691896258
for i in range(0,9):
    c_i = discrete_velocities[i]
    det1 = 3*np.dot(c_i,u_avg)/(Cs**2)
    det2 = (9/2)*np.dot(c_i,u_avg)**2/(Cs**4)
    det3 = (3/2)*np.dot(u_avg,u_avg.T)/(Cs**2)
    print("c_i[{0}]: {1}".format(i,c_i))
    print("u_avg[{0}]: {1}".format(i,u_avg))
    print("det3[{0}]: {1}".format(i,det3))
    feq0_8 = weights[i] * rho * (1 + det1 + det2 - det3)
    print("feq0_8[{0}]: {1}".format(i,feq0_8))

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

# Now perform the dot product
# Since u_avg is (101, 2) and c_i is (2,), the dot product is valid
# The result will be of shape (101,)
result = np.dot(u_avg, c_i)  # Or you can use u_avg @ c_i

# Print the result
print(result)
