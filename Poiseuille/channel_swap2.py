import numpy as np

# Define the dimensions of the lattice
Ny = 3     # Number of rows, including boundaries
Nx = 5     # Number of columns

# Define channel and antichannel indices for swapping
channel_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
antichannel_indices = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Define relevant swap indices for top and bottom boundaries
relevant_indices_top = np.array([False, False, True, False, False, True, True, False, False])
relevant_indices_bottom = np.array([False, False, False, False, True, False, False, True, True])

# Initialize the lattice with specific values for easy verification
# Shape is (Ny + 1) x (Nx + 3) x 9, where 9 represents the channels
_ltc = np.arange((Ny + 1) * (Nx + 3) * 9).reshape((Ny + 1, Nx + 3, 9))

# Function to apply bounce-back boundary conditions with relevant swaps
def apply_bounce_back_boundaries(_ltc, channel_indices, antichannel_indices, relevant_indices_top, relevant_indices_bottom):
    # Create a copy of the lattice to avoid modifying it in place
    updated_ltc = _ltc.copy()
    
    # Define the row indices for the top and bottom boundaries
    upper_boundary = 0
    lower_boundary = Ny

    # Apply swaps on the top boundary
    for i, swap_needed in enumerate(relevant_indices_top):
        if swap_needed:
            # Apply swaps on the upper boundary row for each relevant channel
            updated_ltc[upper_boundary, :, channel_indices[i]] = _ltc[upper_boundary, :, antichannel_indices[i]]
    
    # Apply swaps on the bottom boundary
    for i, swap_needed in enumerate(relevant_indices_bottom):
        if swap_needed:
            # Apply swaps on the lower boundary row for each relevant channel
            updated_ltc[lower_boundary, :, channel_indices[i]] = _ltc[lower_boundary, :, antichannel_indices[i]]
    
    return updated_ltc

# Apply the function
updated_lattice = apply_bounce_back_boundaries(_ltc, channel_indices, antichannel_indices, relevant_indices_top, relevant_indices_bottom)

# Print original and updated lattices for verification
print("Original Lattice (Top Boundary):")
print(_ltc[0, :, :])
print("Updated Lattice (Top Boundary):")
print(updated_lattice[0, :, :])

print("Original Lattice (Bottom Boundary):")
print(_ltc[Ny, :, :])
print("Updated Lattice (Bottom Boundary):")
print(updated_lattice[Ny, :, :])
