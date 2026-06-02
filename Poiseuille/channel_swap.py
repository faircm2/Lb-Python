import numpy as np


def swap_channels(array, channel_indices, antichannel_indices, relevant_channel_indices):
    # Create a copy to avoid modifying the original array
    swapped_array = array.copy()
    
    # Iterate over each index and apply the swap if relevant_channel_indices is True
    for i, should_swap in enumerate(relevant_channel_indices):
        if should_swap:
            # Find the channel and antichannel index
            channel = channel_indices[i]
            antichannel = antichannel_indices[i]
            
            # Swap the elements
            swapped_array[channel] = array[antichannel]
    
    return swapped_array

# Example usage
original_array = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
antichannel_indices = [0, 3, 4, 1, 2, 7, 8, 5, 6]
relevant_channel_indices = [False, False, False, False, True, False, False, True, True]

# Perform the swap
swapped_array = swap_channels(original_array, channel_indices, antichannel_indices, relevant_channel_indices)
print("Original array:", original_array)
print("Swapped array:", swapped_array)
