import numpy as np
Xn = 8
local_Xn = 4
local_Yn = 4
local_Zn = 4

x_offset = 4      # rank 0; change to 4 for rank 1

x_bs = 3
x_fs = 6

local_Zn = 4
z_offset = 0
h_bs = 2

def build_solid_mask():
    local_index_x = np.arange(local_Xn + 2)
    global_index_x = local_index_x + x_offset
    print("global_index_x: {0}".format(global_index_x))
    global_index_x = (global_index_x - 1) % Xn + 1
    solid_x  = (global_index_x < x_bs) | (global_index_x >= x_fs)
    print("solid_x: {0}".format(solid_x))

    local_index_z = np.arange(local_Zn + 2)
    global_index_z = local_index_z + z_offset   
    solid_z  = (global_index_z >= 1) & (global_index_z <= h_bs)
    print("solid_z: {0}".format(solid_z))
    #shift to start at 0, take the remainder after dividing by Xn (%), shift back by 1. Run it again.    

    solid_xz = solid_x[:, None, None] & solid_z[None, None, :]
    solid  = np.broadcast_to(solid_xz, (local_Xn+2, local_Yn+2, local_Zn+2))
    return solid

solid = build_solid_mask()
print(solid.shape)
print(solid[:, 0, :].astype(int))