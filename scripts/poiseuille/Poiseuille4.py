import gc
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import matplotlib

print(matplotlib.__version__)

import math

import matplotlib.pyplot as plt
import numpy as np

TOTAL_TIME = 10
INIT_DENSITY = 1. #0.001

#Cs=1/math.sqrt(3)
Cs=math.sqrt(3)
print("Cs: {0}".format(Cs))
#D=10**-3 #m
#for now D=10**-1m to reduce the dimension of x
D=10**(-3) #m
print("D: {0}".format(D))
D_nd=100
print("D_nd: {0}".format(D_nd))
L=0.1

Ny=D_nd
print("Ny: {0}".format(Ny))
Nx=int(Ny*L/D)
print("Nx: {0}".format(Nx))

dx=D/D_nd #old->5*10**(-5)
dy = dx
print("dx: {0}".format(dx))
delta_t=1
#relaxation time, should be > 0,5
tau=0.6
dP=10**(-2) #Pa
rho_0=10**3
dRho=dP/Cs**2

#TEST STABILITY
dP = 0

p_out=1
p_in=p_out+dP
roh_in=p_in/Cs**2
roh_out=p_out/Cs**2
print("p_in: {0}".format(p_in))
print("p_out: {0}".format(p_out))
print("roh_in: {0}".format(roh_in))
print("roh_out: {0}".format(roh_out))

nu=3*10**(-6) #10**(-6) #m2/s
dt=Cs**2*(tau-1./2)*(dx**2/nu)
print("dt: {0}".format(dt))

#Poiseuille max velocity
U=1/8*rho_0/nu*dP/L*D**2

#TEST STABILITY
U=1

Re=D*U/nu
print("U: {0}".format(U))
print("Re: {0}".format(Re))


# 1. Conversion factor Cl for length
Cl=dx #freely chosen
dx_nd=dx/Cl
print("Cl: {0}".format(Cl))
print("dx_nd: {0}".format(dx_nd))

#2. Conversion factor Croh for density
Croh=rho_0
roh_nd = rho_0/Croh
print("Croh: {0}".format(Croh))
print("roh_nd: {0}".format(roh_nd))

#3. Conversion factor Ct for time
Ct=dt
dt_nd = dt/Ct
print("Ct: {0}".format(Ct))
print("dt_nd: {0}".format(dt_nd))

#4. Conversion factor Cu for time
Cu=Cl/Ct
U_nd = 0.1 #old value->U/Cu
print("Cu: {0}".format(Cu))
print("U_nd: {0}".format(U_nd))

#5. Conversion factor CF for Force
CF=Croh*Cl**4*Ct**(-2)
print("CF: {0}".format(CF))

#6. Conversion factor Cf for frequency
Cf=1/Ct
print("Cf: {0}".format(Cf))

#change nu_nd in order to achieve U_nd=0,1
nu_nd=((D_nd*U_nd)/(D*U))*nu
print("nu_nd: {0}".format(nu_nd))

tau_nd=(nu_nd/Cs**2)+1/2
print("tau_nd: {0}".format(tau_nd))
omega = dt/tau
print("omega: {0}".format(omega))
omega_nd = dt_nd/tau_nd
print("omega_nd: {0}".format(omega_nd))


#discrete velocity channels for D2Q9
discrete_velocities = np.array([[0, 0],     # i=0
                      [1, 0],               # i=1
                      [0, 1],               # i=2
                      [-1, 0],              # i=3
                      [0, -1],              # i=4
                      [1, 1],               # i=5
                      [-1, 1],              # i=6
                      [-1, -1],             # i=7
                      [1, -1]])             # i=8

print("Discrete velocities: {0}".format(discrete_velocities))
channel_indices = [0,1,2,3,4,5,6,7,8] #channel
antichannel_indices =[0,3,4,1,2,7,8,5,6] #anti-channel

#weights
weights = np.array([4./9,1./9,1./9,1./9,1./9,1./36,1./36,1./36,1./36])
print("Weights: {0}".format(weights))

#lattice for phase space; Nx+3 is due to periodic boundary conditions
#lattice = np.zeros((Ny+1, Nx+3, 9), dtype=float)
lattice = np.full((Ny+1, Nx+3, 9), INIT_DENSITY, dtype=float)
print("Shape of lattice:", lattice.shape)  # Should be (103,11,9)

# tmp storage arrays
streamed_lattice = np.zeros_like(lattice)
collision_lattice = np.zeros_like(lattice)
prestream_lattice = np.zeros_like(lattice)
roh_inlet = np.zeros(Ny+1)
roh_outlet = np.zeros(Ny+1)
outlet_midstream_avg_v = [] 

#Collision equilibrium distribution function feq(0->8)
def feq_i(i, rho, u_avg):
    c_i = discrete_velocities[i]
    #feq0_8 = weights[i] * rho * (1 + 3*(np.dot(c_i,u_avg)/(Cs**2)) + 9/2*(np.dot(c_i,u_avg)**2/(Cs**4)) - 3/2*(np.dot(u_avg,u_avg)/(Cs**2)))
    if u_avg.ndim == 2:
        dot = np.dot(c_i, u_avg.T)
    elif u_avg.ndim == 3:
        dot = np.einsum('k,ijk->ij', c_i, u_avg)
        u_avg_relevant = u_avg[:, :, :2]  # Now shape is (Ny + 1, Nx + 3, 2)
        dot_2 = np.tensordot(c_i, u_avg_relevant, axes=([0], [2]))

    _part1 = 3*dot/Cs**2
    _part2 = (9/2)*dot**2/Cs**4
    if u_avg.ndim == 2:
        dot_product_u_avg = np.einsum('ij,ij->i', u_avg, u_avg)
    elif u_avg.ndim == 3:
        dot_product_u_avg = np.einsum('ijk,ijk->ij', u_avg, u_avg)
        dot_product_u_avg1 = np.sum(u_avg * u_avg, axis=2)

    _part3 = (3/2)*dot_product_u_avg/Cs**2
    _ones = np.ones(_part1.shape)
    dets = _ones + _part1 + _part2 - _part3
    feq0_8 = weights[i] * rho * dets
        
    return feq0_8   


#Collision equilibrium distribution function feq(0-8) for each lattice slice at inlet/outlet
def f_eq_slc(j, roh_nd, u_avg):
    _slice = np.stack([feq_i(i, roh_nd, u_avg) for i in range(0, 9)], axis=-1)
    return _slice


# Roll the lattice based on the discrete velocities
def streamLtc(_ltc):
    tmpLtc = [np.roll(np.roll(_ltc[:, :, direction], shift=dy, axis=0), shift=dx, axis=1) for direction, (dx, dy) in enumerate(discrete_velocities)]
    shifted_lattice = np.stack(tmpLtc, axis=2)
    
    return shifted_lattice


#update the first 2 moments (density, current density,  and u_avg
def updateMoments(_ltc):
    #density
    roh = np.sum(_ltc, axis=2)
    roh_reshaped = roh[:, :, np.newaxis]

    ltc_summed = np.sum(_ltc, axis=2, keepdims=True)
    # Squeeze to remove the extra dimension and get a 2D array of shape (Ny+1, Nx+3)
    ltc_summed_2d = np.squeeze(ltc_summed)
    # Find the index of the maximum value in the 2D summed array
    max_index = np.argmax(ltc_summed_2d)
    # Convert the flat index to 2D indices (i, j)
    max_i, max_j = np.unravel_index(max_index, ltc_summed_2d.shape)
    print("The maximum value occurs at node (i, j):", (max_i, max_j))
    print("Maximum value:", ltc_summed_2d[max_i, max_j])

    #current density
    discrete_velocities_reshaped = discrete_velocities[np.newaxis, np.newaxis, :, :]
    _ltc_reshaped = _ltc[:, :, :, np.newaxis]
    current_density_arr = _ltc_reshaped * discrete_velocities_reshaped
    current_density = np.sum(current_density_arr, axis=2) 

    #avg velocity
    current_density_reshaped = current_density.reshape(_ltc.shape[0], _ltc.shape[1], 2)
    u_avg = current_density_reshaped / roh_reshaped

    return roh,current_density, u_avg


#2D Poiseuille inlet velocity u(y) for comparison with numerical result
def Poiseuille2DUy(y):
    u_poiseuille = np.zeros((2), dtype=float)
    #y = j/Ny*D
    #U=1/8*rho_0/nu*dP/L*D**2
    u_y=U*(1-((y-D/2)/(D/2))**2) #/Cu
    u_poiseuille[0] = u_y
    u_poiseuille[1] = 0
    #u_y_nd=u_y/Cu
    #print("Poiseuille2DUy->y:{0}; U:{1}; D:{2}; u_y:{3}; u_y_nd:{4}".format(y,U,D,u_y,u_y_nd))
    #print("u_poiseuille: y:{0}; u_y:{1}; u_poiseuille:{2}".format(y,u_y,u_poiseuille))
    return u_poiseuille


#apply bounce-back conditions on upper and lower boundaries of pipe
def bounceBackTopBottom(_ltc, _ltc_bc, _index_mapping_top, _index_mapping_bottom):
    # row indices for top and bottom boundaries
    upper_boundary = 0
    lower_boundary = Ny

    # swaps on the top boundary
    for i, swap_needed in enumerate(_index_mapping_top):
        if swap_needed:
            _ltc_bc[upper_boundary, :, channel_indices[i]] = _ltc[upper_boundary, :, antichannel_indices[i]]
    
    # swaps on the bottom boundary
    for i, swap_needed in enumerate(_index_mapping_bottom):
        if swap_needed:
            _ltc_bc[lower_boundary, :, channel_indices[i]] = _ltc[lower_boundary, :, antichannel_indices[i]]
    
    return _ltc_bc  


#calulate boundary nodes X(0) and X(N+1) for periodic BC with presssure difference
def calculateXtraNodeLayers(_roh_N, u_avg_N, _roh_inlet, _roh_1, u_avg_1, _roh_outlet, _ltc): 
    #N==Nx+1
    #1==1
    #X(0) = X(0)
    #X(N+1) = X(Nx+2)

    #X0 : fi_prestream(X0,y,t) = fi_eq(roh_inlet, u_avg(Nx+1)) + [fi_prestream(Nx+1,y,t) - fi_eq(Nx+1,y,t)]
    fi_in = f_eq_slc(Nx+1, _roh_inlet, u_avg_N)
    fi_prestream = _ltc[:,Nx+1,:]
    fi_eq_N = f_eq_slc(Nx+1, _roh_N, u_avg_N)
    xNodes_inlet = fi_in + (fi_prestream - fi_eq_N)

    #Nx+2 : fi_prestream(XNx+2,y,t) = fi_eq(roh_outlet, u_avg_1) + [fi_prestream(X1,y,t) - fi_eq(X1,y,t)]
    fi_out = f_eq_slc(1, _roh_outlet, u_avg_1)
    fi_prestream = _ltc[:,1,:]
    fi_eq_1 = f_eq_slc(1, _roh_1, u_avg_1)
    xNodes_outlet = fi_out + (fi_prestream - fi_eq_1) 

    return xNodes_inlet, xNodes_outlet


def print_selected_columns(arr, number_of_central_rows, columns):
    num_dims = arr.ndim

    if num_dims == 1:
        Ny = arr.size
        if len(columns) > 1:
            raise ValueError("Only one column can be specified for 1D arrays.")
        
        central_row_start = (Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        print(f"arr in column {columns[0]} (middle rows):")
        for i in range(central_row_start, central_row_end):
            print(arr[i])

    elif num_dims == 2:
        Ny, Nx = arr.shape
        central_row_start = (Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        # header
        header = ["Row"] + [f"Column {col}" for col in columns]
        print("\t".join(header))

        for row in range(central_row_start, central_row_end):
            row_data = [row]  
            for col in columns:
                if col < 0 or col >= Nx:
                    raise IndexError(f"Column index {col} is out of bounds for 2D array with shape {arr.shape}.")
                row_data.append(arr[row, col])
            print("\t".join(map(str, row_data)))

    elif num_dims == 3:
        Ny, Nx_plus_3, inner_dim = arr.shape
        Nx = Nx_plus_3 - 3
        central_row_start = (Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        # header
        header = ["Row"] + [f"Column {col} (all inner elements)" for col in columns]
        print("\t".join(header))

        for row in range(central_row_start, central_row_end):
            row_data = [row] 
            for col in columns:
                if col < 0 or col >= Nx_plus_3:
                    raise IndexError(f"Column index {col} is out of bounds for 3D array with shape {arr.shape}.")
                if isinstance(arr[row, col], (list, np.ndarray)):
                    inner_elements = ", ".join(map(str, arr[row, col]))
                    row_data.append(f"[{inner_elements}]")
                else:
                    row_data.append(arr[row, col])
            print("\t".join(map(str, row_data)))

    else:
        raise ValueError("Input array must be 1D, 2D, or 3D.")
    
    print("\n")


#preliminary
t = 0
roh_inlet[:] = roh_in
roh_outlet[:] = roh_out

#initialise
u_avg = np.zeros((Ny+1,Nx+3, 2), dtype=float)
roh_lattice = np.full((Ny+1, 1), INIT_DENSITY, dtype=float)
_roh_N = np.full((Ny+1), INIT_DENSITY, dtype=float)
_roh_1 = np.full((Ny+1), INIT_DENSITY, dtype=float)
prestream_lattice = f_eq_slc(0, roh_lattice, u_avg)
index = 0
_index_mapping_top = np.array([False, False, True, False, False, True, True, False, False])
_index_mapping_bottom = np.array([False, False, False, False, True, False, False, True, True])

#1st prestreamed lattice
print_selected_columns(u_avg, 4, [1, Nx // 2, Nx])
print_selected_columns(prestream_lattice, 4, [1, Nx // 2, Nx])

while t < TOTAL_TIME:
    print("\n\nindex: {0} ------------------------------------------------".format(index))

    #update extra node layers 0 and N+1 -> A)
    print("update extra node layers 0 and N+1 -> A): calculateXtraNodeLayers")    

    #lattice = np.full((Ny+1, Nx+3, 9), INIT_DENSITY, dtype=float)
    #uN==Nx+1
    u_avg_N = u_avg[:,Nx+1,:]
    #u1==1
    u_avg_1 = u_avg[:,1,:]
    xNodes_inlet, xNodes_outlet = calculateXtraNodeLayers(_roh_N, u_avg_N, roh_inlet, _roh_1, u_avg_1, roh_outlet, prestream_lattice)
    #assign inlet and outlet boundary values -> A)
    streamed_lattice[:,0,:] = xNodes_inlet
    print("prestream_lattice")
    print_selected_columns(xNodes_inlet, 1, [0,1,2])   
    streamed_lattice[:,Nx+2,:] = xNodes_outlet
    print("prestream_lattice")
    print_selected_columns(xNodes_outlet, 1, [0,1,2])   
    
    #stream lattice
    print("prestream_lattice")
    #copy upper and lower boundaries
    print("prestream_lattice:")
    print_selected_columns(prestream_lattice, 1, [0,1,2])
    streamed_lattice = streamLtc(prestream_lattice)
    print("streamed_lattice:")
    print_selected_columns(streamed_lattice, 1, [0,1,2])    

    #assign inlet and outlet boundary values -> B)
    streamed_lattice[:,0,:] = xNodes_inlet    
    
    #update boundary conditions
    print("update boundary conditions: _applyBounceBackTopBottom")    
    #streamed_lattice = _applyBounceBackTopBottom(streamed_lattice)
    _ltc_bc = streamed_lattice.copy()
    streamed_lattice = bounceBackTopBottom(streamed_lattice, _ltc_bc, _index_mapping_top, _index_mapping_bottom)
    del _ltc_bc
    gc.collect()
    
    #update moments at N
    print("update moments at N: updateMoments")  
    roh,current_density,u_avg = updateMoments(streamed_lattice)
    _roh_N = roh[:,Nx+1]                  
    _roh_1 = roh[:,1]
    #print_selected_columns(roh, 4, [1, Nx // 2, Nx])
    #print("avg exit velocity at middle streamline:")  
    #print_selected_columns(u_avg, 1, [Nx])
    #midstream_vel = u_avg[int(Ny/2),Nx]
    #outlet_midstream_avg_v.append(midstream_vel)

    #calulcate collision
    print("calculate collision")
    print("streamed_lattice:")
    print_selected_columns(streamed_lattice, 1, [0,1,2])
    feq = f_eq_slc(streamed_lattice, roh, u_avg)
    print("feq:")
    print_selected_columns(feq, 1, [0,1,2])
    collision_lattice = (1-omega_nd) * streamed_lattice + omega_nd * feq
    print("collision_lattice:")
    print_selected_columns(collision_lattice, 1, [0,1,2])

    #do not need this, can use lattice directly 
    prestream_lattice = collision_lattice.copy()
    del collision_lattice
    gc.collect()
    
    print_selected_columns(prestream_lattice, 4, [1, Nx // 2, Nx])


    t += dt
    index += 1
    print("Simulation Execution-> TO TAL_TIME:{0}; t:{1}; %:{2}".format(TOTAL_TIME, t, (t/TOTAL_TIME)*100.0))