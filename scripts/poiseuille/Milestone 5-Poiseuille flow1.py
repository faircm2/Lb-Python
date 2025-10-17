import gc
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import matplotlib
import matplotlib.pyplot as plt

print(matplotlib.__version__)

import math

import numpy as np

TOTAL_TIME = 0.01/4.
INIT_DENSITY = 1. #0.001
VERBOSE1=False
VERBOSE2=False

TEST_BC=False           # BC inout&bounceback test ->1
TEST_STREAMING=False    # streaming test ->2
TEST_MOMENTS=True      # moments test ->3
TEST_COLLISION=False    # collision test ->4

Cs=1/math.sqrt(3)
#Cs=math.sqrt(3)
if(VERBOSE1): print("Cs: {0}".format(Cs))

D=1e-3 #m
if(VERBOSE1): print("D: {0}".format(D))
D_nd=100
if(VERBOSE1): print("D_nd: {0}".format(D_nd))
L=0.1 #m

Ny=D_nd
if(VERBOSE1): print("Ny: {0}".format(Ny))
Nx=int(Ny*L/D)
if(VERBOSE1): print("Nx: {0}".format(Nx))

dx=D/D_nd #old->5*10**(-5)
dy = dx
if(VERBOSE1): print("dx: {0}".format(dx))
delta_t=1
#relaxation time tau, should be > 0,5
tau=0.6
dP=1e-2 #Pa
rho_0=1e3 #kg/m^3
dRho=dP/Cs**2

p_out=1.
p_in=p_out+dP
#roh_in=p_in/Cs**2
roh_out=1.
roh_in=roh_out+dRho

if(VERBOSE1): 
    print("p_in: {0}".format(p_in))
    print("p_out: {0}".format(p_out))
    print("roh_in: {0}".format(roh_in))
    print("roh_out: {0}".format(roh_out))

nu=3e-6 #m^2/s 
dt=Cs**2*(tau-1./2)*(dx**2/nu)
if(VERBOSE1): print("dt: {0}".format(dt))

#Poiseuille centerline (max) velocity
U=1/8*(rho_0/nu)*(dP/L)*(D**2)
U=1.25

#TEST STABILITY
#U=1
#TEST STABILITY
#dP = 0
roh_in_TEST=1.
roh_outTEST=1.    

Re=D*U/nu
Ma=U/Cs 
Kn=U*D/nu
if(VERBOSE1): 
    print("U: {0}".format(U))
    print("Re: {0}".format(Re))
    print("Ma: {0}".format(Ma))
    print("Kn: {0}".format(Kn))

# 1. Conversion factor Cl for length
Cl=dx #freely chosen
dx_nd=dx/Cl
if(VERBOSE1): 
    print("Cl: {0}".format(Cl))
    print("dx_nd: {0}".format(dx_nd))

#2. Conversion factor Croh for density
Croh=rho_0
roh_nd = rho_0/Croh
if(VERBOSE1): 
    print("Croh: {0}".format(Croh))
    print("roh_nd: {0}".format(roh_nd))

#3. Conversion factor Ct for time
Ct=dt
dt_nd = dt/Ct
if(VERBOSE1): 
    print("Ct: {0}".format(Ct))
    print("dt_nd: {0}".format(dt_nd))

#4. Conversion factor Cu for velocity
Cu=Cl/Ct
U_nd = 0.1 #old value->U/Cu
if(VERBOSE1): 
    print("Cu: {0}".format(Cu))
    print("U_nd: {0}".format(U_nd))

#5. Conversion factor CF for Force
CF=Croh*Cl**4*Ct**(-2)
if(VERBOSE1): print("CF: {0}".format(CF))

#6. Conversion factor Cf for frequency
Cf=1/Ct
if(VERBOSE1): print("Cf: {0}".format(Cf))

#change nu_nd in order to achieve U_nd=0,1
nu_nd=((D_nd*U_nd)/(D*U))*nu
if(VERBOSE1): print("nu_nd: {0}".format(nu_nd))

tau_nd=(nu_nd/Cs**2)+1/2
if(VERBOSE1): print("tau_nd: {0}".format(tau_nd))
omega = dt/tau
if(VERBOSE1): print("omega: {0}".format(omega))
omega_nd = dt_nd/tau_nd
if(VERBOSE1): print("omega_nd: {0}".format(omega_nd))


#UnitConverter information
print("--------------------- UnitConverter information ---------------------")
print("-- Parameters:")
print("Resolution: \t\t\t\tNx=\t\t{0}".format(Nx))
print("Lattice velocity (Cu): \t\t\tlatticeU=\t{0}".format(Cu))
print("Lattice relaxation frequency: \t\tomega=\t\t{0}".format(omega))
print("Lattice relaxation time: \t\ttau=\t\t{0}".format(omega))
print("Characteristical length(m): \t\tcharL=\t\t{0}".format(D))
print("Characteristical speed(m/s): \t\tU=\t\t{0}".format(U))
print("Phys. kinematic viscosity(m^2/s): \tnu=\t\t{0}".format(nu))
print("Phys. density(kg/m^d): \t\t\trho0=\t\t{0}".format(rho_0))
print("Characteristical pressure(N/m^2): \tp_in=\t\t{0}".format(p_in))
print("Characteristical inlet density: \troh_in=\t\t{0}".format(roh_in))
print("Characteristical outlet density: \troh_out=\t\t{0}".format(roh_out))
print("Mach number: \t\t\t\tmachNumber=\t{0}".format(Ma))
print("Reynolds number: \t\t\treynoldsNumber=\t{0}".format(Re))
print("Knudsen number: \t\t\tknudsenNumber=\t{0}".format(Kn))
print()
print("-- Conversion factors:")
print("Voxel length[Cl](m): \t\t\tphysDeltaX=\t{0}".format(Cl))
print("Time step[Ct](s): \t\t\tphysDeltaT=\t{0}".format(dt))
print("Velocity factor[Cu](m/s): \t\tphysVelocity=\t{0}".format(Cu))
print("Density factor[Croh](kg/m^3): \t\tphysDensity=\t{0}".format(Croh))
print("Mass factor(kg): \t\t\tphysMass=\t{0}".format(Croh))
print("Viscosity factor(m^2/s): \t\tphysViscosity=\t{0}".format(-1))
print("Force factor(N): \t\t\tphysForce=\t{0}".format(CF))
print("Pressure factor(N/m^2): \t\tphysPressure=\t{0}".format(-1))


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

if(VERBOSE1): print("Discrete velocities: {0}".format(discrete_velocities))
channel_indices = [0,1,2,3,4,5,6,7,8] #channel
antichannel_indices =[0,3,4,1,2,7,8,5,6] #anti-channel

#weights
weights = np.array([4./9,1./9,1./9,1./9,1./9,1./36,1./36,1./36,1./36])
if(VERBOSE1): print("Weights: {0}".format(weights))

#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)
lattice = np.full((9, Ny+1, Nx+3), INIT_DENSITY, dtype=float)

# tmp storage arrays
streamed_lattice = np.zeros_like(lattice)
collision_lattice = np.zeros_like(lattice)
prestream_lattice = np.zeros_like(lattice)
roh_inlet = np.zeros(Ny+1)
roh_outlet = np.zeros(Ny+1)
outlet_midstream_avg_v = [] 
outlet_midstream_roh = [] 

index = 0


#collision equilibrium distribution function feq(0->8)
def feq_i(i, rho, u_avg):
    c_i = discrete_velocities[i]

    if u_avg.ndim == 2:
        _dot = np.dot(c_i.T, u_avg) 
    elif u_avg.ndim == 3:
        _dot = np.einsum('k,kij->ij', c_i, u_avg)

    if u_avg.ndim == 2:
        dot_product_u_avg = np.einsum('ki,ki->i', u_avg, u_avg)
    elif u_avg.ndim == 3:
        dot_product_u_avg = np.einsum('kij,kij->ij', u_avg, u_avg)
        dot_product_u_avg1= np.sum(u_avg**2, axis=0)
        
    _ones = np.ones(_dot.shape)

    feq0_8 = weights[i] * rho * (
        _ones + 3. * _dot / Cs**2 + (9. / 2) * _dot**2 / Cs**4 - (3. / 2) * dot_product_u_avg / Cs**2
    )

    if False:
        # detailed computation
        _part1 = 3. * _dot / Cs**2
        _part2 = (9. / 2) * _dot**2 / Cs**4
        _part3 = (3. / 2) * dot_product_u_avg / Cs**2
        dets = _ones + _part1 + _part2 - _part3
        feq0_8 = weights[i] * rho * dets

    return feq0_8


#collision equilibrium distribution function feq(0-8) for each lattice slice at inlet/outlet
def f_eq_slc(j, roh_nd, u_avg):
    _slice = np.stack([feq_i(i, roh_nd, u_avg) for i in range(0, 9)], axis=0) 
    return _slice


#roll the lattice based on the discrete velocities
def streamLtc(_ltc):
    tmpLtc = [np.roll(np.roll(_ltc[direction, :, :], shift=dy, axis=0), shift=dx, axis=1) for direction, (dx, dy) in enumerate(discrete_velocities)]
    shifted_lattice = np.stack(tmpLtc, axis=0) 
    
    return shifted_lattice


#update the first 2 moments (density, current density, u_avg)
def updateMoments(u_avg_TEST, _ltc):
    # density:
    roh = np.sum(_ltc, axis=0)
    roh_reshaped = roh[np.newaxis, :, :]

    '''
    roh_reshaped0 = np.zeros((1, _ltc.shape[1], _ltc.shape[2]))
    for i in range(0,_ltc.shape[1]):
        for j in range(0,_ltc.shape[2]):
            sum = 0
            for k in range(0,9):
                sum += _ltc[k,i,j]
            roh_reshaped0[0,i,j] = sum
    roh_reshaped = roh_reshaped0
    '''
    printSelMoments(index, "roh", roh_reshaped)


    # current density:
    #discrete_velocities_reshaped = discrete_velocities[:, :, np.newaxis, np.newaxis]
    #_ltc_reshaped = _ltc[:, np.newaxis, :, :]
    #current_density_arr0 = _ltc_reshaped * discrete_velocities_reshaped
    #current_density = np.sum(current_density_arr0, axis=0)
    current_density2 = np.einsum('kl,kij->lij', discrete_velocities, _ltc)

    '''
    current_density_arr0 = np.zeros((2, _ltc.shape[1], _ltc.shape[2]))
    for i in range(0,_ltc.shape[1]):
        for j in range(0,_ltc.shape[2]):
            sumx = 0
            sumy = 0
            for k in range(0,9):
                vec = _ltc[k,i,j] * discrete_velocities[k]
                sumx += vec[0]
                sumy += vec[1]
            current_density_arr0[:,i,j] = [sumx, sumy]

    #current_density = current_density_arr0
    '''
    printSelMoments(index, "current_density", current_density2)


    # average velocity
    u_avg = current_density2 / roh_reshaped
    #u_avg = u_avg_TEST
    printSelMoments(index, "u_avg", u_avg)

    return roh, current_density2, u_avg


#2D Poiseuille inlet velocity u(y) for comparison with numerical result
def Poiseuille2DUy(y):
    u_poiseuille = np.zeros((2), dtype=float)
    u_y=U*(1-((y-D/2)/(D/2))**2)
    u_poiseuille[0] = u_y
    u_poiseuille[1] = 0

    return u_poiseuille


#apply bounce-back conditions on upper and lower boundaries of pipe
def bounceBackTopBottom(_ltc, _index_mapping_top, _index_mapping_bottom):
    # row indices for top and bottom boundaries
    upper_boundary = 0
    lower_boundary = Ny

    # swaps on the top boundary
    for i, swap_needed in enumerate(_index_mapping_top):
        if swap_needed:
            _ltc[channel_indices[i], upper_boundary, :] = _ltc[antichannel_indices[i], upper_boundary, :]
    
    # swaps on the bottom boundary
    for i, swap_needed in enumerate(_index_mapping_bottom):
        if swap_needed:
            _ltc[channel_indices[i], lower_boundary, :] = _ltc[antichannel_indices[i], lower_boundary, :]
    
    return _ltc


#calulate boundary nodes X(0) and X(N+1) for periodic BC with presssure difference
def calculateXtraNodeLayers(_roh_N, u_avg_N, _roh_inlet, _roh_1, u_avg_1, _roh_outlet, _ltc): 
    #N==Nx+1
    #1==1
    #X(0) = X(0)
    #X(N+1) = X(Nx+2)

    #X0 : fi_prestream(X0,y,t) = fi_eq(roh_inlet, u_avg(Nx+1)) + [fi_prestream(Nx+1,y,t) - fi_eq(Nx+1,y,t)]
    fi_in = f_eq_slc(Nx+1, _roh_inlet, u_avg_N)
    fi_prestream = _ltc[:,:,Nx+1]
    fi_eq_N = f_eq_slc(Nx+1, _roh_N, u_avg_N)
    xNodes_inlet = fi_in + (fi_prestream - fi_eq_N)

    #Nx+2 : fi_prestream(XNx+2,y,t) = fi_eq(roh_outlet, u_avg_1) + [fi_prestream(X1,y,t) - fi_eq(X1,y,t)]
    fi_out = f_eq_slc(1, _roh_outlet, u_avg_1)
    fi_prestream = _ltc[:,:,1]
    fi_eq_1 = f_eq_slc(1, _roh_1, u_avg_1)
    xNodes_outlet = fi_out + (fi_prestream - fi_eq_1) 

    return xNodes_inlet, xNodes_outlet


def print_min_max(arr):
    # overall minimum and maximum values for input array
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # print  minimum and maximum
    if(VERBOSE1): 
        print(f"Array Shape: {arr.shape}")
        print(f"Min Value: {arr_min}")
        print(f"Max Value: {arr_max}")
        print("\n")    


def print_selected_columns(arr, number_of_central_rows, columns):
    num_dims = arr.ndim
    if num_dims == 1:
        d_Ny = arr.size
        if len(columns) > 1:
            raise ValueError("only one column for 1D array")
        
        central_row_start = (d_Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        print(f"arr in column {columns[0]} (middle rows):")
        for i in range(central_row_start, central_row_end):
            print(arr[i])

    elif num_dims == 2:
        d_Ny, d_Nx = arr.shape
        central_row_start = (d_Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        # header
        header = ["Row"] + [f"Column {col}" for col in columns]
        print("\t".join(header))

        if number_of_central_rows==0 or number_of_central_rows==(d_Ny-1):
            row_data = [number_of_central_rows] 
            for col in columns:
                if col < 0 or col >= d_Nx:
                    raise IndexError(f"column index {col} is out of bounds for 2D array with shape {arr.shape}.")
                row_data.append(arr[number_of_central_rows, col])
            print("\t".join(map(str, row_data)))
        else:
            for row in range(central_row_start, central_row_end):
                row_data = [row]  
                for col in columns:
                    if col < 0 or col >= d_Nx:
                        raise IndexError(f"column index {col} is out of bounds for 2D array with shape {arr.shape}.")
                    row_data.append(arr[row, col])
                print("\t".join(map(str, row_data)))

    elif num_dims == 3:
        d_inner_dim, d_Ny, d_Nx_plus_3  = arr.shape
        d_Nx = d_Nx_plus_3 - 3
        central_row_start = (d_Ny - number_of_central_rows) // 2
        central_row_end = central_row_start + number_of_central_rows

        # header
        header = ["Row"] + [f"column {col} (all inner elements)" for col in columns]
        print("\t".join(header))

        if number_of_central_rows==0 or number_of_central_rows==(d_Ny-1):
            row_data = [number_of_central_rows] 
            for col in columns:
                if col < 0 or col >= d_Nx_plus_3:
                    raise IndexError(f"column index {col} is out of bounds for 3D array with shape {arr.shape}.")
                if isinstance(arr[:,number_of_central_rows,col], (list, np.ndarray)):
                    inner_elements = ", ".join(map(str, arr[:,number_of_central_rows,col]))
                    row_data.append(f"\t[{inner_elements}]")
                else:
                    row_data.append(arr[:,number_of_central_rows,col])
            print("\t".join(map(str, row_data)))            
        else:
            for row in range(central_row_start, central_row_end):
                row_data = [row] 
                for col in columns:
                    if col < 0 or col >= d_Nx_plus_3:
                        raise IndexError(f"column index {col} is out of bounds for 3D array with shape {arr.shape}.")
                    if isinstance(arr[:,row,col], (list, np.ndarray)):
                        inner_elements = ", ".join(map(str, arr[:,row,col]))
                        row_data.append(f"\t[{inner_elements}]")
                    else:
                        row_data.append(arr[:,row,col])
                print("\t".join(map(str, row_data)))            

    else:
        raise ValueError("input array must be 1D, 2D, or 3D.")
    

def printSelLtcPoints(index, name, arr):
    print("\nindex: {0}".format(index))
    print(name + ":")
    print_selected_columns(arr, 2, [0])
    print_selected_columns(arr, 2, [1])
    print_selected_columns(arr, 2, [Nx+1])
    print_selected_columns(arr, 2, [Nx+2])
    print(name + "-top boundary:")
    print_selected_columns(arr, 1, [1])
    print(name + "-bottom boundary:")
    print_selected_columns(arr, Ny, [Nx+2])

def printSelLtcSlicePoints(index, name, arr):
    arr_reshaped = arr.transpose()
    print("\nindex: {0}".format(index))
    print(name + f" [central rows 2]:")
    print_selected_columns(arr_reshaped, 2, [0,1,2,3,4,5,6,7,8])
    print(name + "-top boundary:")
    print_selected_columns(arr_reshaped, 0, [0,1,2,3,4,5,6,7,8])
    print(name + "-bottom boundary:")
    print_selected_columns(arr_reshaped, Ny, [0,1,2,3,4,5,6,7,8])

def printSelMoments(index, name, arr):
    #arr_reshaped = arr.transpose()
    print("\nindex: {0}".format(index))
    print(name + ":")
    print_selected_columns(arr, 2, [0])
    print_selected_columns(arr, 2, [1])
    print_selected_columns(arr, 2, [Nx+1])
    print_selected_columns(arr, 2, [Nx+2])
    print(name + "-top boundary:")
    print_selected_columns(arr, 1, [1])
    print(name + "-bottom boundary:")
    print_selected_columns(arr, Ny, [Nx+2]) 


#preliminary
t = 0
roh_inlet[:] = roh_in
roh_outlet[:] = roh_out
_index_mapping_top = np.array([False, False, True, False, False, True, True, False, False])
_index_mapping_bottom = np.array([False, False, False, False, True, False, False, True, True])

#initialise
u_avg = np.zeros((2, Ny+1, Nx+3), dtype=float)
u_avg[0,:,:] = 2.
u_avg[1,:,:] = 3.
u_avg_TEST = u_avg.copy()

roh_lattice = np.full((Ny+1, Nx+3), INIT_DENSITY, dtype=float)
roh=roh_lattice
_roh_N = np.full((Ny+1), INIT_DENSITY, dtype=float)
_roh_1 = np.full((Ny+1), INIT_DENSITY, dtype=float)

#1st prestreamed lattice
prestream_lattice = f_eq_slc(0, roh_lattice, u_avg)
streamed_lattice = prestream_lattice.copy()
printSelLtcPoints(index, "initialised prestream_lattice", prestream_lattice)

while t < TOTAL_TIME:
    print("\n\nindex: {0} ------------------------------------------------".format(index))

    #update extra node layers 0 and N+1 -> A) & B)
    #uN==Nx+1
    u_avg_N = u_avg[:, :, Nx+1]
    #u1==1
    u_avg_1 = u_avg[:, :, 1]
    xNodes_inlet, xNodes_outlet = calculateXtraNodeLayers(_roh_N, u_avg_N, roh_inlet, _roh_1, u_avg_1, roh_outlet, prestream_lattice)
    #assign inlet and outlet boundary values -> A)
    if TEST_STREAMING:
        printSelLtcPoints(index, "prestream_lattice BEFORE periodicBC", prestream_lattice)
    streamed_lattice[:, :, 0] = xNodes_inlet
    streamed_lattice[:, :, Nx+2] = xNodes_outlet
    if TEST_STREAMING:
        printSelLtcPoints(index, "streamed_lattice AFTER periodicBC", streamed_lattice)    
    #if TEST_MOMENTS:
    #    printSelLtcSlicePoints(index, "xNodes_inlet AFTER periodicBC", xNodes_inlet)
    #    printSelLtcSlicePoints(index, "xNodes_outlet AFTER periodicBC", xNodes_outlet)

    if TEST_BC:
        streamed_lattice[:, :, 0] = 1.
        streamed_lattice[:, :, Nx+2] = 1.   
    
    #stream lattice
    streamed_lattice = streamLtc(streamed_lattice)
    if TEST_BC:
        streamed_lattice = prestream_lattice
    if TEST_STREAMING:
        printSelLtcPoints(index, "streamed_lattice-streamed", streamed_lattice)
    if(VERBOSE2): 
        printSelLtcPoints(index, "streamed_lattice-streamed", streamed_lattice)

    #assign inlet and outlet boundary values -> B)
    streamed_lattice[:, :, 0] = xNodes_inlet
    if TEST_BC:
        streamed_lattice[:, :, 0] = 1.    
    
    #update boundary conditions
    if TEST_BC or TEST_STREAMING:
        streamed_lattice_test = streamed_lattice.copy()
    streamed_lattice = bounceBackTopBottom(streamed_lattice, _index_mapping_top, _index_mapping_bottom)
    if TEST_BC or TEST_STREAMING:
        streamed_lattice = streamed_lattice_test
    if TEST_STREAMING:
        printSelLtcPoints(index, "streamed_lattice-bouncebackBC", streamed_lattice)        
    
    #update moments at N
    #if TEST_MOMENTS:
    #    printSelLtcPoints(index, "pre-updateMoments: roh", roh)
    #    printSelLtcPoints(index, "pre-updateMoments: u_avg", u_avg)   
    roh,current_density,u_avg = updateMoments(u_avg_TEST, streamed_lattice)
    _roh_N = roh[:,Nx+1]                  
    _roh_1 = roh[:,1]
    midstream_vel = u_avg[:,int(Ny/2),Nx+1]
    outlet_midstream_avg_v.append([t, midstream_vel[0]])
    outlet_midstream_roh.append([t, _roh_N[int(Ny/2)]])
    if TEST_BC or TEST_STREAMING:
        roh[:] = 1.
        u_avg = u_avg_TEST

    #if TEST_MOMENTS:
    #    printSelLtcPoints(index, "post-updateMoments: roh", roh)
    #    printSelLtcPoints(index, "post-updateMoments: u_avg", u_avg)   

    #calculate collision
    feq = f_eq_slc(streamed_lattice, roh, u_avg)
    
    collision_lattice = (1-omega_nd) * streamed_lattice + omega_nd * feq
    if TEST_BC or TEST_STREAMING:
        collision_lattice = streamed_lattice
    if(VERBOSE1): 
        print("calculate collision")
        print_min_max(collision_lattice) 

    #if TEST:
    #collision_lattice = streamed_lattice.copy()
    if(VERBOSE2): 
        printSelLtcPoints(index, "collision_lattice", collision_lattice)

    #only use collision_lattice and copy for documentation 
    prestream_lattice = collision_lattice.copy()

    if TEST_BC or TEST_STREAMING:
        printSelLtcPoints(index, "roh", roh)
        printSelLtcPoints(index, "u_avg", u_avg)    

    t += dt
    index += 1
    print("Simulation Execution-> TO TAL_TIME:{0}; t:{1}; %:{2}".format(TOTAL_TIME, t, (t/TOTAL_TIME)*100.0))


time_values = [pair[0] for pair in outlet_midstream_avg_v]
speed_values = [pair[1] for pair in outlet_midstream_avg_v]
density_values = [pair[1] for pair in outlet_midstream_roh]

fig, ax1 = plt.subplots()  # Primary y-axis

# Plot on the primary y-axis (density plot)
ax1.plot(time_values, density_values, color='b', label='Density')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Density', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Adjust primary y-axis limits with 10% padding
y1_min = min(density_values) * 0.99
y1_max = max(density_values) * 1.01
ax1.set_ylim(y1_min, y1_max)

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot on the secondary y-axis (speed plot)
ax2.plot(time_values, speed_values, color='r', label='Avg Speed')
ax2.set_ylabel('Avg Speed', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adjust secondary y-axis limits with 10% padding
y2_min = min(speed_values) * 0.9
y2_max = max(speed_values) * 1.1
ax2.set_ylim(y2_min, y2_max)

plt.title('Density and Avg Speed vs. Time')
fig.tight_layout()  # Adjust layout to prevent overlap
plt.grid(True)
plt.show(block=True)
