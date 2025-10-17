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
VERBOSE1=True
VERBOSE2=False

Cs=math.sqrt(1/3.)
D=1e-3 #m
L=1 #m

D_nd=100

Ny=D_nd+1
Nx=int(Ny*L/D)

if(VERBOSE1): print("Ny: {0}".format(Ny))
if(VERBOSE1): print("Nx: {0}".format(Nx))

dx=D/D_nd #old->5*10**(-5)
dy = dx
if(VERBOSE1): print("dx: {0}".format(dx))
delta_t=1
#relaxation time tau, should be > 0,5
tau=0.6

dP=1e-2 #Pa
rho_0=1e3 #kg/m^3
p_in=1/3
p_out=p_in-dP
roh_in=p_in/Cs**2
roh_out=p_out/Cs**2
dRho=dP/Cs**2


if(VERBOSE1): 
    print("p_in: {0}".format(p_in))
    print("p_out: {0}".format(p_out))
    print("roh_in: {0}".format(roh_in))
    print("roh_out: {0}".format(roh_out))

nu=1e-6 #m^2/s 
dt=Cs**2*(tau-1./2)*(dx**2/nu)
if(VERBOSE1): print("dt: {0}".format(dt))

#Poiseuille centerline (max) velocity
U=1/8*(rho_0/nu)*(dP/L)*(D**2)
#U=1.25

Re=D*U/nu
Ma=U/Cs 
Kn=U*D/nu
if(VERBOSE1): 
    print("U: {0}".format(U))
    print("Re: {0}".format(Re))
    print("Ma: {0}".format(Ma))
    print("Kn: {0}".format(Kn))

#we need Cl, Croh, Ct

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
U_nd = U/Cu #-> limit U_nd=0.1
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

tau_nd=(nu_nd/Cs**2)+1./2
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

index = 0

boundary_values_top_ltc = []
boundary_values_top_pre_ltc = []
columns_to_select = [0, 1, 2, 3, 10, 20, Nx+1]
_roh_at_points_top = []
_roh_at_points_mid = []
_roh_at_points_bottom = []

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
index_mapping_top = np.array([False, False, True, False, False, True, True, False, False])
index_mapping_bottom = np.array([False, False, False, False, True, False, False, True, True])

#weights
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
if(VERBOSE1): print("Weights: {0}".format(weights))


#equilibrium distribution function feq(0->8)
def feq_2D(_rho, _u_ckl):
    _u_ckl_dot = np.dot(discrete_velocities, _u_ckl) 
    _u_ckl_product = np.einsum('ki,ki->i', _u_ckl, _u_ckl)

    _u_ckl_product_reshaped = _u_ckl_product.reshape(1, *(_u_ckl_product.shape))
    _ones = np.ones(_u_ckl_dot.shape)

    _rho_reshaped = _rho.reshape(1, *(_rho.shape))
    weights_reshaped = weights.reshape((9, 1)) * np.ones(_rho.shape)
    factors = weights_reshaped * _rho_reshaped

    feq = factors * (
        _ones + 3. * _u_ckl_dot / Cs**2 + (9. / 2.) * _u_ckl_dot**2 / Cs**4 - (3. / 2.) * _u_ckl_product_reshaped / Cs**2
    )

    return feq

def feq_3D(_rho, _u_ckl):
    _u_ckl_dot = np.einsum('hk,kij->hij', discrete_velocities, _u_ckl) #(9,2) * (2,101,101002)
    _u_ckl_product = np.einsum('kij,kij->ij', _u_ckl, _u_ckl)
    _u_ckl_product_reshaped = _u_ckl_product.reshape(1, *(_u_ckl_product.shape))
    _ones = np.ones(_u_ckl_dot.shape)

    _rho_reshaped = _rho.reshape(1, *(_rho.shape))
    weights_reshaped = weights.reshape((9, 1, 1)) * np.ones(_rho.shape)
    factors = weights_reshaped * _rho_reshaped

    feq = factors * (
        _ones + 3. * _u_ckl_dot / Cs**2 + (9. / 2.) * _u_ckl_dot**2 / Cs**4 - (3. / 2.) * _u_ckl_product_reshaped / Cs**2
    )

    return feq


#roll the lattice based on the discrete velocities
def streamLattice(_ltc):
    tmpLtc = [np.roll(np.roll(_ltc[direction, :, :], shift=-dy, axis=0), shift=dx, axis=1) for direction, (dx, dy) in enumerate(discrete_velocities)]
    shifted_lattice = np.stack(tmpLtc, axis=0) 
    
    return shifted_lattice


#update the first 2 moments (density, current density, _u_ckl)
def updateMoments(_ltc):
    # density:
    _roh = np.sum(_ltc, axis=0)

    _roh_at_points_top.append([index, _roh[0,columns_to_select]])
    _roh_at_points_mid.append([index, _roh[20,columns_to_select]])
    _roh_at_points_bottom.append([index, _roh[Ny-1,columns_to_select]])

    # current density:
    #current_density = np.tensordot(_ltc, discrete_velocities, axes=(0, 0))
    #current_density = current_density.transpose(2, 0, 1)

    _current_density2 =  np.einsum('ki,ijl->kjl', discrete_velocities.T, _ltc)

    # average velocity
    _u_ckl = _current_density2 / _roh

    return _roh, _current_density2, _u_ckl


#2D Poiseuille inlet velocity u(y) for comparison with numerical result
def Poiseuille2DUy(y):
    u_poiseuille = np.zeros((2), dtype=float)
    u_y=U*(1-((y-D/2)/(D/2))**2)
    u_poiseuille[0] = u_y
    u_poiseuille[1] = 0

    return u_poiseuille


#apply bounce-back conditions on upper and lower boundaries of pipe
def bounceBackTopBottom(_ltc, _ltc_pre, _index_mapping_top, _index_mapping_bottom):
    # row indices for top and bottom boundaries
    top_boundary = 0
    bottom_boundary = Ny-1

    _ltc_copy_4_debug = _ltc.copy()

    # swaps on the top boundary
    for i, swap_needed in enumerate(_index_mapping_top):
        if swap_needed:
            _ltc[channel_indices[i], top_boundary, :] = _ltc_pre[antichannel_indices[i], top_boundary, :]

    boundary_values_top_pre_ltc.append([index, _ltc_pre[:, top_boundary, :2]])
    boundary_values_top_ltc.append([index, _ltc[:, top_boundary, :2]])

    
    # swaps on the bottom boundary
    for i, swap_needed in enumerate(_index_mapping_bottom):
        if swap_needed:
            _ltc[channel_indices[i], bottom_boundary, :] = _ltc_pre[antichannel_indices[i], bottom_boundary, :]

    return _ltc


#calulate boundary nodes X(0) and X(N+1) for periodic BC with presssure difference
def calcPeriodicBC(_roh_N, _u_cNl, _roh_inlet, _roh_1, _u_c1l, _roh_outlet, _ltc): 
    #N==Nx+1
    #1==1
    #X(0) = X(0)
    #X(N+1) = X(Nx+2)

    #X0 : fi_prestream(X0,y,t) = fi_eq(roh_inlet, u_ckl(Nx+1)) + [fi_prestream(Nx+1,y,t) - fi_eq(Nx+1,y,t)]
    feq_in = feq_2D(_roh_inlet, _u_cNl)
    fi_prestream = _ltc[:,:,Nx]
    fi_eq_N = feq_2D(_roh_N, _u_cNl)
    fi_x0 = feq_in + (fi_prestream - fi_eq_N)

    #Nx+2 : fi_prestream(XNx+2,y,t) = fi_eq(roh_outlet, u_ckl) + [fi_prestream(X1,y,t) - fi_eq(X1,y,t)]
    feq_out = feq_2D(_roh_outlet, _u_c1l)
    fi_prestream = _ltc[:,:,1]
    fi_eq_1 = feq_2D(_roh_1, _u_c1l)
    fi_xNplus1 = feq_out + (fi_prestream - fi_eq_1) 

    return fi_x0, fi_xNplus1


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

        if number_of_central_rows==0 or number_of_central_rows==(d_Ny-1):
            print(arr[number_of_central_rows])
        else:
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
    print(name + "-top boundary:")
    print_selected_columns(arr, 1, [1])    
    print(name + "-central rows:")
    print_selected_columns(arr, 2, [0])
    print_selected_columns(arr, 2, [1])
    print_selected_columns(arr, 2, [Nx])
    print_selected_columns(arr, 2, [Nx+1])
    print(name + "-bottom boundary:")
    print_selected_columns(arr, Ny-1, [Nx+1])

def printSelLtcSlicePoints(index, name, arr):
    arr_reshaped = arr.transpose()
    print("\nindex: {0}".format(index))
    print(name + "-top boundary:")
    print_selected_columns(arr_reshaped, 0, [0,1,2,3,4,5,6,7,8])    
    print(name + "-central rows 2:")
    print_selected_columns(arr_reshaped, 2, [0,1,2,3,4,5,6,7,8])
    print(name + "-bottom boundary:")
    print_selected_columns(arr_reshaped, Ny-1, [0,1,2,3,4,5,6,7,8])

def printSelMoments(index, name, arr):
    #arr_reshaped = arr.transpose()
    print("\nindex: {0}".format(index))
    print(name + "-top boundary:")
    print_selected_columns(arr, 1, [1])    
    print(name + "-central rows:")
    print_selected_columns(arr, 2, [0])
    print_selected_columns(arr, 2, [1])
    print_selected_columns(arr, 2, [Nx])
    print_selected_columns(arr, 2, [Nx+1])
    print(name + "-bottom boundary:")
    print_selected_columns(arr, Ny-1, [Nx+1]) 

def printSelRohSlicePoints(index, name, arr):
    arr_reshaped = arr.transpose()
    print("\nindex: {0}".format(index))
    print(name + "-top boundary:")
    print_selected_columns(arr_reshaped, 0, [0])    
    print(name + "-central rows 2:")
    print_selected_columns(arr_reshaped, 2, [0])
    print(name + "-bottom boundary:")
    print_selected_columns(arr_reshaped, Ny-1, [0])     


#preliminary
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)
#lattice = np.ones((9, Ny, Nx), dtype=float)
pdf_test = np.ones((9, Ny, Nx+2), dtype=float) #2 xtra external boundaray points
pdf_test[:, :, :] = np.arange(9)[:, np.newaxis, np.newaxis]
roh, current_density, u_ckl = updateMoments(pdf_test)
pdf_test = np.zeros((9, Ny, Nx+2), dtype=float) #2 xtra external boundaray points
pdf_test[:,20,20] = np.arange(9)

#values = np.array([0, 0.5, 0.5, 1, 1, 1, 1, 2, 2])
#values = values.reshape(9, 1, 1)
#lattice = np.broadcast_to(values, (9, Ny, Nx))

# tmp storage arrays
roh_inlet = np.zeros(Ny)
roh_outlet = np.zeros(Ny)
outlet_midstream_avg_v = [] 
outlet_midstream_roh = [] 
outlet_roh = []

t = 0
roh_inlet[:] = roh_in
roh_outlet[:] = roh_out

#initialise
#average velocity, cartesion x,y-directions, k is y-position, l is x-position
u_ckl = np.zeros((2, Ny, Nx+2), dtype=float)
INIT_ROH = roh_out #0.001
roh = np.full((Ny, Nx+2), INIT_ROH, dtype=float)
#streamed_lattice = f_eq_slc(roh, u_ckl)
pdf = feq_3D(roh, u_ckl)


#roh = np.full((Ny, Nx+2), INIT_ROH, dtype=float)
#_roh_1 = roh[:,1]
#_roh_N = roh[:,Nx]

_roh_N = roh[:,Nx] #np.full((Ny), INIT_ROH, dtype=float)
_roh_1 = roh[:,1] #np.full((Ny), INIT_ROH, dtype=float)

#1st prestreamed lattice
#prestream_lattice = f_eq_slc(0, roh, u_ckl)
#streamed_lattice = prestream_lattice.copy()
#printSelLtcPoints(index, "initialised prestream_lattice", prestream_lattice)

TEST_BC=False           # BC inout&bounceback test ->1
TEST_STREAMING=False    # streaming test ->2
TEST_MOMENTS=False      # moments test ->3
TEST_COLLISION=False    # collision test ->4

#plotting
inlet_velocity = []
outlet_velocity = []

# Simulation parameters
num_iterations = int(TOTAL_TIME / dt) + 1
num_nodes1 = int(D/dy) + 1  # Number of nodes in inlet/outlet
num_nodes2 = columns_to_select

# Enable interactive mode
plt.ion()
fig, ax = plt.subplots(4, 1, figsize=(8, 9))

# Initial plots
roh_inlet_plot_0, = ax[0].plot([], [], label="roh_inlet top", color="green")
roh_inlet_plot_1, = ax[1].plot([], [], label="roh_inlet mid", color="blue")
velocity_inlet_plot_0, = ax[2].plot([], [], label="roh_inlet bottom", color="red")
collision_inlet_plot_1, = ax[3].plot([], [], label="Inlet Collision[1]", color="purple")

# Set titles, labels, and legends
ax[0].set_title("Inlet Density Profile Top")
ax[0].set_xlabel("Node Index")
ax[0].set_ylabel("Density")
ax[0].legend()

ax[1].set_title("Inlet Density Profile Mid")
ax[1].set_xlabel("Node Index")
ax[1].set_ylabel("Density")
ax[1].legend()

ax[2].set_title("Inlet Density Profile Bottom")
ax[2].set_ylabel("Density")
ax[2].set_xlabel("Node Index")
ax[2].legend()

ax[3].set_title("Inlet Collision Profile [1]")
ax[3].set_ylabel("Collision")
ax[3].set_xlabel("Node Index")
ax[3].legend()

plt.tight_layout()



while t < TOTAL_TIME:
    print("\n\nindex: {0} ------------------------------------------------".format(index))

    #1. moment update
    if index > 0:
        roh, current_density, u_ckl = updateMoments(pdf)
        #override the densities at 0 and N+1 using ideal gas law in isothermal ﬂuid ﬂow
        # p = cs**2 * roh 
        roh[:,0] = roh_in
        roh[:,Nx+1] = roh_out

        _roh_N = roh[:,Nx]
        _roh_1 = roh[:,1]  

    if np.any(roh > 1):
            # Get the locations where the density exceeds 1
            locations = np.argwhere(roh > 1)
            print(f"Instability detected at iteration {index + 1}")
            for loc in locations:
                print(f"Density exceeds 1 at location {tuple(loc)} with value {roh[tuple(loc)]}")


    #2. compute equilibrium
    #feq = f_eq_slc(roh, u_ckl)
    feq = feq_3D(roh, u_ckl)


    #3. collision term
    #collision_lattice = (1-omega_nd) * streamed_lattice + omega_nd * feq
    collision_lattice = pdf + omega_nd * (feq - pdf)
    pdf = collision_lattice.copy()


    #5.1a Periodic Boundary conditions inlet/outlet with pressure difference
    #update extra node layers 0 and N+1 -> A) & B) acc. to Script: Boundary Conditions for the Lattice Boltzmann Method
    #acc. to Kruger p174 Non-equilibrium populations are determined after collision
    u_cNl = u_ckl[:, :, Nx]
    u_c1l = u_ckl[:, :, 1]   
    u_c0l = u_ckl[:, :, 0]   
    #u_ckl profiles at outlet and inlet
    u_cN_1l = u_ckl[:, :, Nx-1]
    u_c2l = u_ckl[:, :, 2]    
    fi_x0, fi_xNplus1 = calcPeriodicBC(_roh_N, u_cNl, roh_inlet, _roh_1, u_c1l, roh_outlet, pdf)

    #assign inlet and outlet boundary values -> A)
    pdf[:, :, 0] = fi_x0
    pdf[:, :, Nx+1] = fi_xNplus1  

   
    #4. stream lattice 
    #store pre-stream boundary top and bottom values
    _ltc_pre = pdf.copy()
    #streamed_lattice = streamLtc_nested(streamed_lattice)
    pdf = streamLattice(pdf)
    #pdf_test = streamLattice(pdf_test)
   

    #5. Boundary Conditions
    #5.1b. assign inlet boundary values -> B)
    if index > 0:
        pdf[:, :, 0] = fi_x0
        pdf[:, :, Nx+1] = fi_xNplus1

    #5.2 Bounce-Back Top and Bottom
    pdf = bounceBackTopBottom(pdf, _ltc_pre, index_mapping_top, index_mapping_bottom)  


    #if index > 10:
    #    break

    # Update plots
    if len(_roh_at_points_top) > 0:
    
        roh_inlet_plot_0.set_xdata(num_nodes2)
        roh_inlet_plot_0.set_ydata(_roh_at_points_top[-1][1])

        roh_inlet_plot_1.set_xdata(num_nodes2)
        roh_inlet_plot_1.set_ydata(_roh_at_points_mid[-1][1])

        velocity_inlet_plot_0.set_xdata(num_nodes2)
        velocity_inlet_plot_0.set_ydata(_roh_at_points_bottom[-1][1])

        collision_inlet_plot_1.set_xdata(range(num_nodes1))
        collision_inlet_plot_1.set_ydata(collision_lattice[0,:,0])

        # Adjust plot limits dynamically
        for axis in ax:
            axis.relim()
            axis.autoscale_view()

        # Redraw the entire figure canvas to prevent blank areas
        fig.canvas.draw_idle()

        # Pause to allow plot to update
        plt.pause(0.1)

    t += dt
    index += 1
    print("Simulation Execution-> TO TAL_TIME:{0}; t:{1}; %:{2}".format(TOTAL_TIME, t, (t/TOTAL_TIME)*100.0))
    print("roh[0:10,0]: {0}; {1}; {2}".format(roh[0,2],roh[0,3],roh[0,4]))
    print("roh[0:10,1]: {0}; {1}; {2}".format(roh[1,2],roh[1,3],roh[1,4]))
    print("roh[0:10,2]: {0}; {1}; {2}".format(roh[2,2],roh[2,3],roh[2,4]))

# Turn off interactive mode and display the final plot
plt.ioff()
plt.show()