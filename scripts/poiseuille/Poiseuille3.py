import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import matplotlib

print(matplotlib.__version__)

import math

import matplotlib.pyplot as plt
import numpy as np

TOTAL_TIME = 100

Cs=1/math.sqrt(3)
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
p_out=1
p_in=p_out+dP
roh_in=p_in/Cs**2
roh_out=p_out/Cs**2
print("p_in: {0}".format(p_in))
print("p_out: {0}".format(p_out))
print("roh_in: {0}".format(roh_in))
print("roh_out: {0}".format(roh_out))

nu=10**(-6) #m2/s
#nu=Cs**2(tau-1/2)(dx**2/dt)
#rearranged->dt=Cs**2(tau-1/2)(dx**2/nu)
dt=Cs**2*(tau-1/2)*(dx**2/nu)
print("dt: {0}".format(dt))

#Poiseuille max velocity
U=1/8*rho_0/nu*dP/L*D**2
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
#Re=DU/nu=>Re_nd=D_nd*U_nd/nu_nd
#=>nu_nd=(D_nd*U_nd/D*U)*nu=8*10**(-3)
nu_nd=((D_nd*U_nd)/(D*U))*nu
print("nu_nd: {0}".format(nu_nd))

#nu_nd=Cs**2*(tau_nd-1/2)
#=>tau_nd=(nu_nd/Cs**2)+1/2
tau_nd=(nu_nd/Cs**2)+1/2
print("tau_nd: {0}".format(tau_nd))
omega = dt/tau
print("omega: {0}".format(omega))
omega_nd = dt_nd/tau_nd
print("omega_nd: {0}".format(omega_nd))


# Define the discrete velocity_directions for D2Q9
discrete_velocities = np.array([[0, 0],   # i=0
                      [1, 0],   # i=1
                      [0, 1],   # i=2
                      [-1, 0],  # i=3
                      [0, -1],  # i=4
                      [1, 1],   # i=5
                      [-1, 1],  # i=6
                      [-1, -1], # i=7
                      [1, -1]]) # i=8

channel_indices = [0,1,2,3,4,5,6,7,8] #channel
antichannel_indices =[0,3,4,1,2,7,8,5,6] #anti-channel
print("Discrete velocities: {0}".format(discrete_velocities))

#define weights
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
print("Weights: {0}".format(weights))

#use named strcuture for values at each lattice node
# Create a 100x10x9 grid. The number of nodes will then be 103x11x9, 103 b/c of periodic pressure boundary, 11 for Ny x dy + 1, and 9 channels
#Nx+3 is due to periodic boundary conditions
lattice = np.ones((Ny+1, Nx+3, 9), dtype=object)
print("Shape of lattice:", lattice.shape)  # Should be (103,11,9)

# Create a new lattice to store the rolled grid
streamed_lattice = np.zeros_like(lattice)
collision_lattice = np.zeros_like(lattice)
prestream_lattice = np.zeros_like(lattice)
xtraNodes_inlet = lattice[:, 0:1, :]
xtraNodes_outlet = np.zeros_like(xtraNodes_inlet)
roh_inlet = np.zeros(Ny+1)
roh_outlet = np.zeros(Ny+1)


#Collision equilibrium distribution function feq(0)
def feq_0(PRINT, rho,u_avg):
    if(PRINT): print("rho: {0}".format(rho))
    if(PRINT): print("u_avg: {0}".format(u_avg))
    feq0 = weights[0] * rho * (1 - 3/2*(np.dot(u_avg,u_avg)/Cs**2))
    if(PRINT): print("feq(0): {0}".format(feq0))
    return feq0

#Collision equilibrium distribution function feq(1->8)
def feq_i(PRINT, i, rho, u_avg):
    c_i = discrete_velocities[i]

    #feq0_8 = weights[i] * rho * (1 + 3*(np.dot(c_i,u_avg)/(Cs**2)) + 9/2*(np.dot(c_i,u_avg)**2/(Cs**4)) - 3/2*(np.dot(u_avg,u_avg)/(Cs**2)))
    #feq0_8 = weights[i] * rho * (1 + 3*(np.dot(u_avg,c_i)/(Cs**2)) + 9/2*(np.dot(u_avg,c_i)**2/(Cs**4)) - 3/2*(np.dot(u_avg,u_avg.T)/(Cs**2)))
    dot = np.dot(u_avg,c_i)
    #dot = np.dot(c_i, u_avg.T)

    if(PRINT): print("c_i: {0}".format(c_i))
    if(PRINT): print("u_avg.shape: {0}".format(u_avg.shape))
    if(PRINT): print("u_avg: {0}".format(u_avg))
    det0 = dot
    if(PRINT): print("np.dot(c_i,u_avg): {0}".format(det0.shape))

    #feq0_8 = weights[i] * rho * (1 + 3*np.dot(u_avg,c_i)/(Cs**2) + (9/2)*np.dot(u_avg,c_i)**2/(Cs**4) - (3/2)*np.dot(u_avg,u_avg.T)/(Cs**2))
    det1 = 3*dot/(Cs**2)
    if(PRINT): print("det1.shape: {0}".format(det1.shape))
    if(PRINT): print("det1: {0}".format(det1))
    det2 = (9/2)*dot**2/(Cs**4)
    if(PRINT): print("det2.shape: {0}".format(det2.shape))
    if(PRINT): print("det2: {0}".format(det2))
    if u_avg.ndim == 2:
        dot_product_u_avg = np.einsum('ij,ij->i', u_avg, u_avg)
    elif u_avg.ndim == 3:
        dot_product_u_avg = np.einsum('ijk,ijk->ij', u_avg, u_avg)

    det3 = (3/2)*dot_product_u_avg/(Cs**2)
    
    if(PRINT): print("det3.shape: {0}".format(det3.shape))
    if(PRINT): print("det3: {0}".format(det3))
    _ones = np.ones(det1.shape)
        
    dets = (_ones + det1 + det2 - det3)
    feq0_8 = weights[i] * np.multiply(rho, dets)
        
    if(PRINT): print("feq(0-8): {0}".format(feq0_8))
    return feq0_8   

#Collision equilibrium distribution function feq(0-8) for initialisation
def f_eq_init(PRINT, _lattice, _roh, u_avg):
    # Applying it across all points (idy, idx) using slicing
    
    det1 = np.squeeze(np.stack([feq_i(PRINT, 0, _roh, u_avg)], axis=-1))
    #det1 = np.stack([feq_i(True, 0, roh_nd, u_avg)], axis=1)
    if PRINT: print("det1.shape: {}".format(det1.shape))
    if PRINT: print("det1: {}".format(det1))
    #det1_reshape = det1.reshape(Ny+1,1)
    _lattice[:, :, 0] = det1
    
    # For indices 1 to 4
    det2 = np.stack([feq_i(PRINT, i, _roh, u_avg) for i in range(1, 5)], axis=-1)
    #det2 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(1, 5)], axis=1)
    if PRINT: print("det2.shape: {}".format(det2.shape))
    if PRINT: print("det2: {}".format(det2))      
    #det2_broadcasted = np.tile(det2[:, np.newaxis, :], (1, Nx+3, 1))
    _lattice[:, :, 1:5] = det2    

    # For indices 5 to 8
    det3 = np.stack([feq_i(PRINT, i, _roh, u_avg) for i in range(5, 9)], axis=-1)
    #det3 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(5, 9)], axis=1)
    if PRINT: print("det3.shape: {}".format(det3.shape))
    if PRINT: print("det3: {}".format(det3))
    #det3_broadcasted = np.tile(det3[:, np.newaxis, :], (1, Nx+3, 1))
    _lattice[:, :, 5:9] = det3

    if PRINT: print("_lattice[:, :4, :]: {0}".format(_lattice[:, :4, :]))

    return _lattice

#Collision equilibrium distribution function feq(0-8) for each lattice roll
def f_eq_lattice(PRINT, _lattice, roh_nd, u_avg):
    ret_lattice = _lattice.copy()
    # Applying it across all points (idy, idx) using slicing
    #_lattice[:, :, 0] = feq_0(PRINT, roh_nd, u_avg)
    det1 = np.stack([feq_i(PRINT, 0, roh_nd, u_avg)], axis=-1)
    # For indices 1 to 4
    det2 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(1, 5)], axis=-1)
    # For indices 5 to 8
    det3 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(5, 9)], axis=-1)
    ret_lattice[:, :, 0] = np.squeeze(det1)
    ret_lattice[:, :, 1:5] = np.squeeze(det2)
    ret_lattice[:, :, 5:9] = np.squeeze(det3)
    if PRINT: print("_lattiret_latticece[:, :, :].shape: {}".format(ret_lattice[:, :, :].shape))
    if PRINT: print("ret_lattice[:, :, :]: {}".format(ret_lattice[:, :, :]))
    return ret_lattice

#Collision equilibrium distribution function feq(0-8) for each lattice slice at inlet/outlet
def f_eq_slice(PRINT, j, roh_nd, u_avg):
    # Applying it across all points (idy, idx) using slicing
    _slice = np.ones((Ny+1, 9), dtype=object)
    # For indices 0
    if PRINT: print("f_eq_slice->u_avg: {0}".format(u_avg))
    #det1 = np.stack([feq_i(PRINT, 0, roh_nd, u_avg)], axis=0)
    det1 = np.squeeze(np.stack([feq_i(PRINT, 0, roh_nd, u_avg)], axis=-1))
    _slice[:, 0] = det1
    # For indices 1 to 4
    #det2 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(1, 5)], axis=0)
    det2 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(1, 5)], axis=-1)
    _slice[:, 1:5] = det2
    # For indices 5 to 8
    #det3 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(5, 9)], axis=0)
    det3 = np.stack([feq_i(PRINT, i, roh_nd, u_avg) for i in range(5, 9)], axis=-1)
    _slice[:, 5:9] = det3
    if PRINT: print("_slice[:, :].shape: {}".format(_slice[:, :].shape))
    if PRINT: print("_slice[:, :]: {}".format(_slice[:, :]))
    return _slice

# Roll the lattice based on the discrete velocities
def streamLattice(PRINT, _lattice):
    shifted_lattice = np.zeros_like(_lattice) 

    # Vectorized loop over directions using list comprehension
    shifted_lattice = np.stack([
        np.roll(np.roll(lattice[:, :, direction], shift=dy, axis=1), shift=dx, axis=0)
        for direction, (dx, dy) in enumerate(discrete_velocities)
    ], axis=2)
            
    if PRINT: print("shifted_lattice[:, :, :].shape: {}".format(shifted_lattice[:, :, :].shape))

    if PRINT:
        # Define the number of directions to print
        k = 10
        
        # Print the lattice after rolling for direction 0
        print("\nAfter rolling (direction 0):")
        for i in range(9):
            print(f"Direction {i}:")
            print(shifted_lattice[i, :, :])  # Print the first 10 rows and columns
    
    return shifted_lattice

#update the first 2 moments (density, current density,  and u_avg
def updateMoments(PRINT, _lattice):
    roh = np.sum(_lattice, axis=2)
    roh_reshaped = roh[:, :, np.newaxis]
    if PRINT: print(f"Density-reshaped: {roh_reshaped}")
    current_density = np.tensordot(_lattice, discrete_velocities, axes=(2, 0))
    
    if PRINT: print(f"Current Density: {current_density}")
    u_avg = np.divide(current_density,roh_reshaped)
    if PRINT: print(f"Average velocity: {u_avg}")

    return roh,current_density,u_avg

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

#invert directions -> bounce off wall
#fi_(xb, t+dt) = fi_nd(xb, t)
def applyBounceBack(PRINT, _lattice, j):
    if PRINT: 
        print("\n\napplyBounceBack:")
        for z in range(0,9):
            print("_lattice-before[{0}]: {1}".format(z, _lattice[0, :10, z]))
    _lattice[:,j,:] = _lattice[:,j,antichannel_indices]
    if PRINT: 
        for z in range(0,9):    
            print("_lattice-after[{0}]: {1}".format(z, _lattice[0, :10, z]))
    return _lattice


def _applyBounceBackTopBottom(PRINT, _lattice):
    # Create a boundary mask only for the top and bottom boundaries
    boundary_mask = np.zeros((Ny+1, Nx+3), dtype=bool)
    # Top boundary: row index 0
    boundary_mask[0, :] = True  
    # Bottom boundary: row index Ny
    boundary_mask[Ny, :] = True  

    if PRINT: 
        print("\n\_applyBounceBackTopBottom:")
        for z in range(0,9):
            print("_lattice-before[{0}]: {1}".format(z, _lattice[0, :2, z]))

    # Apply bounce-back only at the top and bottom boundaries
    for channel, antichannel in zip(channel_indices, antichannel_indices):
        # Swap channel values with antichannel values at the top and bottom boundaries
        _lattice[boundary_mask, channel] = _lattice[boundary_mask, antichannel]

    if PRINT: 
        for z in range(0,9):
            print("_lattice-after[{0}]: {1}".format(z, _lattice[0, :2, z]))

    return _lattice

def calculateXtraNodeLayers(PRINT, _roh_N, u_avg_N, _roh_in, _roh_1, u_avg_1, _roh_out, _ltc): 
    #0 
    #u_avg_N = u_avg[:,Nx,:]
    feq = f_eq_slice(PRINT, Nx, _roh_N, u_avg_N)        
    fneq = _ltc[:,Nx,:]
    f01 = f_eq_slice(PRINT, Nx, _roh_in, u_avg_N)
    xtraNodes_inlet = f01 + (fneq - feq)

    #N+1
    #u_avg_1 = u_avg[:,1,:]
    feq = f_eq_slice(PRINT,  Nx, _roh_1, u_avg_1)
    fne = _ltc[:,1,:]
    fN1 = f_eq_slice(PRINT, 1, _roh_out, u_avg_1)
    xtraNodes_outlet = fN1 + (fne - feq) 

    return xtraNodes_inlet, xtraNodes_outlet


#preliminary
t = 0
PRINT = False
roh_inlet[:] = roh_in
print(f"roh_in: ({roh_inlet[:]})")
print(f"roh_in.shape: ({roh_inlet[:].shape})")
roh_outlet[:] = roh_out
print(f"roh_outlet: ({roh_outlet[:]})")
print(f"roh_outlet.shape: ({roh_outlet[:].shape})")


#initialise
u_avg = np.zeros((Ny+1,Nx+3, 2), dtype=float)
print(f"u_avg: ({u_avg})")
print(f"u_avg.shape: ({u_avg.shape})")

roh_lattice = np.ones((Ny+1, 1), dtype=float)
print(f"roh_lattice: ({roh_lattice})")
print(f"roh_lattice.shape: ({roh_lattice.shape})")

_roh_N = np.ones((Ny+1), dtype=float)
print(f"_roh_N: ({_roh_N})")
print(f"_roh_N.shape: ({_roh_N.shape})")

_roh_1 = np.ones((Ny+1), dtype=float)
print(f"_roh_1: ({_roh_1})")
print(f"_roh_1.shape: ({_roh_1.shape})")

lattice = f_eq_init(False, lattice, roh_lattice, u_avg)
print(f"lattice: ({lattice})")
print(f"lattice.shape: ({lattice.shape})")

#1st prestreamed lattice
prestream_lattice = lattice.copy()

while t < TOTAL_TIME:
    #update extra node layers 0 and N+1 -> A)
    print("update extra node layers 0 and N+1 -> A): calculateXtraNodeLayers")    
    #0
    u_avg_N = u_avg[:,Nx,:]
    #N+1
    u_avg_1 = u_avg[:,1,:]
    xtraNodes_inlet, xtraNodes_outlet = calculateXtraNodeLayers(PRINT, _roh_N, u_avg_N, roh_inlet, _roh_1, u_avg_1, roh_outlet, prestream_lattice)
    #xtraNodes_inlet, xtraNodes_outlet = calculateXtraNodeLayers(PRINT, roh_inlet, roh_outlet, u_avg, streamed_lattice, prestream_lattice) 
    
    #stream lattice
    print("stream lattice")    
    streamed_lattice = streamLattice(PRINT, prestream_lattice)

    #assign inlet and outlet boundary values -> B)
    streamed_lattice[:,0,:] = xtraNodes_inlet
    #streamed_lattice[Nx+1,:,:] = xtraNodes_outlet
    
    #update boundary conditions
    print("update boundary conditions: _applyBounceBackTopBottom")    
    streamed_lattice = _applyBounceBackTopBottom(PRINT, streamed_lattice)
    #streamed_lattice = _applyBounceBack(PRINT, streamed_lattice, 0)
    #streamed_lattice = _applyBounceBack(PRINT, streamed_lattice, Ny)
    
    #update moments at N
    print("update moments at N: updateMoments")  
    roh,current_density,u_avg = updateMoments(PRINT, streamed_lattice)

    #calulcate collision
    print("calculate collision")      
    obj1 = (1-omega_nd)*streamed_lattice + omega_nd*f_eq_lattice(PRINT, streamed_lattice, roh, u_avg)
    #obj2 = np.sum((1-omega_nd)*streamed_lattice, omega_nd*f_eq_lattice(False, streamed_lattice, roh, u_avg)) 
    collision_lattice = obj1

    #do not need this, can use lattice directly 
    prestream_lattice = collision_lattice.copy()

    t += dt
    print("Simulation Execution-> TOTAL_TIME:{0}; t:{1}; %:{2}".format(TOTAL_TIME, t, (t/TOTAL_TIME)*100.0))