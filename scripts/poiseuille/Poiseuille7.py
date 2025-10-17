import gc
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import matplotlib
import matplotlib.pyplot as plt

print(matplotlib.__version__)

import math

import numpy as np

TOTAL_TIME = 0.01
INIT_DENSITY = 1. #0.001
VERBOSE = True

Cs=1/math.sqrt(3)
#Cs=math.sqrt(3)
if(VERBOSE): print("Cs: {0}".format(Cs))

D=10**(-3) #m
if(VERBOSE): print("D: {0}".format(D))
D_nd=100
if(VERBOSE): print("D_nd: {0}".format(D_nd))
L=0.1

Ny=D_nd
if(VERBOSE): print("Ny: {0}".format(Ny))
Nx=int(Ny*L/D)
if(VERBOSE): print("Nx: {0}".format(Nx))

dx=D/D_nd #old->5*10**(-5)
dy = dx
if(VERBOSE): print("dx: {0}".format(dx))
delta_t=1
#relaxation time tau, should be > 0,5
tau=0.6
dP=10**(-2) #Pa
rho_0=10**3
dRho=dP/Cs**2

#TEST STABILITY
#dP = 0

p_out=1
p_in=p_out+dP
roh_in=p_in/Cs**2
roh_out=p_out/Cs**2
if(VERBOSE): 
    print("p_in: {0}".format(p_in))
    print("p_out: {0}".format(p_out))
    print("roh_in: {0}".format(roh_in))
    print("roh_out: {0}".format(roh_out))

nu=3*10**(-6) #m2/s
dt=Cs**2*(tau-1./2)*(dx**2/nu)
if(VERBOSE): print("dt: {0}".format(dt))

#Poiseuille max velocity
U=1/8*rho_0/nu*dP/L*D**2

#TEST STABILITY
#U=1

Re=D*U/nu
if(VERBOSE): 
    print("U: {0}".format(U))
    print("Re: {0}".format(Re))


# 1. Conversion factor Cl for length
Cl=dx #freely chosen
dx_nd=dx/Cl
if(VERBOSE): 
    print("Cl: {0}".format(Cl))
    print("dx_nd: {0}".format(dx_nd))

#2. Conversion factor Croh for density
Croh=rho_0
roh_nd = rho_0/Croh
if(VERBOSE): 
    print("Croh: {0}".format(Croh))
    print("roh_nd: {0}".format(roh_nd))

#3. Conversion factor Ct for time
Ct=dt
dt_nd = dt/Ct
if(VERBOSE): 
    print("Ct: {0}".format(Ct))
    print("dt_nd: {0}".format(dt_nd))

#4. Conversion factor Cu for time
Cu=Cl/Ct
U_nd = 0.1 #old value->U/Cu
if(VERBOSE): 
    print("Cu: {0}".format(Cu))
    print("U_nd: {0}".format(U_nd))

#5. Conversion factor CF for Force
CF=Croh*Cl**4*Ct**(-2)
if(VERBOSE): print("CF: {0}".format(CF))

#6. Conversion factor Cf for frequency
Cf=1/Ct
if(VERBOSE): print("Cf: {0}".format(Cf))

#change nu_nd in order to achieve U_nd=0,1
nu_nd=((D_nd*U_nd)/(D*U))*nu
if(VERBOSE): print("nu_nd: {0}".format(nu_nd))

tau_nd=(nu_nd/Cs**2)+1/2
if(VERBOSE): print("tau_nd: {0}".format(tau_nd))
omega = dt/tau
if(VERBOSE): print("omega: {0}".format(omega))
omega_nd = dt_nd/tau_nd
if(VERBOSE): print("omega_nd: {0}".format(omega_nd))

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

if(VERBOSE): print("Discrete velocities: {0}".format(discrete_velocities))
channel_indices = [0,1,2,3,4,5,6,7,8] #channel
antichannel_indices =[0,3,4,1,2,7,8,5,6] #anti-channel

#weights
weights = np.array([4./9,1./9,1./9,1./9,1./9,1./36,1./36,1./36,1./36])
if(VERBOSE): print("Weights: {0}".format(weights))

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
    _slice = np.stack([feq_i(i, roh_nd, u_avg) for i in range(0, 9)], axis=0) #-1
    return _slice


#roll the lattice based on the discrete velocities
def streamLtc(_ltc):
    tmpLtc = [np.roll(np.roll(_ltc[direction, :, :], shift=dy, axis=0), shift=dx, axis=1) for direction, (dx, dy) in enumerate(discrete_velocities)]
    shifted_lattice = np.stack(tmpLtc, axis=0) #2
    
    return shifted_lattice


#update the first 2 moments (density, current density, u_avg)
def updateMoments(_ltc):
    # density:
    roh = np.sum(_ltc, axis=0)
    roh_reshaped = roh[np.newaxis, :, :]

    # current density:
    discrete_velocities_reshaped = discrete_velocities[:, :, np.newaxis, np.newaxis]
    _ltc_reshaped = _ltc[:, np.newaxis, :, :]
    current_density_arr = _ltc_reshaped * discrete_velocities_reshaped
    current_density = np.sum(current_density_arr, axis=0)

    # average velocity
    u_avg = current_density / roh_reshaped

    return roh, current_density, u_avg


#2D Poiseuille inlet velocity u(y) for comparison with numerical result
def Poiseuille2DUy(y):
    u_poiseuille = np.zeros((2), dtype=float)
    u_y=U*(1-((y-D/2)/(D/2))**2)
    u_poiseuille[0] = u_y
    u_poiseuille[1] = 0

    return u_poiseuille


#apply bounce-back conditions on upper and lower boundaries of pipe
def bounceBackTopBottom(_ltc, _ltc_bc, _index_mapping_top, _index_mapping_bottom):
    # row indices for top and bottom boundaries
    upper_boundary = 0
    lower_boundary = Ny

    # swaps on the top boundary
    for i, swap_needed in enumerate(_index_mapping_top):
        if swap_needed:
            _ltc_bc[channel_indices[i], upper_boundary, :] = _ltc[antichannel_indices[i], upper_boundary, :]
    
    # swaps on the bottom boundary
    for i, swap_needed in enumerate(_index_mapping_bottom):
        if swap_needed:
            _ltc_bc[channel_indices[i], lower_boundary, :] = _ltc[antichannel_indices[i], lower_boundary, :]
    
    return _ltc_bc  


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
    if(VERBOSE): 
        print(f"Array Shape: {arr.shape}")
        print(f"Min Value: {arr_min}")
        print(f"Max Value: {arr_max}")
        print("\n")    



#preliminary
t = 0
roh_inlet[:] = roh_in
roh_outlet[:] = roh_out
_index_mapping_top = np.array([False, False, True, False, False, True, True, False, False])
_index_mapping_bottom = np.array([False, False, False, False, True, False, False, True, True])

#initialise
u_avg = np.zeros((2, Ny+1, Nx+3), dtype=float)
roh_lattice = np.full((Ny+1, Nx+3), INIT_DENSITY, dtype=float)
_roh_N = np.full((Ny+1), INIT_DENSITY, dtype=float)
_roh_1 = np.full((Ny+1), INIT_DENSITY, dtype=float)
index = 0

#1st prestreamed lattice
prestream_lattice = f_eq_slc(0, roh_lattice, u_avg)

while t < TOTAL_TIME:
    print("\n\nindex: {0} ------------------------------------------------".format(index))

    #update extra node layers 0 and N+1 -> A) & B)
    #uN==Nx+1
    u_avg_N = u_avg[:, :, Nx+1]
    #u1==1
    u_avg_1 = u_avg[:, :, 1]
    xNodes_inlet, xNodes_outlet = calculateXtraNodeLayers(_roh_N, u_avg_N, roh_inlet, _roh_1, u_avg_1, roh_outlet, prestream_lattice)
    #assign inlet and outlet boundary values -> A)
    streamed_lattice[:, :, 0] = xNodes_inlet
    if(VERBOSE): print("prestream_lattice")
    #print_selected_columns(xNodes_inlet, 1, [0,1,2])   
    if(VERBOSE): print_min_max(xNodes_inlet)
    streamed_lattice[:, :, Nx+2] = xNodes_outlet
    if(VERBOSE): print("prestream_lattice")
    #print_selected_columns(xNodes_outlet, 1, [0,1,2])   
    if(VERBOSE): print_min_max(xNodes_outlet)

    
    #stream lattice
    #stream lattice
    if(VERBOSE): print("prestream_lattice")
    #copy upper and lower boundaries
    if(VERBOSE): print("prestream_lattice:")
    #print_selected_columns(prestream_lattice, 1, [0,1,2])
    if(VERBOSE): print_min_max(prestream_lattice)
    streamed_lattice = streamLtc(prestream_lattice)
    if(VERBOSE): print("streamed_lattice:")
    #print_selected_columns(streamed_lattice, 1, [0,1,2])    
    if(VERBOSE): print_min_max(streamed_lattice)  


    #assign inlet and outlet boundary values -> B)
    streamed_lattice[:, :, 0] = xNodes_inlet    
    
    #update boundary conditions
    _ltc_bc = streamed_lattice.copy()
    streamed_lattice = bounceBackTopBottom(streamed_lattice, _ltc_bc, _index_mapping_top, _index_mapping_bottom)
    del _ltc_bc
    gc.collect()
    
    #update moments at N
    roh,current_density,u_avg = updateMoments(streamed_lattice)
    _roh_N = roh[:,Nx+1]                  
    _roh_1 = roh[:,1]
    midstream_vel = u_avg[:,int(Ny/2),Nx+1]
    outlet_midstream_avg_v.append([t, midstream_vel[0]])

    #calulcate collision
    if(VERBOSE): print("calculate collision")
    if(VERBOSE): print("streamed_lattice:")
    #print_selected_columns(streamed_lattice, 1, [0,1,2])
    if(VERBOSE): print_min_max(streamed_lattice) 
    feq = f_eq_slc(streamed_lattice, roh, u_avg)
    if(VERBOSE): print("feq:")
    #print_selected_columns(feq, 1, [0,1,2])
    if(VERBOSE): print_min_max(feq) 
    collision_lattice = (1-omega_nd) * streamed_lattice + omega_nd * feq
    if(VERBOSE): print("collision_lattice:")
    #print_selected_columns(collision_lattice, 1, [0,1,2])
    if(VERBOSE): print_min_max(collision_lattice) 

    #only use collision_lattice and copy for documentation 
    prestream_lattice = collision_lattice.copy()
    del collision_lattice
    gc.collect()

    if(VERBOSE): print("prestream_lattice prior to next loop:")
    #print_selected_columns(prestream_lattice, 4, [1, Nx // 2, Nx])
    if(VERBOSE): print_min_max(prestream_lattice) 
    

    t += dt
    index += 1
    print("Simulation Execution-> TO TAL_TIME:{0}; t:{1}; %:{2}".format(TOTAL_TIME, t, (t/TOTAL_TIME)*100.0))


time_values = [pair[0] for pair in outlet_midstream_avg_v]
speed_values = [pair[1] for pair in outlet_midstream_avg_v]

plt.plot(time_values, speed_values)
plt.xlabel('Time (t)')
plt.ylabel('Speed (s)')
plt.title('Speed vs. Time')
plt.grid(True)
plt.show(block=True)