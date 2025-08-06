#this simulation a microchannel flow is modelled along a plane inserted vertically and symetrically in the channel center
#aligned to the z-axis vertically and in the x-axis direction along hte channel length
#the axes in the plane are y-axis in the vertical direction and x-axis in the channel horizontal direction
#The simulation is based on the paper by Inamuro et al, 2003, Journal of Computational Physics 198 
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
import time
import numpy as np
import math


TOTAL_ITERATION = 10000
VERBOSE1=True
VERBOSE2=False
VERBOSE_MAX_RHO=False

GC_COLLECT = True

PRESSURE_IN_DENSITY_MAP = True

Cs=np.sqrt(1/3)
D=1e-3 #m
L=1 #m

D_nd=50 #100

Yn=D_nd #+1
Xn=200 #int(Yn*L/D)

if(VERBOSE1): print("Ny: {0}".format(Yn))
if(VERBOSE1): print("Nx: {0}".format(Xn))

dx=D/D_nd #old->5*10**(-5)
dy = dx
if(VERBOSE1): print("dx: {0}".format(dx))
#relaxation time tau, should be > 0,5
tau=0.6

dP=1e-2 #Pa
rho_0=1e3 #kg/m^3
p_in=1/3
p_out=p_in-dP
rho_in=p_in/Cs**2
rho_out=p_out/Cs**2
dRho=dP/Cs**2


if(VERBOSE1): 
    print("p_in: {0}".format(p_in))
    print("p_out: {0}".format(p_out))
    print("rho_in: {0}".format(rho_in))
    print("rho_out: {0}".format(rho_out))

nu=2.9e-6 #m^2/s => in OLB this is 1/Re, with Re=148. So Re must become Re= in order to conform with this simulation
dt=Cs**2*(tau-0.5)*(dx**2/nu)

nu_ = Cs**2*(tau-0.5)*(dx**2) * dx
dt_=Cs**2*(tau-0.5)*(dx**2/nu_)

Cs_ = np.sqrt(1/3*(dx**2/dt**2))
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

#we need Cl, Crho, Ct

# 1. Conversion factor Cl for length
Cl=dx #freely chosen
dx_nd=dx/Cl
if(VERBOSE1): 
    print("Cl: {0}".format(Cl))
    print("dx_nd: {0}".format(dx_nd))

#2. Conversion factor Crho for density
Crho=rho_0
rho_nd = rho_0/Crho
if(VERBOSE1): 
    print("Crho: {0}".format(Crho))
    print("rho_nd: {0}".format(rho_nd))

#3. Conversion factor Ct for time
Ct=dt
dt_nd = dt/Ct
if(VERBOSE1): 
    print("Ct: {0}".format(Ct))
    print("dt_nd: {0}".format(dt_nd))

#4. Conversion factor Cu for velocity
Cu=Cl/Ct
U_nd = U/Cu #-> limit U_nd=0.1
U_nd=0.1

if(VERBOSE1): 
    print("Cu: {0}".format(Cu))
    print("U_nd: {0}".format(U_nd))

#5. Conversion factor CF for Force
CF=Crho*Cl**4*Ct**(-2)
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

#Inamuro eq(8): constant E
E = np.array([0,    # i=0
            2/9,    # i=1
            1/9,    # i=2
            1/9,    # i=3
            1/9,    # i=4
            1/9,    # i=5
            1/9,    # i=6
            1/9,    # i=7
            1/72])  # i=8
#Inamuro eq(8): constant H
H = np.array([1,    # i=0
            0,      # i=1
            0,      # i=2
            0,      # i=3
            0,      # i=4
            0,      # i=5
            0,      # i=6
            0,      # i=7
            0])     # i=8
#Inamuro eq(8): constant F
F = 3*E
F[0] = -7/3
c_tensor = np.zeros((9, 2, 2))
for i in range(9):
    c_tensor = np.outer(discrete_velocities[i], discrete_velocities[i])


if(VERBOSE1): print("Discrete velocities: {0}".format(discrete_velocities))

#Krüger: force weights
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
if(VERBOSE1): print("Weights: {0}".format(weights))

#Inamuro eq(4): particle velocity distribution
def phi(_f):
    _f = np.sum(_f, axis=0)
    
    return _f

#Inamuro eq(11): bulk free-energy density
def psi(_phi):
    _psi = _phi*T*math.ln(_phi/(1-b*_phi)) - a*_phi**2

    return _psi

#Inamuro eq(10): p0 from eq(6)
def p0(_phi):
    _p0 = (_phi * T) / (1 - b * _phi) - a * _phi**2

    return _p0

#Inamuro eq(24): pressure
def gradient_p(p):
    _gradient_p = np.stack(np.gradient(p, axis=(1,0)), axis=0)

    return _gradient_p

#Inamuro eq(12): first derivatives - partial dphi/dx_a, du_b/dx_a, drho/dx_a
def c_first_derivative(lamda, dx=1.0, dy=1.0):

    is_3d = len(lamda.shape) == 3 and lamda.shape[0] == 2
    is_2d = len(lamda.shape) == 2

    if not (is_2d or is_3d):
        raise ValueError(f"Expected lamda shape (202,52) or (2,202,52), got {lamda.shape}")

    dlamda_dx = np.zeros_like(lamda)
    dlamda_dy = np.zeros_like(lamda)
    roll_axes = (1,2) if is_3d else (0,1)

    for k in range(1,9):
        shift_x, shift_y = -discrete_velocities[k,0], -discrete_velocities[k,1]
        lamda_shift = np.roll(lamda, shift=(shift_x, shift_y), axis=roll_axes)
        dlamda_dx += discrete_velocities[k,0] * lamda_shift
        dlamda_dy += discrete_velocities[k,1] * lamda_shift

    dlamda_dx /= (10 * dx)
    dlamda_dy /= (10 * dy)

    return dlamda_dx, dlamda_dy

#Inamuro eq(13): first derivatives partial - partial dphi²/dx_a²
def c_second_derivative(lamda, dx=1.0, dy=1.0):
    sum_xx = np.zeros((Xn+2, Yn+2))
    sum_yy = np.zeros((Xn+2, Yn+2))

    for k in range(1,9):
        shift_x, shift_y = -discrete_velocities[k,0], -discrete_velocities[k,1]        
        lamda_shift = np.roll(lamda, shift=(shift_x, shift_y), axis=(0,1))
        sum_xx = discrete_velocities[k, 0] * lamda_shift
        sum_yy = discrete_velocities[k, 0] * lamda_shift

    laplacian = (sum_xx + sum_yy - 14 * lamda)/(5*dx)

    return laplacian  

#Inamuro eq(9): Gab - shear terms
def Gab(func):
    phi_x, phi_y = c_first_derivative(_phi)
    grad_func = phi_x**2 + phi_y**2

    G = np.zeros((Xn+2, Yn+2, 2, 2))
    G[:,:,0,0] = (9/2) * phi_x * phi_x - (3/2) * grad_func
    G[:,:,0,1] = (3/2) * phi_x * phi_y
    G[:,:,1,0] = (3/2) * phi_x * phi_y
    G[:,:,1,1] = (9/2) * phi_y * phi_y - (3/2) * grad_func

    return G

#Inamuro eq(14 & 15): density rho and viscosity mu
def density_and_viscosity(rho_G, rho_L, _phi, phi_cutoff_G, phi_cutoff_L, mu_L, mu_G):
    _rho = 0.0
    delta_rho = rho_L - rho_G
    delta_phi_cutoff = phi_cutoff_L - phi_cutoff_G
    dash_phi_cutoff = (phi_cutoff_L + phi_cutoff_G) / 2
    rho_center = (delta_rho/2) * (np.sin(((_phi - dash_phi_cutoff)/delta_phi_cutoff) * np.pi) + 1) + rho_G

    #if _phi < phi_cutoff_G:
    #    _rho = rho_G
    #elif _phi >= phi_cutoff_G and _phi <= phi_cutoff_L:
    #    _rho = rho_center
    #elif _phi > phi_cutoff_L:
    #    _rho = rho_L
    _rho = np.where(_phi < phi_cutoff_G, rho_G, np.where(_phi <= phi_cutoff_L, rho_center, rho_L))

    mu = ((_rho - rho_G) / (rho_L - rho_G)) * (mu_L - mu_G) + mu_G

    return _rho, mu

#Inamuro eq(23): relaxation time tau_h
def tau_h(_rho):
    _tau_h = 1/_rho + 1/2
    return _tau_h

#Inamuro eq(22,24): evolution equation of the velocity distribution function h(i) and pressure
def ph(hn, _rho, u_ckl, dx=1.0, dy=1.0):
    _p = np.zeros(hn.shape[1:])
    shift_x = discrete_velocities * dx
    x_shifts = shift_x[:,0]
    y_shifts = shift_x[:,1]
    E_exp = np.expand_dims(E, axis=(1,2)) #(9,1,1)
    p_exp = np.expand_dims(_p, axis=0) #(1,Xn,Yn)
    E_p = E_exp * p_exp
    #term1 = hn      

    tau = 1/_rho + 1/2 #(Xn,Yn)
    tau_exp = np.expand_dims(1/tau, axis=0)
    #term2 = tau_exp * (hn - E_p)    

    du_dx, du_dy = c_first_derivative(u_ckl[0])
    #term3 = (1/3) * E_exp * der_u_exp

    #full evolution equation
    collision = hn - tau_exp * (hn - E_p) - (1/3) * E_exp * du_dx
    hn_plus1 = np.zeros_like(hn)
    for k in range(1,9):
        hn_plus1[k] = np.roll(collision[k],shift=(int(x_shifts[k]),int(y_shifts[k])), axis=(0,1))
    _p = np.sum(hn_plus1[1:], axis=0)        

    return _p, hn_plus1

#Inamuro eq(7): calculation of predicted velocity of the two phase fluid - collision term
def gi_c(u, rho, tau_g, Kg, F):
    grad_u = np.stack([c_first_derivative(u[:,:,0]), c_first_derivative(u[:,:,1])], axis=0)
    grad_rho = c_first_derivative(rho)
    _gi_c = np.zeros(9, Xn+2, Yn+2)

    #_gi_c
    for i in range(9):
        #cia*ua
        c_dot_u = np.tensordot(discrete_velocities[i], u, axes=(0,0))

        #ua*ua
        u_dot_u = np.sum(u*u, axis=0)

        #cia*cib*ua*ub
        c_dot_u_tensor = c_dot_u ** 2 #(ci.u)^2, shape(Xn, Yn)

        #(dub/dxa + dua/dxb)
        grad_sym = grad_u + np.transpose(grad_u, (1,0,2,3)) #symmetric gradient
        grad_term = np.einsum('ijab,ab->ij', grad_sym, c_tensor[i])

        #Gab(rho)*cia*cib
        Gab_rho = Gab(rho)
        _Gab = np.einsum('ijab,ab->ij', Gab_rho, c_tensor[i])

        #(drho/dxa)^2
        drho_dxa = np.sum(grad_rho**2, axis=0) #shape(Xn, Yn)

        _gi_c[i] = E[i] * (1 + 3*c_dot_u - 1.5*u_dot_u + 4.5*c_dot_u_tensor + 1.5*(tau_g - 0.5)*dx*grad_term) \
            + E[i]*Kg/rho*_Gab - (2/3)*F[i]*Kg/rho*drho_dxa
        
    return _gi_c

#Inamuro eq(6): calculation of the order parameter which distiguishes the two phases - collision term
def fi_c(u, Kf, F, _phi):
    _fi_c = np.zeros((9, Xn+2, Yn+2))
    d1phi_dxa = c_first_derivative(_phi, 0)
    d2phi_dxa = c_second_derivative(_phi)

    #gi
    for i in range(9):
        #cia*ua
        c_dot_u = np.tensordot(discrete_velocities[i], u, axes=(0,0))

        #Gab(rho)*cia*cib
        Gab_phi = Gab(_phi)
        _Gab = np.einsum('ijab,ab->ij', Gab_phi, c_tensor[i])

        _fi_c[i] = H[i]*_phi + F[i]*(p0(_phi) - Kf*_phi*d2phi_dxa - Kf/6*(d1phi_dxa)**2) \
            + 3*E[i]*_phi*c_dot_u \
            + E[i]*Kf*_Gab
        
    return _fi_c


def force(rho, g_x, g_y, cs):
    _force = 3 * E[:,None,None] * rho[None,:,:] * (discrete_velocities[:,0]*g_x + discrete_velocities[:,1]*g_y)[:,None,None] / cs**2
    return _force


#apply bounce-back conditions on upper and lower boundaries of pipe
def bounceBackTopBottom1(f, nx, ny):
    '''Performs the bounce back step
    
    Arguements
    -----------
    f: np.array (nx, ny, 9)
        probability density function
    nx: int
        number of grid points in x direction
    ny: int
        number of grid points in y direction
    
    Returns
    ---------
    f: np.array (nx, ny, 9)
        probability density function after the bounce back step
    '''
     
    # rigid lower wall 
    f[2, 1 : nx + 1, 1] = f[4, 1 : nx + 1, 0]
    f[5, 1 : nx + 1, 1] = np.roll(f[7, 1 : nx + 1, 0], 1)
    f[6, 1 : nx + 1, 1] = np.roll(f[8, 1 : nx + 1, 0], -1)
    
    # rigid upper wall
    f[4, 1 : nx + 1, ny] = f[2, 1 : nx + 1, ny + 1]
    f[7, 1 : nx + 1, ny] = np.roll(f[5, 1 : nx + 1, ny + 1], -1)
    f[8, 1 : nx + 1, ny] = np.roll(f[6, 1 : nx + 1, ny + 1], 1)

    return f    


#calulate boundary nodes X(0) and X(N+1) for periodic BC with presssure difference
def calcPeriodicBC00(pdf, _rho_N, _u_cNl, _rho_in, _rho_1, _u_c1l, _rho_out, rho, u_ckl, iteration): 
    
   # Reshape inputs for f_eq_3D compatibility
    _rho_in_2d = _rho_in[None, :]  # (ny,) -> (1, ny)
    _u_cNl_2d = _u_cNl[:, None, :]  # (2, ny) -> (2, 1, ny)
    f_eq_in = f_eq_3D(_rho_in_2d, _u_cNl_2d)[:, 0, :]  # (9, 1, ny) -> (9, ny)
    fi_xN_prestream = pdf[:,Xn,:]
    _rho_N_2d = _rho_N[None, :]  # (ny,) -> (1, ny)
    fi_eq_N = f_eq_3D(_rho_N_2d, _u_cNl_2d)[:, 0, :]  # (9, 1, ny) -> (9, ny)
    fi_x0 = f_eq_in + (fi_xN_prestream - fi_eq_N)  # (9, ny)    

    # Reshape inputs to add a singleton x-dimension for f_eq_3D
    _rho_out_2d = _rho_out[None, :]  # (ny,) -> (1, ny)
    _u_c1l_2d = _u_c1l[:, None, :]   # (2, ny) -> (2, 1, ny)
    f_eq_out = f_eq_3D(_rho_out_2d, _u_c1l_2d)[:, 0, :]  # (9, 1, ny) -> (9, ny)
    fi_x0_prestream = pdf[:,1,:]  # (9, ny)
    _rho_1_2d = _rho_1[None, :]  # (ny,) -> (1, ny)
    fi_eq_1 = f_eq_3D(_rho_1_2d, _u_c1l_2d)[:, 0, :]  # (9, 1, ny) -> (9, ny)
    fi_xNplus1 = f_eq_out + (fi_x0_prestream - fi_eq_1)  # (9, ny)
    
    return fi_x0, fi_xNplus1  



def get_velocity_y_values4Poiseuille2D(num_nodes1, D):
    R = D / 2.0
    
    return np.linspace(-R, R, num_nodes1)


#roll the lattice based on the discrete velocities
def streamLattice0(_ltc):
    global discrete_velocities, weights
    
    shifted_lattice = np.stack([
        np.roll(np.roll(_ltc[d, :, :], shift=dx, axis=0), shift=dy, axis=1)
        for d, (dx, dy) in enumerate(discrete_velocities)
    ], axis=0)

    return shifted_lattice


def bounceBackTopBottom2(f, nx, ny):
    '''Performs the bounce back step
    
    Arguements
    -----------
    f: np.array (nx, ny, 9)
        probability density function
    nx: int
        number of grid points in x direction
    ny: int
        number of grid points in y direction
    
    Returns
    ---------
    f: np.array (nx, ny, 9)
        probability density function after the bounce back step
    '''
     
    # rigid lower wall 
    f[2, 1 : nx + 1, 1] = f[4, 1 : nx + 1, 0]
    f[5, 1 : nx + 1, 1] = np.roll(f[7, 1 : nx + 1, 0], 1)
    f[6, 1 : nx + 1, 1] = np.roll(f[8, 1 : nx + 1, 0], -1)
    
    # rigid upper wall
    f[4, 1 : nx + 1, ny] = f[2, 1 : nx + 1, ny + 1]
    f[7, 1 : nx + 1, ny] = np.roll(f[5, 1 : nx + 1, ny + 1], -1)
    f[8, 1 : nx + 1, ny] = np.roll(f[6, 1 : nx + 1, ny + 1], 1)

    return f    


def amplitude_plot(ax1, u_full_range, listIterations, axis, xlabel, ylabel, title, nx, ny, poiseuille_velocities=None):
    for iteration, combined_u_ckl in u_full_range.items():
        u = combined_u_ckl[nx, 1:ny + 1]
        ax1.plot(axis, u, label="t=" + str(iteration))
    
    if poiseuille_velocities is not None:
        ax1.plot(axis, poiseuille_velocities, label="Analytical", color="red", linestyle="--")
    
    ax1.legend(ncol=len(u_full_range) // 4, loc='upper right')
    ax1.grid()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    
    ax1.set_xlim(-1, 51)
    ax1.set_xticks(np.arange(0, 51, 10))
    
    ax1.margins(x=0, y=0)
    y_min, y_max = ax1.get_ylim()
    y_max_new = y_max + 0.1 * (y_max - y_min)
    ax1.set_ylim(y_min, y_max_new)


# Density profile
def density_profile(ax1, den_eq, nx, ny):
    ax1.plot(den_eq[1:nx, ny // 2])
    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("Density [rho]")
    ax1.margins(x=0, y=0)
    ax1.set_title("Longitudinal density profile")
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_xlim(0, nx)
    ax1.set_xticks(np.linspace(0, nx, 5))
    ax1.grid()


#2D Density map
def density_map(ax, rho_full_range, rho_out, rho_in, title):
    print(f"Debug density_map: min={np.min(rho_full_range):.6f}, max={np.max(rho_full_range):.6f}, rho_out={rho_out:.6f}, rho_in={rho_in:.6f}")
    im = ax.imshow(rho_full_range.T, interpolation='nearest', origin='lower', cmap='viridis', vmin=rho_out, vmax=rho_in)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(title)
    ax.margins(x=0, y=0)
    ax.set_aspect('auto')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([rho_out, (rho_out + rho_in) / 2, rho_in])


#2D Velocity map
def velocity_map(ax, u_magnitude, _iteration):
    im = ax.imshow(u_magnitude.T, interpolation='nearest', origin='lower', cmap='plasma')  # Capture imshow return
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Velocity [u$_x$] map")
    ax.margins(x=0, y=0)
    ax.set_aspect('auto')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 


def filter_u_ckl_fullrange(velocities_dict, iterationsOfInterest):
    filtered_velocities = {iteration: velocities_dict[iteration] for iteration in iterationsOfInterest if iteration in velocities_dict}

    return filtered_velocities   



#preliminary
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)

#initialise
#average velocity, cartesion x,y-directions, k is y-position, l is x-position
rho_in_k = np.full((Yn+2), rho_in, dtype=np.float32)
rho_out_k = np.full((Yn+2), rho_out, dtype=np.float32)
u_ckl = np.zeros((2, Xn+2, Yn+2), dtype=np.float32)
INIT_RHO = 1 #0.001
rho = np.full((Xn+2, Yn+2), INIT_RHO, dtype=np.float32)
#pdf = f_eq_3D(rho, u_ckl)

# Simulation parameters
R = D / 2  # Radius of the pipe

iteration = 0
iterations = []

list_avg_velocities = {}

TOTAL_ITERATION = 12001
start = time.perf_counter()
w = 1
epsilon_cutoff = 10e-5
tau_f = 1
tau_g = 1
a=1
b=6.7
T=3.5e-2
Kf = 0.5 * dx**2
Kg = 1e-5 * dx**2
C = 0
Sh = U/Cs

rho_G = 1
rho_L = 50
_phi = np.ones((Xn+2, Yn+2))
phi_cutoff_G = 1.5e-2
phi_cutoff_L = 9.2e-2
mu_G = 1.6e-4*dx
mu_L = 8e-3*dx
h0 = np.ones((9,Xn+2, Yn+2))
p0 = np.zeros((9,Xn+2, Yn+2))
#_h = np.zeros((9,Xn,Yn))
_fi_c = np.zeros((9,Xn+2, Yn+2))
fi = np.zeros((9,Xn+2, Yn+2))
_gi_c = np.zeros((9, Xn+2, Yn+2))
gi = np.zeros((9, Xn+2, Yn+2))
h = np.zeros((9,Xn+2, Yn+2))
alpha = 20
g = 9.81
g_x = g * np.sin(np.radians(alpha))
g_y = -g * np.cos(np.radians(alpha))

#initial conditions
y0 = (Yn-1)/2
xi = 0.75
x,y = np.meshgrid(np.arange(Xn+2),np.arange(Yn+2),indexing='ij')
_phi = (phi_cutoff_L + phi_cutoff_G)/2 - (phi_cutoff_L - phi_cutoff_G)/2 * np.tanh((y-y0)/xi)
rho = rho_G + (_phi - phi_cutoff_G) / (phi_cutoff_L - phi_cutoff_G) * (rho_L - rho_G)


while iteration < TOTAL_ITERATION:
    #Inamuro §2.3 Algorithm of computation:
    #Step 1. Using eqs (1) and (2), compute (fi(x, t+dt) and g(x, t+dt), and then compute phi(x, t+dt) and _u(x, t+dt)= with eqs (4) and (5).
    #Also rho(x, t+dt) is calculated with eq (4)
    if iteration > 0:
        _fi_c = fi_c(u_ckl, Kf, F, _phi)
    
    #Inamuro eq(2): calculation of the order parameter which distiguishes the two phases
    #calculate f(i)
    fi = fi - (1/tau_f) * (fi - _fi_c)
    #Calculation of order parameter to distiguish the 2 phases
    _phi = phi(fi)
    #Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    if iteration > 0:
        _gi_c = gi_c(u_ckl, rho, tau_g, Kg, F)
    rho, mu = density_and_viscosity(rho_G, rho_L, _phi, phi_cutoff_G, phi_cutoff_L, mu_L, mu_G)

    #Inamuro eq(3): calculation of the predicted velocity of the two phase fluid
    E_exp = np.expand_dims(E, axis=(1,2)) #(9,1,1)
    u_dx, u_dy = c_first_derivative(u_ckl)[0] 
    v_dx, v_dy = c_first_derivative(u_ckl)[1] 
    div_u = u_dx + v_dy
    mu_div_u = mu * div_u
    d_dx, d_dy = c_first_derivative(mu)
    discrete_velocities_exp = discrete_velocities[:,:,None,None]
    derivatives = np.stack([d_dx, d_dy], axis=0)
    deriv_term = np.sum(discrete_velocities_exp*derivatives[None,:,:,:], axis=1)
    deriv_term = 3 * E_exp * deriv_term / rho[None,:,:] * dx
    force_term = force(rho, g_x, g_y, Cs)
    gi = gi - (1/tau_g) * (gi - _gi_c) + deriv_term + force_term
    
    #Inamuro eq(5): Compute u(x,t+dt)
    u_ckl = np.einsum('ia,ijk->ajk', discrete_velocities, gi)    

    #Step2a. calculate h, p
    epsilon0 = epsilon_cutoff * 10.0
    epsilon = np.full_like(h, epsilon0, dtype=np.float32)
    while np.all(epsilon > epsilon_cutoff):
        p, h = ph(h, rho, u_ckl)
        epsilon = np.abs(p-p0)/rho
        
    #Inamuro eq(22 & 24): assign resultant p to p0 for next iteration
    p0 = p

    #Step 3: Compute u(x,t+dt) using eq. (20)
    #Inamuro eq(20): corrected current velocity u which satisfies the continuity equation div.u=0
    u_ckl = (-gradient_p(p)/rho * dt + u_ckl)/Sh


    #stream lattice 
    #store pre-stream boundary top and bottom values
    fi = streamLattice0(fi)


    #Bounce-Back Top and Bottom
    fi = bounceBackTopBottom2(fi, Xn, Yn)     


    # Get the maximum density and its location
    max_density = np.max(rho)
    max_location = np.unravel_index(np.argmax(rho), rho.shape)
    if VERBOSE_MAX_RHO:
        if np.any(rho > 1):
            print(f"Instability detected at iteration {iteration + 1}")
        print(f"Maximum density: {max_density} at location {max_location}")


    #4.1b. assign inlet boundary values -> B)
    fi[:, 0, :] = fi[:,Xn,:]
    fi[:, Xn+1, :] = fi[:,1,:]  


    # Update plots and parameters
    _rho_full_range = rho
    list_avg_velocities[iteration] = u_ckl[0, 1:-1, :]

    #Step 4: re-iterate

    iteration += 1
    if VERBOSE2:
        if (iteration % 100) == 0:
            print(f"Simulation Execution -> TOTAL_ITERATION: {TOTAL_ITERATION}; iteration: {iteration}; {((iteration/TOTAL_ITERATION)*100.0):.1f} %")


end = time.perf_counter()
iterationsOfInterest = [0, 10, 50, 100, 200, 500, 1000, 5000, 10000, 12000]
diff = end - start
rho_in, rho_out, cs_2 = _rho_full_range[1, Yn // 2], _rho_full_range[Xn, Yn // 2], 1 / 3
dp = (rho_in - rho_out) * cs_2
rho_min = np.min(_rho_full_range)
rho_max = np.max(_rho_full_range)
print(f"Debug: _rho_full_range min = {rho_min:.6f}, max = {rho_max:.6f}")
paneLabel = f"LB: 2D Poiseuille with pressure difference; Lattice [{Xn},{Yn}]; Single processor"

filtered_u_ckl_dict = filter_u_ckl_fullrange(list_avg_velocities, iterationsOfInterest)
filtered_u_ckl_list = list(filtered_u_ckl_dict.values())


# height ratios based on lattice dimensions
aspect_ratio = Xn / Yn
top_row_height = 1 #1.5
bottom_row_height = 1 #max(top_row_height * (Yn / Xn) * 2, 0.5)
height_ratios = [top_row_height, bottom_row_height]

# 2x2 multi-plot grid with swapped column widths
fig, ax = plt.subplots(
    2, 2,
    figsize=(15, 10),
    gridspec_kw={
        'width_ratios': [2, 4],  # Swapped from [4, 2] to [2, 4]
        'height_ratios': height_ratios,
        'left': 0.15, 'right': 0.85, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

# Row 0, Col 0: Amplitude plot (now in narrower column, width 2)
sectionPosition = Xn-1
U_max = np.max(filtered_u_ckl_list[-1][sectionPosition, 1:Yn+1])
amplitude_plot(ax[0, 0], filtered_u_ckl_dict, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_x$", f"Amplitude u$_x$ at x={Xn}", sectionPosition, Yn, None)

# Row 0, Col 1: Velocity map (now in wider column, width 4)
_iteration = -1
velocity_map(ax[0, 1], filtered_u_ckl_list[-1][1:-1, 1:Yn+1], _iteration)


# Row 1, Col 0: Density profile (now in narrower column, width 2)
density_profile(ax[1, 0], _rho_full_range, Xn, Yn)

# Row 1, Col 1: Density map (now in wider column, width 4)
if PRESSURE_IN_DENSITY_MAP:
    min_value = 0 #np.min(_rho_full_range)
    _pressure_full_range = (_rho_full_range - min_value) * Cs**2  
    _pressure_out = (rho_out - min_value) * Cs**2
    _pressure_in = (rho_in - min_value) * Cs**2
    title = "Pressure map"
    density_map(ax[1, 1], _pressure_full_range, _pressure_out, _pressure_in, title)
else:
    title = "Density [rho] map"
    density_map(ax[1, 1], _rho_full_range, rho_out, rho_in, title)

# Text at top
text = f"Run-time: {diff:.1f} s; rho_in: {rho_in:.6f}; rho_out: {rho_out:.6f}; dp: {dp:.6f}"
fig.text(0.5, 0.98, text, ha='center', va='top', fontsize=12)
fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.show()