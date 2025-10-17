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

E = np.array([2/9,  # i=0
            1/9,    # i=1
            1/9,    # i=2
            1/9,    # i=3
            1/9,    # i=4
            1/9,    # i=5
            1/9,    # i=6
            1/9,    # i=7
            1/72])  # i=8

H = np.array([1,    # i=0
            0,      # i=1
            0,      # i=2
            0,      # i=3
            0,      # i=4
            0,      # i=5
            0,      # i=6
            0,      # i=7
            0])     # i=8

F = 3*E
F[0] = -7/3
c_tensor = np.zeros((9, 2, 2))
for i in range(9):
    c_tensor = np.outer(discrete_velocities[i], discrete_velocities[i])


if(VERBOSE1): print("Discrete velocities: {0}".format(discrete_velocities))

#weights
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
if(VERBOSE1): print("Weights: {0}".format(weights))

#equilibrium distribution function feq(0->8)
def f_eq_3D(_rho, _u_ckl):
    global discrete_velocities, weights

    _u_ckl_dot = np.einsum('hk,kij->hij', discrete_velocities, _u_ckl) #(9,2) * (2,101,101002)
    _u_ckl_product = np.einsum('kij,kij->ij', _u_ckl, _u_ckl)
    _u_ckl_product_reshaped = _u_ckl_product.reshape(1, *(_u_ckl_product.shape))
    _ones = np.ones(_u_ckl_dot.shape)

    _rho_reshaped = _rho.reshape(1, *(_rho.shape))
    weights_reshaped = weights.reshape((9, 1, 1)) * np.ones(_rho.shape)
    factors = weights_reshaped * _rho_reshaped

    #Part_2_BTE_to_LBM, BTE3
    #feq = factors * (
    #    _ones + 3. * _u_ckl_dot / Cs**2 + ^(9. / 2.) * _u_ckl_dot**2 / Cs**4 - (3. / 2.) * _u_ckl_product_reshaped / Cs**2
    #)^

    #BGK formula, Kruger pp67
    f_eq = factors * (
        #_ones + _u_ckl_dot / Cs**2 + (1/2.) * _u_ckl_dot**2 / Cs**4 - (1/2.) * _u_ckl_product_reshaped / Cs**2
        _ones + _u_ckl_dot / Cs**2 + (1/2.) * _u_ckl_dot**2 / Cs**4 - (1/2.) * _u_ckl_product / Cs**2
    )
    
    return f_eq


#roll the lattice based on the discrete velocities
def streamLattice0(_ltc):
    global discrete_velocities, weights
    
    shifted_lattice = np.stack([
        np.roll(np.roll(_ltc[d, :, :], shift=dx, axis=0), shift=dy, axis=1)
        for d, (dx, dy) in enumerate(discrete_velocities)
    ], axis=0)

    return shifted_lattice


def updateMoments(_ltc):
    global discrete_velocities, weights
    rho = np.sum(_ltc, axis=0)
    u = np.einsum('ki,ijl->kjl', discrete_velocities.T, _ltc) / np.sum(_ltc, axis=0) 
    
    return rho, u   


def phi(_pdf):
    _pdf = np.sum(_pdf, axis=0)
    
    return _pdf


def psi(_phi):
    _psi = _phi*T*math.ln(_phi/(1-b*_phi)) - a*_phi**2

    return _psi


def p0(_phi):
    _p0 = (_phi*T)/(1-b*_phi)-a*_phi**2

    return _p0


def c_gradient(field):
    grad = np.zeros((Xn, Yn, 2))
    grad[1:-1,1:-1,0] = (field[2:,1:-1] - field[:-2,1:-1]) / (2 * dx)
    grad[1:-1,1:-1,1] = (field[1:-1,2:] - field[1:-1,:-2]) / (2 * dy)
    #boundary
    grad[0,:,:] = grad[1,:,:]
    grad[-1,:,:] = grad[-2,:,:]
    grad[:,0,:] = grad[:,1,:]
    grad[:,-1,:] = grad[:,-2,:]
    return grad


def Gab(func):
    grad_func = np.stack(np.gradient(func, dx, axis=(0,1))[::-1], axis=0)
    first_term = (9/2)*np.einsum('axy,bxy->abxy', grad_func, grad_func)
    grad_phi_sq = np.sum(grad_func ** 2, axis=0)
    delta_ab = np.eye(2)
    second_term = (9/4)*np.einsum('xy,ab->abxy',grad_phi_sq,delta_ab)

    _Gab = first_term - second_term
    return _Gab


def gi_c(Xn, Yn, u, rho, tau_g, Kg, F):
    grad_u = np.stack([c_gradient(u[:,:,0]), c_gradient(u[:,:,1])], axis=0)
    grad_rho = c_gradient(rho)
    gi = np.zeros(9, Xn, Yn)

    #gi
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

        gi[i] = E[i] * (1 + 3*c_dot_u - 1.5*u_dot_u + 4.5*c_dot_u_tensor + 1.5*(tau_g - 0.5)*dx*grad_term) \
            + E[i]*Kg/rho*_Gab - (2/3)*F[i]*Kg/rho*drho_dxa
        
    return gi


def fi_c(Xn, Yn, u, _pdf, Kf, F):
    fi = np.zeros(9, Xn, Yn)
    _phi = phi(_pdf)
    grad_phi = c_gradient(_phi)    

    #gi
    for i in range(9):
        #cia*ua
        c_dot_u = np.tensordot(discrete_velocities[i], u, axes=(0,0))

        #Gab(rho)*cia*cib
        Gab_phi = Gab(_phi)
        _Gab = np.einsum('ijab,ab->ij', Gab_phi, c_tensor[i])

        #(dphi/dxa)^2
        dphi_dxa = np.sum(grad_phi**2, axis=0) #shape(Xn, Yn)

        fi[i] = H[i]*_phi + F[i] * (p0(_pdf) - Kf*_phi*dphi_dxa -Kf/6*(grad_phi)**2) \
            + 3*E[i]*_phi*c_dot_u \
            + E[i]*Kf*_Gab
        
    return fi


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


#2D Poiseuille inlet velocity u(y) for comparison with numerical result
def Poiseuille2DUy2(y, U, D):
    """
    Returns the velocity profile for Poiseuille flow in a pipe.
    :param y: Array of radial distances (y values).
    :param U: Maximum velocity at the center of the pipe.
    :param D: Diameter of the pipe.
    :return: Velocity profile at each y position.
    """
    R = D / 2  # Pipe radius
    u_poiseuille = U * (1 - (y / R) ** 2)
    return u_poiseuille  


def get_velocity_y_values4Poiseuille2D(num_nodes1, D):
    R = D / 2.0
    
    return np.linspace(-R, R, num_nodes1)


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
pdf = f_eq_3D(rho, u_ckl)

# Simulation parameters
R = D / 2  # Radius of the pipe

iteration = 0
iterations = []

list_avg_velocities = {}

TOTAL_ITERATION = 12001
start = time.perf_counter()
w = 1

tau_f = 1
tau_g = 1
a=1
b=1
T=2.93e-1
Kf = 0.05 * dx**2
Kg = 1e-5 * dx**2

while iteration < TOTAL_ITERATION:
    #1. moment update
    if iteration > 0:
        rho, u_ckl = updateMoments(pdf)


    # Get the maximum density and its location
    max_density = np.max(rho)
    max_location = np.unravel_index(np.argmax(rho), rho.shape)
    if VERBOSE_MAX_RHO:
        if np.any(rho > 1):
            print(f"Instability detected at iteration {iteration + 1}")
        print(f"Maximum density: {max_density} at location {max_location}")


    #2a Calculation of order parameter to distiguish the 2 phases
    _f = fi_c(Xn, Yn, u_ckl, pdf, Kf, F)
    pdf = pdf - (1/tau_f) * (pdf - _f)

    #2b Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    _g = gi_c(Xn, Yn, u_ckl, rho, tau_g, Kg, F)
    u_ckl = np.einsum('i,ijk->jk', discrete_velocities, _g)




    #2. compute equilibrium
    f_eq = f_eq_3D(rho, u_ckl)


    #3. collision term
    pdf = pdf * (1 - w) + w * f_eq 
        
    
    #4.1a Periodic Boundary conditions inlet/outlet with pressure difference
    #update extra node layers 0 and N+1 -> A) & B) acc. to Script: Boundary Conditions for the Lattice Boltzmann Method
    u_c1l = u_ckl[:, 1, :]
    u_cNl = u_ckl[:, Xn, :]    
    #u_ckl profiles at outlet and inlet
    _rho_k1 = rho[1, :]  
    _rho_kN = rho[Xn, :]    
    #assign inlet and outlet boundary values -> A)    
    fi_x0, fi_xNplus1 = calcPeriodicBC00(pdf, _rho_kN, u_cNl, rho_in_k, _rho_k1, u_c1l, rho_out_k, rho, u_ckl, iteration)


    #5. stream lattice 
    pdf = streamLattice0(pdf)


    #4.2 Bounce-Back Top and Bottom
    pdf = bounceBackTopBottom1(pdf, Xn, Yn)
    

    #4.1b. assign inlet boundary values -> B)
    pdf[:, 0, :] = fi_x0
    pdf[:, Xn+1, :] = fi_xNplus1


    # Update plots and parameters
    _rho_full_range = rho
    list_avg_velocities[iteration] = u_ckl[0, 1:-1, :]


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
poiseuille_velocities = np.array([Poiseuille2DUy2(y, U_max, D) for y in np.linspace(-R, R, Yn)])
amplitude_plot(ax[0, 0], filtered_u_ckl_dict, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_x$", f"Amplitude u$_x$ at x={Xn}", sectionPosition, Yn, poiseuille_velocities)

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