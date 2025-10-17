#this simulation a microchannel flow is modelled along a plane inserted vertically and symetrically in the channel center#aligned to the z-axis vertically and in the x-axis direction along hte channel length
#the axes in the plane are y-axis in the vertical direction and x-axis in the channel horizontal direction
#The simulation is based on the paper by Inamuro et al, 2003, Journal of Computational Physics 198
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
import math
import os
import time

import matplotlib.cm as cm  # For color mapping in plots
import matplotlib.pyplot as plt
import numpy as np
# Optional, but plt.Normalize works too
from matplotlib.colors import \
    ListedColormap  # Optional: for custom colors if needed
from matplotlib.colors import Normalize
from scipy.special import erf

#flags
VERBOSE1=True
VERBOSE2=True
VERBOSE_MAX_RHO=False
RAISE_LESS_THAN_ZERO_ERROR = False
RAISE_NaN_ERROR = False
PRESSURE_IN_DENSITY_MAP = False
PHI_NONLINEAR = True
ADD_FORCING_TERM = 1
TOTAL_ITERATION = 12001 * 10

SCRIPT_FILENAME = os.path.splitext(os.path.basename(__file__))[0]

Cs=np.sqrt(1/3)
D=1e-3 #m
L=1 #m

D_nd=50 #100

Yn=int(D_nd) #+1
Xn=200 #int(Yn*L/D)

if(VERBOSE1): print("Ny: {0}".format(Yn))
if(VERBOSE1): print("Nx: {0}".format(Xn))

dx=D/D_nd #old->5*10**(-5)
dy = dx
if(VERBOSE1): print("dx: {0}".format(dx))
#relaxation time n_tau, should be > 0,5
n_tau = 0.6

dP=0 #Pa
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
dt = Cs**2*(n_tau-0.5)*(dx**2/nu)

if(VERBOSE1): print("dt: {0}".format(dt))

#Assume U at centerline (max) velocity
U=1.0
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
Cl = dx #freely chosen
n_dx = dx/Cl #-> dx_nd=1
n_dy = n_dx
if(VERBOSE1):
    print("Cl: {0}".format(Cl))
    print("n_dx: {0}".format(n_dx))

#2. Conversion factor Crho for density
Crho = rho_0
rho_nd = rho_0/Crho #-> rho_nd=1
if(VERBOSE1):
    print("Crho: {0}".format(Crho))
    print("rho_nd: {0}".format(rho_nd))

#3. Conversion factor Ct for time
Ct=dt
n_dt = dt/Ct #-> dt_nd=1
if(VERBOSE1):
    print("Ct: {0}".format(Ct))
    print("dt_nd: {0}".format(n_dt))

#4. Conversion factor Cu for velocity
Cu=Cl/Ct
U_nd = U/Cu #-> limit U_nd=0.1
U_nd=0.1

if(VERBOSE1):
    print("Cu: {0}".format(Cu))
    print("U_nd: {0}".format(U_nd))

#5. Conversion factor CF for Force
CF=Crho*Cl/(Ct**2)
if(VERBOSE1): print("CF: {0}".format(CF))

#6. Conversion factor Cf for frequency
Cf=1/Ct
if(VERBOSE1): print("Cf: {0}".format(Cf))

#change nu_nd in order to achieve U_nd=0,1
nu_nd=((D_nd*U_nd)/(D*U))*nu
if(VERBOSE1): print("nu_nd: {0}".format(nu_nd))

tau_nd=(nu_nd/Cs**2)+1./2
if(VERBOSE1): print("tau_nd: {0}".format(tau_nd))
omega = dt/n_tau
if(VERBOSE1): print("omega: {0}".format(omega))
omega_nd = n_dt/tau_nd
if(VERBOSE1): print("omega_nd: {0}".format(omega_nd))


#discrete velocity channels for D2Q9
c = np.array([[0, 0],     # i=0
                      [1, 0],               # i=1
                      [0, 1],               # i=2
                      [-1, 0],              # i=3
                      [0, -1],              # i=4
                      [1, 1],               # i=5
                      [-1, 1],              # i=6
                      [-1, -1],             # i=7
                      [1, -1]])             # i=8

#Inamuro eq(8): constant E
E = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
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
F[0] = -5/3 #in the original Inamuro scheme with N=15 F[0]=-7/3
if(VERBOSE1): print("Discrete velocities: {0}".format(c))

#Krüger: force weights
#weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
if(VERBOSE1): print("Weights: {0}".format(E))

#Inamuro eq(4): particle velocity distribution
def phi(_f):
    __phi = np.sum(_f, axis=0)

    return __phi


#Inamuro eq(11): bulk free-energy density
def psi(_phi):
    """
    Inamuro eq(11): Compute bulk free-energy density.
    _phi: Order parameter (numpy array, shape (Xn+2, Yn+2)).
    Returns: psi (same shape as _phi).
    Raises: ValueError if _phi is out of valid range (0, 1/b).
    """
    # Check for valid phi range
    if np.any(_phi <= 0) or np.any(_phi >= 1/b):
        min_phi = np.min(_phi)
        max_phi = np.max(_phi)
        invalid_idx = np.where((_phi <= 0) | (_phi >= 1/b))
        raise ValueError(
            f"Invalid _phi in psi: min={min_phi:.3e}, max={max_phi:.3e}, "
            f"must be in (0, {1/b:.3e}). Invalid indices: {invalid_idx}"
        )
    _psi = _phi * T * np.log(_phi / (1 - b * _phi)) - a * _phi**2
    return _psi


#Inamuro eq(10): p0 from eq(6)F[i]*
def p0(_phi):
    _p0 = ((_phi * T) / (1 - b * _phi)) - a * _phi**2

    return _p0


#Inamuro eq(24): pressure
def gradient_p(p):
    """
    Compute pressure gradient using c_first_derivative with bounce-back at y=0, Yn+1.
    p: Pressure field (shape (Xn+2, Yn+2)).
    n_dx, n_dy: Grid spacings.
    Returns: Gradient (shape (2, Xn+2, Yn+2)), [dp/dx, dp/dy].
    """
    if np.any(np.isnan(p)):
        raise ValueError(f"NaN in pressure field p: min={np.min(p):.3e}, max={np.max(p):.3e}")
    
    dp_dx, dp_dy = c_first_derivative(p, n_dx, n_dy)
    
    # Apply bounce-back for dp_dy at y=0, Yn+1
    dp_dy[:, 0] = 0.0  # No y-gradient at bottom wall
    dp_dy[:, Yn+1] = 0.0  # No y-gradient at top wall
    
    return np.stack([dp_dx, dp_dy], axis=0)


#Inamuro eq(12): first derivatives - partial dphi/dx_a, du_b/dx_a, drho/dx_a
def c_first_derivative(lamda, n_dx=1.0, n_dy=1.0):
    if len(lamda.shape) != 2:
        raise ValueError(f"Expected lamda shape (nx, ny) like (202,52), got {lamda.shape}")

    scale_x = (8/14) / (10 * n_dx)
    scale_y = (8/14) / (10 * n_dy)

    neigh_c_x = c[1:9, 0]
    neigh_c_y = c[1:9, 1]

    roll_axes = (0, 1)

    shifted_neighbors = [
        np.roll(lamda, shift=(c[k+1, 0], c[k+1, 1]), axis=roll_axes) for k in range(8)
    ]
    neighbors_stack = np.stack(shifted_neighbors, axis=0)

    dlamda_dx = np.einsum('k,kij->ij', neigh_c_x, neighbors_stack) * scale_x
    dlamda_dy = np.einsum('k,kij->ij', neigh_c_y, neighbors_stack) * scale_y

    return dlamda_dx, dlamda_dy


def c_first_derivative_old(lamda, n_dx=1.0, n_dy=1.0):
    """
    Original eq(12) in Inamuro
    Compute ∂λ/∂x and ∂λ/∂y using Eq. (12): (1/(10*dx)) * sum(c_{a,i} * λ(x + c_i*dx))
    lamda: Scalar (nx, ny) or vector (2, nx, ny) field
    n_dx, n_dy: Lattice spacing in x, y directions
    Returns: dlamda_dx, dlamda_dy
    """
    is_3d = len(lamda.shape) == 3 and lamda.shape[0] == 2
    is_2d = len(lamda.shape) == 2
    if not (is_2d or is_3d):
        raise ValueError(f"Expected lamda shape (202,52) or (2,202,52), got {lamda.shape}")

    dlamda_dx = np.zeros_like(lamda)
    dlamda_dy = np.zeros_like(lamda)
    roll_axes = (1, 2) if is_3d else (0, 1)

    for k in range(1, 9):
        shift_x, shift_y = c[k, 0], c[k, 1]
        #lamda_shifted = np.roll(lamda, shift=(-shift_x, -shift_y), axis=roll_axes)
        lamda_shifted = np.roll(lamda, shift=(shift_x, shift_y), axis=roll_axes)
        dlamda_dx += c[k, 0] * lamda_shifted  # c_{x,k} * λ(x + c_k*dx)
        dlamda_dy += c[k, 1] * lamda_shifted  # c_{y,k} * λ(x + c_k*dx)

    dlamda_dx *= (8/14) * 1/(10*n_dx)
    dlamda_dy *= (8/14) * 1/(10*n_dy)
    
    return dlamda_dx, dlamda_dy


def c_first_derivative_alt(φ, n_dx=1.0, n_dy=1.0):
    dφ_dx = np.zeros_like(φ)
    dφ_dy = np.zeros_like(φ)
    for i in range(1, 9):
        φ_neighbor = np.roll(np.roll(φ, c[i][0], axis=0), c[i][1], axis=1)
        dφ_dx += (3/(2*n_dx)) * E[i] * c[i][0] * φ_neighbor
        dφ_dy += (3/(2*n_dy)) * E[i] * c[i][1] * φ_neighbor

    return dφ_dx, dφ_dy


#Inamuro eq(13): second derivatives partial - partial dlambda²/dx_a²
def c_second_derivative_old(lamda, n_dx=1.0, n_dy=1.0):
    """
    Original eq(13) in Inamuro
    Compute ∂²λ/∂x² and ∂²λ/∂y² using Eq. (13): (1/(5*dx)) * (sum(λ(x + c_i*dx)) - 14*λ(x))
    lamda: Scalar (nx, ny) or vector (2, nx, ny) field
    n_dx, n_dy: Lattice spacing in x, y directions
    Returns: d2lamda_dx2, d2lamda_dy2
    """
    is_3d = len(lamda.shape) == 3 and lamda.shape[0] == 2
    is_2d = len(lamda.shape) == 2
    if not (is_2d or is_3d):
        raise ValueError(f"Expected lamda shape (202,52) or (2,202,52), got {lamda.shape}")

    sum_neighbors = np.zeros_like(lamda)
    roll_axes = (1, 2) if is_3d else (0, 1)
    for k in range(1, 9):
        shift_x, shift_y = c[k, 0], c[k, 1]
        #sum_neighbors += np.roll(lamda, shift=(-shift_x, -shift_y), axis=roll_axes)
        sum_neighbors += np.roll(lamda, shift=(shift_x, shift_y), axis=roll_axes)

    d2lamda_dx2 = (8/14) * 1/(5*n_dx)*(sum_neighbors - 8 * lamda) 
    d2lamda_dy2 = (8/14) * 1/(5*n_dy)*(sum_neighbors - 8 * lamda) 

    return d2lamda_dx2, d2lamda_dy2


def c_second_derivative(lamda, n_dx=1.0, n_dy=1.0):
    """
    Vectorized version of eq(13) in Inamuro - 2D scalar field only.
    Compute ∂²λ/∂x² and ∂²λ/∂y² using weighted sum over pre-shifted neighbors:
    scale * (sum_neighbors - 8 * lamda), where sum_neighbors = sum over 8 directions.
    Assumes lamda is scalar 2D field (nx, ny). Removed 3D support for simplicity.
    """
    # Enforce 2D scalar input (matches your usage: called on _phi or components separately)
    if len(lamda.shape) != 2:
        raise ValueError(f"Expected lamda shape (nx, ny) like (202,52), got {lamda.shape}")

    # Precompute scaling factors (cache globally if called often)
    scale_x = (8.0/14.0) / (5.0 * n_dx)
    scale_y = (8.0/14.0) / (5.0 * n_dy)

    # Roll axes for 2D: (0=x, 1=y)
    roll_axes = (0, 1)

    # Extract neighbor directions (k=1 to 8) - precompute globally if possible
    neigh_c = c[1:9]  # Shape (8, 2) - constants

    # Create shifted versions for all 8 directions (list comprehension for clarity)
    shifted_neighbors = [
        np.roll(lamda, shift=(neigh_c[k, 0], neigh_c[k, 1]), axis=roll_axes) for k in range(8)
    ]
    # Stack to (8, nx, ny) for vectorized sum
    neighbors_stack = np.stack(shifted_neighbors, axis=0)  # Shape (8, nx, ny)

    # Vectorized sum over neighbors (single BLAS call, no loop)
    sum_neighbors = np.sum(neighbors_stack, axis=0)  # Shape (nx, ny)

    # Apply formula: scale * (sum - 8 * lamda)
    d2lamda_dx2 = scale_x * (sum_neighbors - 8.0 * lamda)
    d2lamda_dy2 = scale_y * (sum_neighbors - 8.0 * lamda)

    return d2lamda_dx2, d2lamda_dy2


def c_second_derivative_alt(φ, n_dx=1.0, n_dy=1.0):
    d2φ_dx2 = np.zeros_like(φ)
    d2φ_dy2 = np.zeros_like(φ)
    for i in range(1, 9):
        φ_neighbor = np.roll(np.roll(φ, c[i][0], axis=0), c[i][1], axis=1)
        d2φ_dx2 += (3/(2*n_dx)) * E[i] * (φ_neighbor - (5/9) * φ)
        d2φ_dy2 += (3/(2*n_dy)) * E[i] * (φ_neighbor - (5/9) * φ)

    return d2φ_dx2, d2φ_dy2


def c_laplacian(lamda, n_dx=1.0, n_dy=1.0):
    """
    Compute ∇²λ = ∂²λ/∂x² + ∂²λ/∂y² using Eq. (13) for each direction
    """
    d2lamda_dx2, d2lamda_dy2 = c_second_derivative(lamda, n_dx, n_dy)
    laplacian = d2lamda_dx2 + d2lamda_dy2
    return laplacian


#Inamuro eq(9): Gab - shear terms
def Gab(_phi):
    dphi_x, dphi_y = c_first_derivative(_phi)

    Nx, Ny = _phi.shape
    G = np.zeros((Nx, Ny, 2, 2))

    # |grad phi|^2
    mag2 = dphi_x**2 + dphi_y**2

    G[:, :, 0, 0] = (9/2) * dphi_x * dphi_x - (3/2) * mag2 #G_xx
    G[:, :, 0, 1] = (9/2) * dphi_x * dphi_y #G_xy
    G[:, :, 1, 0] = (9/2) * dphi_y * dphi_x #G_yx
    G[:, :, 1, 1] = (9/2) * dphi_y * dphi_y - (3/2) * mag2 #G_yy

    return G


#Inamuro eq(14 & 15): density rho and viscosity mu
def density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L):
    _rho = np.zeros_like(_phi)
    _mu = np.zeros_like(_phi)

    delta_rho = rho_L - rho_G
    phi_dash_star = (phi_star_L + phi_star_G) / 2
    phi_delta_star = phi_star_L - phi_star_G   

    #if _phi < phi_star_G:
    #    _rho = rho_G
    mask1 = _phi < phi_star_G
    _rho[mask1] = rho_G
    #elif _phi >= phi_star_G and _phi <= phi_star_L:
    #    _rho = rho_center
    mask2 = (_phi >= phi_star_G) & (_phi <= phi_star_L)
    rho_center = (delta_rho/2) * (np.sin(np.pi * ((_phi[mask2] - phi_dash_star)/phi_delta_star)) + 1) + rho_G   
    _rho[mask2] = rho_center
    #elif _phi > phi_star_L:
    #    _rho = rho_L
    mask3 = _phi > phi_star_L
    _rho[mask3] = rho_L

    _mu = ((_rho - rho_G) / (rho_L - rho_G)) * (mu_L - mu_G) + mu_G

    return _rho, _mu


#Inamuro eq(23): relaxation time tau_h
def tau_h(_rho):
    _tau_h = 1/_rho + 1/2
    return _tau_h


#Inamuro eq(22,24): evolution equation of the velocity distribution function h(i) and pressure
def ph_old(hn, _rho, u_ckl_star, n_dx=1.0, n_dy=1.0):
    _p = np.sum(hn, axis=0)  # Include i=0

    shift_x = c * n_dx
    x_shifts = shift_x[:,0]
    y_shifts = shift_x[:,1]
    E_exp = np.expand_dims(E, axis=(1,2))
    p_exp = np.expand_dims(_p, axis=0)
    h_eq = E_exp * p_exp

    tau_h = 1./_rho + 1./2. #(Xn,Yn)
    omega = 1.0 / tau_h  # 1/tau_h for BGK

    du_dx, _ = c_first_derivative(u_ckl_star[0])
    _, dv_dy = c_first_derivative(u_ckl_star[1])
    div_u = du_dx + dv_dy
    div_u_exp = np.expand_dims(div_u, axis=0)

    #full evolution equation
    collision = hn - omega * (hn - h_eq) - (1./3.) * E_exp * div_u_exp * n_dx
    hn_plus1 = np.zeros_like(hn)
    hn_plus1[0] = collision[0]
    for k in range(1,9):
        temp = np.roll(collision[k], shift=x_shifts[k], axis=0)
        hn_plus1[k] = np.roll(temp,shift=y_shifts[k], axis=1)

    _p = np.sum(hn_plus1, axis=0)     

    return _p, hn_plus1


def ph(hn, _rho, u_ckl_star, iteration, n_dx=1.0, n_dy=1.0):
    """
    Vectorized evolution for h (pressure distribution) - D2Q9 streaming/collision.
    Replaces per-direction roll loop with batch shifts on stacked collision[1:9].
    Assumes hn shape (9, nx, ny), periodic boundaries via roll.
    """
    # Pressure from hn (sum over directions, including i=0)
    _p = np.sum(hn, axis=0)  # Shape (nx, ny)

    # Equilibrium: h_eq = E_i * p (broadcast)
    E_exp = E[:, np.newaxis, np.newaxis]  # (9, 1, 1) - expand once
    h_eq = E_exp * _p  # Broadcast to (9, nx, ny)

    # BGK relaxation
    tau_h = 1.0 / _rho + 0.5  # (nx, ny)
    omega = 1.0 / tau_h  # (nx, ny)
    omega_exp = omega[np.newaxis, :, :]  # (1, nx, ny) for broadcasting

    # Divergence term: (1/3) * E_i * div_u * n_dx
    du_dx, _ = c_first_derivative(u_ckl_star[0], n_dx, n_dy)
    _, dv_dy = c_first_derivative(u_ckl_star[1], n_dx, n_dy)
    ######################################## 6. Switch Derivative Scheme: I ########################
    du_dx, _ = np.gradient(u_ckl_star[0], n_dx, n_dy)
    _, dv_dy = np.gradient(u_ckl_star[1], n_dx, n_dy)
    ################################################################################################
    div_u = du_dx + dv_dy

    ############# 4. Fix Boundary Inconsistencies for div_u=0 Enforcement ###########################
    # In ph(), after div_u compute
    div_u[:, [0, Yn+1]] = 0.0  # Enforce no div at walls
    div_u[[0, Xn+1], :] = div_u[[1, Xn], :]  # Extrapolate periodic but clamp
    #################################################################################################

    div_u_exp = div_u[np.newaxis, :, :]  # (1, nx, ny)


    ############ 1. Inspect and Damp the Source Term (div_u) Directly ###############################
    # In ph(), before forcing_like_term
    du_dx, _ = c_first_derivative(u_ckl_star[0], n_dx, n_dy)
    _, dv_dy = c_first_derivative(u_ckl_star[1], n_dx, n_dy)
    div_u_raw = du_dx + dv_dy
    max_div_u = np.max(np.abs(div_u_raw))
    if iteration % 100 == 0:
        print(f"Iter {iteration} ph_start: max|div_u_raw|={max_div_u:.3e}")
    if max_div_u > 1.0:  # Threshold: paper assumes small div_u <<1
        div_u = div_u_raw / (1.0 + max_div_u)  # Normalize/damp extremes
    else:
        div_u = div_u_raw
    div_u_exp = div_u[np.newaxis, :, :]
    ##################################################################################################

    forcing_like_term = (1.0 / 3.0) * E_exp * div_u_exp * n_dx  # (9, nx, ny)

    # Collision in vectorized form: hn - omega*(hn - h_eq) - term
    # Broadcast omega and h_eq to (9, nx, ny) via subtraction rules
    collision = hn - omega_exp * (hn - h_eq) - forcing_like_term  # (9, nx, ny)

    # Streaming: i=0 stays (no shift), i=1-8 shift by c[k] in x,y
    hn_plus1 = np.empty_like(hn)
    hn_plus1[0] = collision[0]  # Rest particle, no stream

    # Vectorized streaming for neighbors (k=1 to 8)
    neigh_collision = collision[1:9]  # (8, nx, ny)
    neigh_c = c[1:9]  # (8, 2) - directions

    # Batch roll: roll in x then y - nested rolls on axis 0 of temp stack
    # First roll in x (axis=1 in original array -> axis=0 after slice, but adjust)
    # Reshape/stack trick: roll each direction independently via list comp, then assign
    # But for speed, use stride tricks or just list (faster than loop in py, but vectorized assign)
    # Actually, precompute shifts and assign in batch - but np.roll doesn't batch natively.
    # Optimal: keep list comp for clarity/speed (8 rolls still, but no inner py loop overhead)
    streamed_neighbors = [
        np.roll(np.roll(neigh_collision[i], shift=neigh_c[i, 0], axis=0),  # x-shift (axis=0 -> x)
                shift=neigh_c[i, 1], axis=1)  # y-shift (axis=1 -> y)
        for i in range(8)
    ]
    hn_plus1[1:9] = np.stack(streamed_neighbors, axis=0)  # (8, nx, ny) -> assign

    # Update pressure from streamed hn_plus1
    _p_new = np.sum(hn_plus1, axis=0)

    return _p_new, hn_plus1    


def velocity_gradient(u_ckl, n_dx=1.0, n_dy=1.0):
    """
    Compute ∂u_beta/∂x_alpha for a 2D velocity field.
    Returns array with shape (2, 2, nx, ny).
    """
    u_x, u_y = u_ckl  # unpack

    # Compute derivatives (np.gradient returns [∂/∂x0, ∂/∂x1])
    du_x_dx, du_x_dy = np.gradient(u_x, n_dx, n_dy, edge_order=2)
    du_y_dx, du_y_dy = np.gradient(u_y, n_dx, n_dy, edge_order=2)

    # Stack properly into 4D tensor (β, α, nx, ny)
    du_b_dx_a = np.stack([
        np.stack([du_x_dx, du_x_dy], axis=0),   # β = 0
        np.stack([du_y_dx, du_y_dy], axis=0)    # β = 1
    ], axis=0)

    return du_b_dx_a


#Inamuro eq(3): calculation of the predicted velocity of the two phase fluid
def gi_old(_gi, _gi_c, u_ckl, rho, mu, iteration):
    du_x_dx, du_x_dy = c_first_derivative(u_ckl[0,:,:], n_dx, n_dy)
    du_y_dx, du_y_dy = c_first_derivative(u_ckl[1,:,:], n_dx, n_dy)

    S_xx = 2 * du_x_dx
    S_yy = 2 * du_y_dy
    S_xy = du_y_dx + du_x_dy
    S_yx = S_xy 

    sigma_xx = mu * S_xx
    sigma_yy = mu * S_yy
    sigma_xy = mu * S_xy
    sigma_yx = mu * S_yx

    div_sigma_x = (np.roll(sigma_xx, -1, axis=1) - np.roll(sigma_xx, 1, axis=1) + 
                   np.roll(sigma_xy, -1, axis=0) - np.roll(sigma_xy, 1, axis=0)) / (2 * n_dx)
    div_sigma_y = (np.roll(sigma_yx, -1, axis=1) - np.roll(sigma_yx, 1, axis=1) + 
                   np.roll(sigma_yy, -1, axis=0) - np.roll(sigma_yy, 1, axis=0)) / (2 * n_dx)

    abs_div_sigma_x = np.max(np.abs(div_sigma_x))
    GrowthMetric_gi_div_sigma_x.append((iteration, abs_div_sigma_x))    
    abs_div_sigma_y = np.max(np.abs(div_sigma_y))
    GrowthMetric_gi_div_sigma_y.append((iteration, abs_div_sigma_y))

    _force = np.zeros((9,))
    _gi_old = np.copy(_gi)
    _gi_star = np.zeros_like(_gi)

    # Collision step
    for i in range(9):
        # Take derivative along the lattice direction c[i]
        # For D2Q9, you can select x or y component as needed
        viscous_term = 3 * E[i] * (c[i,0] * div_sigma_x/rho + c[i,1] * div_sigma_y/rho) * n_dx

        _gi_star[i,:,:] = (
            _gi_old[i,:,:]
            - (1/tau_g) * (_gi_old[i,:,:] - _gi_c[i,:,:])
            + viscous_term
            #+ _force
        )

    # Streaming step
    for i in range(9):
        _gi[i,:,:] = np.roll(_gi_star[i,:,:], (c[i, 0], c[i, 1]), axis=(0, 1))        

    return _gi


def gi(_gi, _gi_c, u_ckl, rho, mu, iteration):
    """
    Vectorized Inamuro eq(3): predicted velocity evolution for gi in two-phase LBM.
    Full batch operations on (9, nx, ny) - assumes padded grids (e.g., 202x52).
    Collision via broadcasting, streaming via batched rolls.
    """
    # Velocity gradients on full grid (nx, ny = 202,52)
    grads = [c_first_derivative(u_ckl[i], n_dx, n_dy) for i in range(2)]
    du_x_dx, du_x_dy = grads[0]
    du_y_dx, du_y_dy = grads[1]

    # Strain rate tensor S
    S_xx = 2 * du_x_dx
    S_yy = 2 * du_y_dy
    S_xy = du_y_dx + du_x_dy
    S_yx = S_xy

    # Stress sigma = mu * S
    sigma_xx = mu * S_xx
    sigma_yy = mu * S_yy
    sigma_xy = mu * S_xy
    sigma_yx = sigma_xy

    # Divergence via central differences (use n_dy for y-derivs)
    d_sigma_xx_dx = (np.roll(sigma_xx, -1, axis=1) - np.roll(sigma_xx, 1, axis=1)) / (2 * n_dx)
    d_sigma_xy_dy = (np.roll(sigma_xy, -1, axis=0) - np.roll(sigma_xy, 1, axis=0)) / (2 * n_dy)
    div_sigma_x = d_sigma_xx_dx + d_sigma_xy_dy

    d_sigma_yx_dx = (np.roll(sigma_yx, -1, axis=1) - np.roll(sigma_yx, 1, axis=1)) / (2 * n_dx)
    d_sigma_yy_dy = (np.roll(sigma_yy, -1, axis=0) - np.roll(sigma_yy, 1, axis=0)) / (2 * n_dy)
    div_sigma_y = d_sigma_yx_dx + d_sigma_yy_dy

    # Metrics
    abs_div_sigma_x = np.max(np.abs(div_sigma_x))
    GrowthMetric_gi_div_sigma_x.append((iteration, abs_div_sigma_x))
    abs_div_sigma_y = np.max(np.abs(div_sigma_y))
    GrowthMetric_gi_div_sigma_y.append((iteration, abs_div_sigma_y))

    # Expand for viscous term broadcasting over i=0..8
    div_sigma_x_exp = div_sigma_x[np.newaxis, :, :]  # (1, nx, ny)
    div_sigma_y_exp = div_sigma_y[np.newaxis, :, :]
    rho_exp = rho[np.newaxis, :, :]  # (1, nx, ny)
    c_x_exp = c[:, 0][:, np.newaxis, np.newaxis]  # (9, 1, 1)
    c_y_exp = c[:, 1][:, np.newaxis, np.newaxis]
    E_exp = E[:, np.newaxis, np.newaxis]  # (9, 1, 1)

    # Viscous term: 3 * E_i * (c_{i,x} * div_x / rho + c_{i,y} * div_y / rho) * n_dx
    dot_term = c_x_exp * (div_sigma_x_exp / rho_exp) + c_y_exp * (div_sigma_y_exp / rho_exp)
    viscous_term = 3.0 * E_exp * dot_term * n_dx  # (9, nx, ny)

    # Collision: fully vectorized BGK + forcing
    _gi_old = np.copy(_gi)  # (9, nx, ny)
    _gi_star = _gi_old - (1.0 / tau_g) * (_gi_old - _gi_c) + viscous_term

    # Streaming: batched rolls for all directions
    # List comp applies shifts, stack reassembles
    streamed = [
        np.roll(_gi_star[i], shift=(c[i, 0], c[i, 1]), axis=(0, 1)) for i in range(9)
    ]
    _gi[:] = np.stack(streamed, axis=0)  # In-place overwrite

    return _gi


def force_(F_lattice, rho):
    _force = F_lattice[:, None, None]* rho 

    return _force


def force(i, F_lattice):
    dot_product = np.dot(c[i], F_lattice)
    _force = E[i] * dot_product / Cs**2
    return _force


#Inamuro eq(7): calculation of predicted velocity of the two phase fluid - collision term
def gi_c_old(u, rho, tau_g, Kg, iteration):
    grad_rho_x,grad_rho_y = c_first_derivative(rho)
    _gi_c = np.zeros((9, Xn+2, Yn+2))
    ones = np.ones((Xn+2, Yn+2))

    #ua*ua
    u_dot_u = np.sum(u**2, axis=0)    

    #_gi_c
    for i in range(9):
        #cia*ua
        c_dot_u = np.einsum('i,ikl->kl', c[i], u)

        #cia*cib*ua*ub
        c_dot_u_tensor = np.einsum('a,b,axy,bxy->xy', c[i], c[i], u, u) #c_dot_u**2

        #(dub/dxa + dua/dxb)
        dua_dx,dua_dy = c_first_derivative(u[0,:,:])  
        dub_dx,dub_dy = c_first_derivative(u[1,:,:])  
        term_xx = (dua_dx + dua_dx) * c[i,0] * c[i,0]
        term_xy = (dua_dy + dub_dx) * c[i,0] * c[i,1]
        term_yx = (dub_dx + dua_dy) * c[i,1] * c[i,0]
        term_yy = (dub_dy + dub_dy) * c[i,1] * c[i,1]
        velocity_gradient_term = term_xx + term_xy + term_yx + term_yy

        #Gab(rho)*cia*cib
        Gab_phi = Gab(_phi)  
        _Gab = np.einsum("a,b,xyab->xy", c[i], c[i], Gab_phi)

        term1 = E[i]*ones
        term2 = E[i]*3*c_dot_u
        abs_term2 = np.max(np.abs(term2))
        GrowthMetric_gi_c_term2.append((iteration, abs_term2))

        term3 = E[i]*(3/2)*u_dot_u
        term4 = E[i]*(9/2)*c_dot_u_tensor
        term5 = E[i]*(3/2)*(tau_g - 1/2)*n_dx*velocity_gradient_term
        term6 = E[i]*(Kg/rho)*_Gab

        abs_term6 = np.max(np.abs(term6))
        GrowthMetric_gi_c_term6_Gab.append((iteration, abs_term6))
    
        term7 = (2/3)*F[i]*(Kg/rho)*grad_rho_x**2

        _gi_c[i] = term1 + term2 - term3 + term4 + term5 + term6 - term7

    return _gi_c


def gi_c(u, rho, tau_g, Kg, iteration):
    """
    Vectorized gi_c equilibrium (Inamuro) on full padded grid (9, Xn+2, Yn+2).
    Matches original loop exactly: all ops on full padded shapes, no slicing.
    Assumes inputs u (2, Xn+2, Yn+2), rho (Xn+2, Yn+2), _phi (Xn+2, Yn+2) are padded.
    Globals: c(9,2), E(9), F(9), n_dx, Gab func.
    """
    # Full padded shapes
    nx_pad, ny_pad = rho.shape  # e.g., (202,52)
    
    # Shared terms on full padded (derivs use rolls on full for periodic/accuracy)
    grad_rho_x, grad_rho_y = c_first_derivative(rho, n_dx, n_dy)  # (nx_pad, ny_pad)
    u_dot_u = np.sum(u**2, axis=0)  # (nx_pad, ny_pad)
    
    # Velocity gradients on full
    du_a_dx, du_a_dy = c_first_derivative(u[0], n_dx, n_dy)
    du_b_dx, du_b_dy = c_first_derivative(u[1], n_dx, n_dy)
    
    # Gab on full
    Gab_phi = Gab(_phi)  # (nx_pad, ny_pad, 2, 2)
    
    # Expansions for i-broadcast
    ones = np.ones((nx_pad, ny_pad))
    E_exp = E[:, np.newaxis, np.newaxis]  # (9,1,1)
    
    # term1: E[i] * 1
    term1 = E_exp * ones  # (9,nx,ny)
    
    # term2: E[i] * 3 * (c_i . u)
    c_dot_u = np.einsum('ia,axy->ixy', c, u)  # (9,nx,ny)
    term2 = E_exp * 3.0 * c_dot_u
    abs_term2 = np.max(np.abs(term2))
    GrowthMetric_gi_c_term2.append((iteration, abs_term2))
    
    # term3: E[i] * (3/2) * u_dot_u
    term3 = E_exp * (3.0 / 2.0) * u_dot_u
    
    # term4: E[i] * (9/2) * (c_dot_u)^2
    c_dot_u_tensor = c_dot_u ** 2  # Matches original comment c_dot_u**2
    term4 = E_exp * (9.0 / 2.0) * c_dot_u_tensor
    
    # term5: E[i] * (3/2) * (tau_g - 1/2) * n_dx * velocity_gradient_term
    # To match original exactly (including the buggy du+du), broadcast per-component on full grid
    c_x_sq = c[:, 0][:, np.newaxis, np.newaxis] ** 2  # (9,1,1)
    c_y_sq = c[:, 1][:, np.newaxis, np.newaxis] ** 2
    c_xy = (c[:, 0] * c[:, 1])[:, np.newaxis, np.newaxis]  # (9,1,1)
    
    term_xx = 2.0 * du_a_dx * c_x_sq  # (dua_dx + dua_dx) * c_x^2
    term_xy = (du_a_dy + du_b_dx) * c_xy
    term_yx = (du_b_dx + du_a_dy) * c_xy
    term_yy = 2.0 * du_b_dy * c_y_sq  # (dub_dy + dub_dy) * c_y^2
    velocity_gradient_term = term_xx + term_xy + term_yx + term_yy  # (9,nx,ny)
    
    term5 = E_exp * (3.0 / 2.0) * (tau_g - 0.5) * n_dx * velocity_gradient_term
    
    # term6: E[i] * (Kg/rho) * (c_i . Gab . c_i)
    _Gab = np.einsum('ia,ib,xyab->ixy', c, c, Gab_phi)  # (9,nx,ny)
    term6 = E_exp * (Kg / rho) * _Gab
    abs_term6 = np.max(np.abs(term6))
    GrowthMetric_gi_c_term6_Gab.append((iteration, abs_term6))
    
    # term7: (2/3) * F[i] * (Kg/rho) * grad_rho_x**2
    F_exp = F[:, np.newaxis, np.newaxis]  # (9,1,1)
    term7 = (2.0 / 3.0) * F_exp * (Kg / rho) * (grad_rho_x ** 2)

    # Assemble: exact match to loop addition
    _gi_c = term1 + term2 - term3 + term4 + term5 + term6 - term7  # (9,nx_pad,ny_pad)

    ######################## 5. Incorporate Gravity Consistently and Ramp It: ####################################
    # In gi_c, after term7
    # --- Buoyancy Gravity Term (after term7, uses your exact vars/shapes) ---
    # Lattice y-velocity components expanded (matches D2Q9 c[:,1])
    c_y = c[:, 1]  # [0, 0, 1, 0, -1, 1, 1, -1, -1]
    c_y_exp = c_y[:, np.newaxis, np.newaxis]  # (9,1,1) broadcastable to grid

    # Your ramped lattice gravity (F_lattice from globals, iteration from func arg)
    ramp_factor = min(1.0, iteration / 5000.0)  # Ramp over 5000 steps
    F_lattice_ramped = F_lattice * ramp_factor
    F_lattice_y = F_lattice_ramped[1]  # y-component

    # Boussinesq buoyancy: projects y-force via density contrast
    buoyancy_factor = F_lattice_y * (rho - rho_G) / np.maximum(rho, 0.1)  # rho_G=your gas density
    buoyancy_factor_exp = buoyancy_factor[np.newaxis, :, :]  # (1, nx, ny)

    # Directional add: E_i * factor * c_{i,y} (shape matches _gi_c)
    buoyancy = E_exp * buoyancy_factor_exp * c_y_exp
    _gi_c += buoyancy  # Integrates into collision equilibrium
    # --- End Term ---
    ##############################################################################################################
    
    return _gi_c


#Inamuro eq(2): calculation of the order parameter which distiguishes the two phases
def fi_old(_fi, _fi_c, tau_f):
    _fi_old = np.copy(_fi)
    _fi_star = np.zeros_like(_fi)
    for i in range(9):
        _fi_star[i,:,:] = _fi_old[i,:,:] - (1/tau_f) * (_fi_old[i,:,:] - _fi_c[i,:,:])
        if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi_star[i] < 0):
            print(f"Iter {iteration}: Negative _fi_star[{i}] at y=50: min={np.min(_fi_star[i][:,50]):.3e}")
    for i in range(9):
        _fi[i,:,:] = np.roll(_fi_star[i,:,:], (c[i, 0], c[i, 1]), axis=(0, 1))
    if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi < 0):
        raise ValueError(f"Iter {iteration}: Negative _fi: min={np.min(_fi):.3e}")
    return _fi


def fi(_fi, _fi_c, tau_f):
    """
    Fully vectorized collision and streaming for fi distribution (D2Q9).
    No Python loops anywhere - negative debug uses vectorized argmin/where.
    Assumes globals: RAISE_LESS_THAN_ZERO_ERROR, iteration, c(9,2).
    """
    # Collision: fully vectorized BGK
    _fi_old = np.copy(_fi)  # (9, nx, ny)
    omega_f = 1.0 / tau_f
    _fi_star = _fi_old - omega_f * (_fi_old - _fi_c)  # Broadcast all

    # Vectorized negative check on _fi_star
    if RAISE_LESS_THAN_ZERO_ERROR:
        neg_mask = _fi_star < 0  # (9,nx,ny)
        if np.any(neg_mask):
            # Find directions with negatives
            neg_dirs = np.where(np.any(neg_mask, axis=(1,2)))[0]  # array of i's
            for i in neg_dirs:
                print(f"Iter {iteration}: Negative _fi_star[{i}] at y=50: min={np.min(_fi_star[i, :, 50]):.3e}")

    # Streaming: batched rolls
    streamed = [
        np.roll(_fi_star[i], shift=(c[i, 0], c[i, 1]), axis=(0, 1)) for i in range(9)
    ]
    _fi[:] = np.stack(streamed, axis=0)

    # Vectorized post-check
    if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi < 0):
        raise ValueError(f"Iter {iteration}: Negative _fi: min={np.min(_fi):.3e}")

    return _fi


#Inamuro eq(6): calculation of the order parameter which distiguishes the two phases - collision term
def fi_c_old(u, Kf, F, _phi):
    _fi_c = np.zeros((9, Xn+2, Yn+2))
    dphi_dxa, dphi_dya = c_first_derivative(_phi, n_dx, n_dy)
    if RAISE_NaN_ERROR and (np.any(np.isnan(dphi_dxa)) or np.any(np.isnan(dphi_dya))):
        raise ValueError(f"NaN in dphi_dxa or dphi_dya")
    laplacian_phi = c_laplacian(_phi, n_dx, n_dy)
    if RAISE_NaN_ERROR and np.any(np.isnan(laplacian_phi)):
        raise ValueError(f"NaN in laplacian_phi")
    G = Gab(_phi)
    if RAISE_NaN_ERROR and np.any(np.isnan(G)):
        raise ValueError(f"NaN in Gab")
    p0_val = p0(_phi)
    if RAISE_NaN_ERROR and np.any(np.isnan(p0_val)):
        raise ValueError(f"NaN in p0")
    for i in range(9):
        c_dot_u = c[i, 0] * u[0] + c[i, 1] * u[1]
        _Gab = (c[i, 0] * c[i, 0] * G[:, :, 0, 0] + 
                c[i, 0] * c[i, 1] * G[:, :, 0, 1] + 
                c[i, 1] * c[i, 0] * G[:, :, 1, 0] + 
                c[i, 1] * c[i, 1] * G[:, :, 1, 1])
        term1 = H[i] * _phi
        term2 = F[i] * p0_val
        term3 = F[i] * Kf * _phi * laplacian_phi
        term4 = F[i] * Kf/6 * (dphi_dxa**2 + dphi_dya**2)
        term5 = 3 * E[i] * _phi * c_dot_u
        term6 = E[i] * Kf * _Gab
        _fi_c[i] = term1 + term2 - term3 - term4 + term5 + term6
        if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi_c[i] < 0):
            print(f"Iter {iteration}: _fi_c[{i}] at y=50: min={np.min(_fi_c[i][:,50]):.3e}, terms at y=50: t1={np.min(term1[:,50]):.3e}, t2={np.min(term2[:,50]):.3e}, t3={np.min(term3[:,50]):.3e}, t4={np.min(term4[:,50]):.3e}, t5={np.min(term5[:,50]):.3e}, t6={np.min(term6[:,50]):.3e}")
    if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi_c < 0):
        raise ValueError(f"Iter {iteration}: Negative _fi_c: min={np.min(_fi_c):.3e}")
    return _fi_c


def fi_c(u, Kf, F, _phi):
    """
    Vectorized fi_c equilibrium on full padded grid (9, Xn+2, Yn+2).
    Full batch operations over i=0..8 directions via broadcasting/einsum.
    Matches original logic exactly, including per-term debug prints.
    Assumes globals: H(9), E(9), c(9,2), n_dx, n_dy, RAISE_NaN_ERROR, RAISE_LESS_THAN_ZERO_ERROR, iteration.
    """
    # Full padded shapes
    nx_pad, ny_pad = _phi.shape  # (Xn+2, Yn+2) e.g., 202x52
    
    # Shared derivatives and terms on full grid
    dphi_dxa, dphi_dya = c_first_derivative(_phi, n_dx, n_dy)  # (nx_pad, ny_pad)
    if RAISE_NaN_ERROR and (np.any(np.isnan(dphi_dxa)) or np.any(np.isnan(dphi_dya))):
        raise ValueError("NaN in dphi_dxa or dphi_dya")
    
    laplacian_phi = c_laplacian(_phi, n_dx, n_dy)  # (nx_pad, ny_pad)
    if RAISE_NaN_ERROR and np.any(np.isnan(laplacian_phi)):
        raise ValueError("NaN in laplacian_phi")
    
    G = Gab(_phi)  # (nx_pad, ny_pad, 2, 2)
    if RAISE_NaN_ERROR and np.any(np.isnan(G)):
        raise ValueError("NaN in Gab")
    
    p0_val = p0(_phi)  # (nx_pad, ny_pad)
    if RAISE_NaN_ERROR and np.any(np.isnan(p0_val)):
        raise ValueError("NaN in p0")
    
    # Expansions for broadcasting over i
    H_exp = H[:, np.newaxis, np.newaxis]  # (9,1,1)
    F_exp = F[:, np.newaxis, np.newaxis]  # (9,1,1)
    E_exp = E[:, np.newaxis, np.newaxis]  # (9,1,1)
    c_exp = c[:, :, np.newaxis, np.newaxis]  # (9,2,1,1)
    
    # term1: H[i] * _phi
    term1 = H_exp * _phi  # (9,nx,ny)
    
    # term2: F[i] * p0_val
    term2 = F_exp * p0_val
    
    # term3: F[i] * Kf * _phi * laplacian_phi
    term3 = F_exp * Kf * _phi * laplacian_phi
    
    # term4: F[i] * Kf/6 * (dphi_dxa**2 + dphi_dya**2)
    grad_phi_sq = dphi_dxa**2 + dphi_dya**2
    term4 = F_exp * (Kf / 6.0) * grad_phi_sq
    
    # term5: 3 * E[i] * _phi * c_dot_u
    c_dot_u = np.einsum('ia,axy->ixy', c, u)  # (9,nx,ny) - c_i . u
    term5 = 3.0 * E_exp * _phi * c_dot_u
    
    # term6: E[i] * Kf * _Gab (c^T G c)
    _Gab = np.einsum('ia,ib,xyab->ixy', c, c, G)  # (9,nx,ny)
    term6 = E_exp * Kf * _Gab
    
    # Assemble _fi_c
    _fi_c = term1 + term2 - term3 - term4 + term5 + term6  # (9,nx_pad,ny_pad)
    
    # Vectorized negative check with per-i debug
    if RAISE_LESS_THAN_ZERO_ERROR:
        neg_mask = _fi_c < 0  # (9,nx,ny)
        if np.any(neg_mask):
            # Directions with negatives
            neg_dirs = np.where(np.any(neg_mask, axis=(1,2)))[0]
            y50_slice = (slice(None), 50)  # For min at y=50
            for i in neg_dirs:
                min_val = np.min(_fi_c[i][y50_slice])
                t1_min = np.min(term1[i][y50_slice])
                t2_min = np.min(term2[i][y50_slice])
                t3_min = np.min(term3[i][y50_slice])
                t4_min = np.min(term4[i][y50_slice])
                t5_min = np.min(term5[i][y50_slice])
                t6_min = np.min(term6[i][y50_slice])
                print(f"Iter {iteration}: _fi_c[{i}] at y=50: min={min_val:.3e}, terms at y=50: t1={t1_min:.3e}, t2={t2_min:.3e}, t3={t3_min:.3e}, t4={t4_min:.3e}, t5={t5_min:.3e}, t6={t6_min:.3e}")
    
    if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi_c < 0):
        raise ValueError(f"Iter {iteration}: Negative _fi_c: min={np.min(_fi_c):.3e}")
    
    return _fi_c


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


def amplitude_plot(ax1, u_full_range, listIterations, axis, xlabel, ylabel, title, nx=Xn, ny=Yn, angle=45, font_size=10):
    for iteration, combined_u_ckl in u_full_range.items():
        u = combined_u_ckl[nx, 1:ny + 1]
        ax1.plot(u, axis, label=f"t={iteration}")
    ax1.grid()
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_title(title)
    ax1.set_ylim(-1, 51); ax1.set_yticks(np.arange(0, 51, 10))
    ax1.legend(ncol=max(1, len(u_full_range)//4), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)
    plt.setp(ax1.get_xticklabels(), rotation=angle, ha='right', fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.margins(x=0, y=0)
    x_min, x_max = ax1.get_xlim()
    ax1.set_xlim(x_min, x_max + 0.1*(x_max-x_min))
    ax1.figure.tight_layout()
    ax1.get_figure().set_size_inches(10, 6)


# Density profile
def density_profile(ax1, den_eq, nx, ny, iteration=0):
    # Calculate y-positions for 1/3, 1/2, and 2/3 of the channel height
    y_third = ny // 3  # 1/3 * ymax
    y_half = ny // 2   # 1/2 * ymax
    y_two_thirds = 2 * ny // 3  # 2/3 * ymax

    # Plot density along x for each y-position
    ax1.plot(den_eq[1:nx, y_third], label=f"y = {y_third} (~1/3 ymax)")
    ax1.plot(den_eq[1:nx, y_half], label=f"y = {y_half} (~1/2 ymax)")
    ax1.plot(den_eq[1:nx, y_two_thirds], label=f"y = {y_two_thirds} (~2/3 ymax)")

    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("Density [rho]")
    ax1.margins(x=0, y=0)
    ax1.set_title("Longitudinal density profile")
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_xlim(0, nx)
    ax1.set_xticks(np.linspace(0, nx, 5))
    ax1.grid()

    ax1.legend(loc='best')  # Add legend to distinguish the three lines

    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist   

    # Save in same directory as the script
    filename = "{0}_{1}_{2}.png".format(SCRIPT_FILENAME, "density_profile", iteration)
    save_path = os.path.join(images_dir, filename)

    ax1.figure.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved density profile: {save_path}")


def density_profile_transition(ax1, den_eq, x__position, y_position, nx, ny, granularity, iteration=0):
    # Fixed x-position at Xn/2
    #x_fixed = nx // 2  # Xn/2 = 100 for nx = 200

    # Define y-coordinates for the plot
    y_coords = np.arange(1, ny)  # y = 1 to ny-1 (1 to 50 for ny = 51)

    # Plot density (rho) on x-axis, y-coordinate on y-axis
    ax1.plot(den_eq[x__position, 1:ny], y_coords, label=f"x = {x__position}")

    ax1.set_xlabel("Density [rho]")
    ax1.set_ylabel("y-axis")
    ax1.margins(x=0, y=0)
    ax1.set_title("Vertical density profile at x = Xn/2")
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_xlim(-1, 52)  # Density range: rho_G - 2 = -1, rho_L + 2 = 52
    ax1.set_xticks(np.linspace(-1, 52, 5))
    ax1.set_ylim(y_position - granularity, y_position + granularity) 
    ax1.set_yticks(np.linspace(y_position - granularity, y_position + granularity, 5))
    ax1.grid()

    ax1.legend(loc='best')  # Add legend

    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist   

    # Save in same directory as the script
    filename = "{0}_{1}_{2}.png".format(SCRIPT_FILENAME, "density_profile", iteration)
    save_path = os.path.join(images_dir, filename)

    ax1.figure.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved density profile: {save_path}")


# Plot density profiles saved at given iterations
def density_profiles(ax, density_slices, x__position, nx, ny):
    """Compact overlay of density slices with right legend."""
    if not density_slices: return
    y_coords = np.arange(0, ny + 2)
    iterations, slices = zip(*density_slices)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, (it, rho_slice) in enumerate(zip(iterations, slices)):
        ax.plot(rho_slice, y_coords, color=colors[idx % 10], label=f"Iter {it}")
    ax.set(xlabel="Density [rho]", ylabel="y-axis", title=f"Evolution at x={x__position}",
           xlim=(-1, 52), xticks=np.linspace(-1, 52, 5), ylim=(0, ny + 1), yticks=np.linspace(0, ny + 1, 6))
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    ax.figure.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Milestone-Images"); os.makedirs(images_dir, exist_ok=True)
    save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_density_profile_evolution.png")
    ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")


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


def density_mapExt(ax, full_range, min, max, title, _iteration):
    print(f"Debug density_map: min={np.min(full_range):.6f}, max={np.max(full_range):.6f}, rho_out={min:.6f}, rho_in={max:.6f}")
    im = ax.imshow(full_range.T, interpolation='nearest', origin='lower', cmap='viridis', vmin=min, vmax=max)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(title)
    ax.margins(x=0, y=0)
    ax.set_aspect('auto')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_ticks([min, (min + max) / 2, max])  
    cbar.set_ticks(np.linspace(min, max, 3)) 

    # --- Save in subfolder 'images' ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist    

    # Save in same directory as the script
    filename = "{0}_{1}_{2}.png".format(SCRIPT_FILENAME, title, _iteration)
    save_path = os.path.join(images_dir, filename)
    ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')


def density_map_standalone(full_range, min, max, title, _iteration):
    print(f"Debug density_map: min={np.min(full_range):.6f}, max={np.max(full_range):.6f}, rho_out={min:.6f}, rho_in={max:.6f}")

    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(6,6))  # you can adjust the size

    # Plot the image
    im = ax.imshow(full_range.T, interpolation='nearest', origin='lower',
                   cmap='viridis', vmin=min, vmax=max)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(title)
    ax.margins(x=0, y=0)
    ax.set_aspect('auto')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(min, max, 3))  # min, mid, max
    cbar.set_label("Density")

    # --- Save in subfolder 'Milestone-Images' ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)

    filename = f"{SCRIPT_FILENAME}_{title}_{_iteration}.png"
    save_path = os.path.join(images_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # close the figure to free memory

 

# 2D Velocity map
def velocity_map(ax, u_magnitude, _iteration, title):
    im = ax.imshow(u_magnitude.T, interpolation='nearest', origin='lower', cmap='plasma')  
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(title)
    ax.margins(x=0, y=0)
    ax.set_aspect('auto')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Get folder where the script is located
    # --- Save in subfolder 'images' ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist     

    # Save image
    filename = "{0}_{1}.png".format(SCRIPT_FILENAME, title)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist 

    save_path = os.path.join(images_dir, filename)
    ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved velocity map: {images_dir}")


def filter_u_ckl_fullrange(velocities_dict, iterationsOfInterest):
    filtered_velocities = {iteration: velocities_dict[iteration] for iteration in iterationsOfInterest if iteration in velocities_dict}

    return filtered_velocities   


def plot_bounds(results, context, k=0, ax=None):
    """Plot iterative results, attaching to ax if provided."""
    if not results or not isinstance(results, (list, tuple)) or len(results[0]) < 2:
        return
    iterations = [r[0] for r in results]

    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)

    plt.xlabel("Iteration")
    filename = f"{SCRIPT_FILENAME}_{context}.png"
    
    match context:
        case "Invariants":
            plt.ylabel("Invariants (global x-momentum)")
            plt.title("Invariants vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="Invariants")
        case "GrowthMetric_uckl_x":
            plt.ylabel("GrowthMetric_uckl_x (peak abs. x-velocity)")
            plt.title("GrowthMetric_uckl_x vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="GrowthMetric_uckl_x")
        case "GrowthMetric_uckl_y":
            plt.ylabel("GrowthMetric_uckl_y (peak abs. y-velocity)")
            plt.title("GrowthMetric_uckl_y vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="GrowthMetric_uckl_y")
        case "AuxFields":
            plt.ylabel("AuxField1/AuxField2")
            plt.title("AuxField1/AuxField2 vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="AuxField1")
            plt.plot(iterations, [r[2] for r in results], label="AuxField2")
        case "u_ckl_x_bounds":
            plt.ylabel("u_ckl_x")
            plt.title("u_ckl_x_min and u_ckl_x_max vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="u_ckl_x_min")
            plt.plot(iterations, [r[2] for r in results], label="u_ckl_x_max")
            filename = f"{SCRIPT_FILENAME}_{context}_Kf{k}.png"
        case "u_ckl_y_bounds":
            plt.ylabel("u_ckl_y")
            plt.title("u_ckl_y_min and u_ckl_y_max vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="u_ckl_y_min")
            plt.plot(iterations, [r[2] for r in results], label="u_ckl_y_max")
            filename = f"{SCRIPT_FILENAME}_{context}_Kf{k}.png"
        case "rho_bounds":
            plt.ylabel("rho")
            plt.title("rho_min and rho_max vs Iteration")
            plt.plot(iterations, [r[1] for r in results], label="rho_min")
            plt.plot(iterations, [r[2] for r in results], label="rho_max")
        case _:
            return

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if ax is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "Milestone-Images")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")


def plot_bounds_ext(results, context, ax=None, series_labels=None, k=None, script_filename=None):
    """Generic plotter for iterative results with 2+ data series."""
    if results is None or not isinstance(results, (list, tuple)) or not results or len(results[0]) < 2:
        raise ValueError("results must be non-empty list of (iteration, values)")
    if isinstance(results, np.ndarray): results = results.tolist()
    
    iterations = [r[0] for r in results]
    data_series = list(zip(*[r[1:] for r in results]))
    n_series = len(data_series)
    series_labels = series_labels or [f"{context}_{i+1}" for i in range(n_series)]
    
    ylabel, title = context, f"{context} vs Iteration"
    
    # Use existing ax if provided, else create new figure
    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    
    for series, label in zip(data_series, series_labels):
        plt.plot(iterations, series, label=label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if ax is None:
        if script_filename is None: script_filename = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_filename}_{context}_Kf{k}.png" if k else f"{script_filename}_{context}.png"
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Milestone-Images")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved to {save_path}")


# New plot function (add after existing):
def plot_momentum_bounds(results, _filename, ax=None):
    iterations = [r[0] for r in results]
    invariant = [r[3] for r in results]  # Index 3
    
    # Use existing ax if provided, else create new figure
    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=(8, 5))
    plt.plot(iterations, invariant, label="total_mom_x", color="green")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Total invariant")
    plt.title("Total invariant conservation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save only if new figure
    if ax is None:
        filename = f"{SCRIPT_FILENAME}_{_filename}.png"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "Milestone-Images")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Momentum plot saved to {save_path}")


def save_phi_snapshot(_phi, iteration, phi_star_G, phi_star_L):
    """
    Save a snapshot plot of the order parameter φ and print min/max/mean stats.

    Args:
        _phi: 2D numpy array of φ values (Ny, Nx).
        iteration: Current iteration number (int).
        phi_star_G: Lower bound for colorbar (float).
        phi_star_L: Upper bound for colorbar (float).
        script_dir: Directory to save PNG (str; default: script's directory).

    Returns:
        None (saves PNG and prints stats).
    """
    # Ensure filename ends with .png
    # Save in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
    images_dir = os.path.join(script_dir, "Milestone-Images")
    os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist 

    # Create plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(_phi.T, origin='lower', cmap='RdBu', vmin=phi_star_G, vmax=phi_star_L)
    plt.colorbar(im, label='φ')
    plt.title(f'Order parameter φ at iteration {iteration}')
    plt.xlabel('x-index')
    plt.ylabel('y-index')

    # Save PNG
    _filename = f'phi_snapshot_iter_{iteration:05d}.png'
    filename = "{0}_{1}.png".format(SCRIPT_FILENAME, _filename)
    save_path = os.path.join(images_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print stats
    phi_min = np.min(_phi)
    phi_max = np.max(_phi)
    phi_mean = np.mean(_phi)
    print(f'φ at iter {iteration}: min={phi_min:.3f}, max={phi_max:.3f}, mean={phi_mean:.3f}')
    print(f'Saved φ snapshot: {save_path}')    


def initialize_nonlinear_phi_old(xn, yn, phi_star_g, phi_star_l, xi=2.0):
    """
    Initialize the order parameter _phi based on the specified distribution.
    
    Args:
        xn (int): Number of grid points in x-direction (Xn+2).
        yn (int): Number of grid points in y-direction (Yn+2).
        phi_star_g (float): Order parameter value for gas phase.
        phi_star_l (float): Order parameter value for liquid phase.
        xi (float, optional): Transition width parameter. Defaults to 2.0.
    
    Returns:
        np.ndarray: 2D array of shape (xn, yn) containing the initialized _phi values.
    """
    import numpy as np
    from scipy.special import erf

    # Define midlines for transitions
    y0_mid = (yn - 1) / 2  # Midline for the left half
    y0_two_thirds = 2 * (yn - 1) / 3  # Two-thirds line for the right half

    # Create meshgrid
    x, y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')

    # Initialize _phi array
    _phi = np.zeros((xn, yn), dtype=np.float64)

    # Left half (0 < x <= Xn/2)
    mask_left = x <= xn/2
    _phi[mask_left] = np.where(
        y[mask_left] < y0_mid,
        phi_star_l,
        phi_star_g
    )

    # Right half (Xn/2 < x <= Xn)
    mask_right = x > xn/2
    _phi[mask_right] = np.where(
        y[mask_right] < y0_two_thirds,
        phi_star_l,
        phi_star_g
    )

    # Apply smooth transition for left half (around y = Yn/2)
    transition_left = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * erf((y0_mid - y) / (np.sqrt(2) * xi))
    _phi[mask_left] = np.where(
        np.abs(y[mask_left] - y0_mid) <= 3 * xi,  # Limit transition to ~3*xi nodes
        transition_left[mask_left],
        _phi[mask_left]
    )

    # Apply smooth transition for right half (around y = 2*Yn/3)
    transition_right = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * erf((y0_two_thirds - y) / (np.sqrt(2) * xi))
    _phi[mask_right] = np.where(
        np.abs(y[mask_right] - y0_two_thirds) <= 3 * xi,  # Limit transition to ~3*xi nodes
        transition_right[mask_right],
        _phi[mask_right]
    )

    return _phi


def initialize_nonlinear_phi(xn, yn, phi_star_g, phi_star_l, xi=2.0):
    """
    Compact init of _phi with step phases and smooth erf transitions.
    Liquid regions: 
    - 0 < x <= Xn/2, 0 < y < Yn/2
    - Xn/2 < x <= 3*Xn/4, 0 < y < 2*Yn/3
    - 3*Xn/4 < x <= Xn, 0 < y < Yn/2
    Gas elsewhere. Transition layer: 2*xi nodes via erf.
    Args: xn, yn (padded grid), phi_star_g/l (gas/liquid values), xi (transition width).
    Returns: _phi (xn, yn) array.
    """
    x, y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')
    y_mid, y_23 = (yn - 1) / 2, 2 * (yn - 1) / 3
    phi_mid = (phi_star_l + phi_star_g) / 2
    phi_diff = (phi_star_l - phi_star_g) / 2
    sigma = np.sqrt(2) * xi

    # Base step: liquid regions
    _phi = np.where(
        (x <= xn / 2) & (y < y_mid) |  # Left: y < Yn/2
        (x > xn / 2) & (x <= 3 * xn / 4) & (y < y_23) |  # Mid: y < 2*Yn/3
        (x > 3 * xn / 4) & (y < y_mid),  # Right: y < Yn/2
        phi_star_l, phi_star_g
    )

    # Smooth erf transitions in 2*xi layer (~3*xi band for accuracy)
    trans_width = 3 * xi
    # Left and right use y_mid interface (0 < x <= Xn/2, 3*Xn/4 < x <= Xn)
    erf_factor = phi_diff * erf((y_mid - y) / sigma)
    mask_mid = ((x <= xn / 2) | (x > 3 * xn / 4)) & (np.abs(y - y_mid) <= trans_width)
    _phi[mask_mid] = phi_mid + erf_factor[mask_mid]

    # Middle region uses y_23 interface (Xn/2 < x <= 3*Xn/4)
    erf_factor_r = phi_diff * erf((y_23 - y) / sigma)
    mask_r = (x > xn / 2) & (x <= 3 * xn / 4) & (np.abs(y - y_23) <= trans_width)
    _phi[mask_r] = phi_mid + erf_factor_r[mask_r]

    return _phi


def initialize_phi_line(xn, yn, phi_star_g, phi_star_l, height=None, xi=2.0):
    """
    Initialize the order parameter _phi along a line at a specified height y of the channel,
    with a smooth transition from phi_star_l to phi_star_g across the x-axis.
    
    Args:
        xn (int): Number of grid points in x-direction (Xn+2).
        yn (int): Number of grid points in y-direction (Yn+2).
        phi_star_g (float): Order parameter value for gas phase.
        phi_star_l (float): Order parameter value for liquid phase.
        height (float, optional): Height y at which the transition occurs. Defaults to yn/2.
        xi (float, optional): Transition width parameter. Defaults to 2.0.
    
    Returns:
        np.ndarray: 2D array of shape (xn, yn) containing the initialized _phi values.
    """
    import numpy as np
    from scipy.special import erf

    # Default height to the middle of the channel if not specified
    y0 = (yn - 1) / 2 if height is None else height

    # Create meshgrid
    x, y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')

    # Initialize _phi array
    _phi = np.zeros((xn, yn), dtype=np.float64)

    # Set _phi to phi_star_l below the transition line and phi_star_g above
    _phi = np.where(
        y < y0,
        phi_star_l,
        phi_star_g
    )

    # Apply smooth transition using erf along the x-axis at height y0
    transition = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * erf((y0 - y) / (np.sqrt(2) * xi))
    _phi = np.where(
        np.abs(y - y0) <= 3 * xi,  # Limit transition to ~3*xi nodes vertically
        transition,
        _phi
    )

    return _phi


def update_ghost_nodes_top_bottom(_fi, _gi, Yn):
    """
    Vectorized update of ghost nodes at top and bottom boundaries.
    Applies mirror/extrapolation for fi and gi distributions based on lattice directions.
    Assumes _fi and _gi are (9, nx, Yn+2) with ghosts at y=0 and y=Yn+1.
    Called in main loop before bounceBackTopBottom2.
    """
    # Upward directions (c[i,1] > 0: i=2,5,6): mirror opposite from interior y=1 to ghost y=0
    up_dirs = [2, 5, 6]
    opp_up = [4, 7, 8]  # i+2
    _fi[up_dirs, :, 0] = _fi[opp_up, :, 1]
    _gi[up_dirs, :, 0] = _gi[opp_up, :, 1]

    # Downward directions (c[i,1] < 0: i=4,7,8): mirror opposite from interior y=Yn to ghost y=Yn+1
    down_dirs = [4, 7, 8]
    opp_down = [2, 5, 6]  # i-2
    _fi[down_dirs, :, Yn+1] = _fi[opp_down, :, Yn]
    _gi[down_dirs, :, Yn+1] = _gi[opp_down, :, Yn]

    # Zero y-velocity directions (i=0,1,3): copy/extrapolate from adjacent interior
    zero_y_dirs = [0, 1, 3]
    _fi[zero_y_dirs, :, 0] = _fi[zero_y_dirs, :, 1]
    _fi[zero_y_dirs, :, Yn+1] = _fi[zero_y_dirs, :, Yn]
    _gi[zero_y_dirs, :, 0] = _gi[zero_y_dirs, :, 1]
    _gi[zero_y_dirs, :, Yn+1] = _gi[zero_y_dirs, :, Yn]


def apply_periodic_boundary_conditions(_fi, _gi):
    _gi[:, 0, :] = _gi[:,Xn,:]
    _gi[:, Xn+1, :] = _gi[:,1,:]      
    _fi[:, 0, :] = _fi[:,Xn,:]
    _fi[:, Xn+1, :] = _fi[:,1,:]      


#preliminary
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)

#initialise
#average velocity, cartesion x,y-directions, k is y-position, l is x-position
rho_in_k = np.full((Yn+2), rho_in, dtype=np.float64)
rho_out_k = np.full((Yn+2), rho_out, dtype=np.float64)
u_ckl = np.zeros((2, Xn+2, Yn+2), dtype=np.float64)
INIT_RHO = 1 #0.001
rho = np.full((Xn+2, Yn+2), INIT_RHO, dtype=np.float64)

# Simulation parameters
R = D / 2  # Radius of the pipe

iteration = 0
iterations = []

list_avg_velocities_x = {}
list_avg_velocities_y = {}

start = time.perf_counter()
epsilon_cutoff = 10e-5
tau_f = 1.5 #1
tau_g = 1.
a=1
b=6.7
T=3.5e-2

#Kf = 0.5 * n_dx**2
Kf = 0.0001 * n_dx**2  # Reduce by 5x from current value
#Kg = 2.5e-4 * n_dx**2
Kg = 2.5e-5 * n_dx**2

Sh = U_nd/Cs
#used in Inamuro as scaling factor
Sh = 1.0

rho_G = 1
rho_L = 50
phi_star_G = 1.5e-2
phi_star_L = 9.2e-2
mu_G = 1.6e-4*n_dx
mu_L = 8.0e-3*n_dx

#inclination and force
alpha = 0.0
g = 9.81
alpha_rad = np.radians(alpha)
g_x = g * np.sin(alpha_rad)
g_y = -g * np.cos(alpha_rad)
F_body = np.array([g_x, g_y])
F_lattice1 = F_body * (n_dt**2 / n_dx)
F_lattice = F_body / CF

#initial conditions
y0 = (Yn-1)/2
#xi = 0.75
xi = 2.0  # Increase for smoother transition (test 4, 8, 12)
x,y = np.meshgrid(np.arange(Xn+2),np.arange(Yn+2),indexing='ij')


#phi initialisation
# Corrected order parameter: phi decreases from phi_star_L to phi_star_G as y > y0
_phi = (phi_star_L + phi_star_G) / 2 + (phi_star_L - phi_star_G) / 2 * erf((y0 - y) / (np.sqrt(2) * xi))

# Example usage during initialization
# Replace the original _phi initialization with a call to the method
_phi = initialize_phi_line(Xn+2, Yn+2, phi_star_G, phi_star_L, height=(Yn+1)/2, xi=2.0)
density_profile_x_position = Xn//2
density_profile_y_position = Yn//2

if PHI_NONLINEAR:
    # Example usage during initialization
    # Replace the original _phi initialization with a call to the method
    _phi = initialize_nonlinear_phi(Xn+2, Yn+2, phi_star_G, phi_star_L, xi=2.0) 
    density_profile_x_position = int(2/3*Xn)
    density_profile_y_position = int(2/3*Yn)   


h0 = np.zeros((9,Xn+2, Yn+2),dtype=np.float64)
h = np.zeros((9, Xn+2, Yn+2), dtype=np.float64)
_p0 = np.zeros((Xn+2, Yn+2),dtype=np.float64)
_fi_c = np.zeros((9,Xn+2, Yn+2),dtype=np.float64)
_fi = np.zeros((9,Xn+2, Yn+2),dtype=np.float64)
# Before the main loop
_fi = fi_c(np.zeros_like(u_ckl), Kf, F, _phi)
_gi_c = np.zeros((9, Xn+2, Yn+2),dtype=np.float64)
_gi = np.zeros((9, Xn+2, Yn+2),dtype=np.float64)


rho, mu = density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L)

#Bootstrap h and _p0 to equilibrium (prevents ph divergence on iter 0)
# FIXED: Bootstrap h and _p0 to equilibrium (prevents ph divergence on iter 0)
p_eq = rho / 3.0 # Lattice p_eq = rho * Cs^2 = rho / 3
_p0 = p_eq.copy()  # Initial _p0 = p_eq for ph loop

initial_p_check = np.sum(h[:], axis=0)
print(f"Initial h_eq check: sum h[1:] mean={np.mean(initial_p_check):.3f}, should = p_eq mean={np.mean(p_eq):.3f}")


ADD_METRICS = False
ADD_METRICS_PRINT = False

if ADD_METRICS:
    u_x_bounds = []
    u_y_bounds = []

rho_bounds = []
Invariants = []
AuxFields = []
MomentumBounds = []
GrowthMetric_uckl_x = []
GrowthMetric_uckl_y = []

GrowthMetric_uckl_star_y = []

if ADD_METRICS:
    GrowthMetric_grad_p_y = []
    GrowthMetric_forcing_term_y = []

GrowthMetric_gi_y_contribution = []
GrowthMetric_du_dv_div_u = []    

GrowthMetric_gi_div_sigma_x = []
GrowthMetric_gi_div_sigma_y = []

if ADD_METRICS:
    GrowthMetric_bounceBackTopBottom2_u_ckl_bottom = []
    GrowthMetric_bounceBackTopBottom2_u_ckl_top = []
    GrowthMetric_bounceBackTopBottom2_gi_up = []
    GrowthMetric_bounceBackTopBottom2_gi_down = []

GrowthMetric_gi_c_term2 = []
GrowthMetric_gi_c_term6_Gab = []

DivU_max = []
PhEps_max = []
PhIters = []

# --- Compact Iteration Snapshots (Directly Using TOTAL_ITERATION) ---
no_slices = 11
slice_iters = np.linspace(0, TOTAL_ITERATION - 1, no_slices, dtype=int).tolist()
# Remove duplicates if any (e.g., small TOTAL_ITERATION)
slice_iters = sorted(set(slice_iters))
density_slices = []

rho_min = np.min(rho)
rho_max = np.max(rho)
title = "Density map"
density_map_standalone(rho, rho_min, rho_max, title, iteration)
save_phi_snapshot(_phi, iteration, phi_star_G, phi_star_L)

u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2)]
epsilon_u_ckl = 0
epsilon_u_ckl_list = []

while iteration < TOTAL_ITERATION:
    if iteration % 100 == 0:
        print(f"Iter {iteration}: phi min={np.min(_phi):.3e}, max={np.max(_phi):.3e}")

    #Inamuro §2.3 Algorithm of computation:
    #Step 1. Using eqs (1) and (2), compute (fi(x, t+n_dt) and g(x, t+n_dt), and then compute phi(x, t+n_dt) and _u(x, t+n_dt)= with eqs (4) and (5).
    #Also rho(x, t+n_dt) is calculated with eq (4)
    if iteration == 0:
        u_zero = np.zeros_like(u_ckl)
        _fi_c = fi_c(u_zero, Kf, F, _phi)
        print(f"Iter 0: _fi_c with u=0 (no advection)")
    else:
        _fi_c = fi_c(u_ckl, Kf, F, _phi)
    if RAISE_NaN_ERROR and np.any(np.isnan(_fi_c)):
        raise ValueError(f"Iter {iteration}: NaN in _fi_c")
    if ADD_METRICS_PRINT: print(f"Iter {iteration}: fi_c min={np.min(_fi_c):.3e}, max={np.max(_fi_c):.3e}")        

    #Inamuro eq(2): calculation of the order parameter which distiguishes the two phases
    _fi = fi(_fi, _fi_c, tau_f)
    if RAISE_NaN_ERROR and np.any(np.isnan(_fi)):
        raise ValueError(f"Iter {iteration}: NaN in _fi")
    if ADD_METRICS_PRINT: print(f"Iter {iteration}: fi min={np.min(_fi):.3e}, max={np.max(_fi):.3e}")

    #Calculation of order parameter to distiguish the 2 phases
    _phi = phi(_fi)
    # In the main loop, after _phi = phi(_fi)
    if ADD_METRICS_PRINT: print(f"Iter {iteration}: phi at y=0: {np.mean(_phi[:,1]):.3e}, y=50: {np.mean(_phi[:,50]):.3e}, y=51: {np.mean(_phi[:,51]):.3e}")    
    if RAISE_NaN_ERROR and np.any(np.isnan(_phi)):
        raise ValueError(f"Iter {iteration}: NaN in _phi")
    if RAISE_LESS_THAN_ZERO_ERROR and (np.any(_phi <= 0) or np.any(_phi >= 1/b)):
        invalid_idx = np.where((_phi <= 0) | (_phi >= 1/b))
        raise ValueError(
            f"Iter {iteration}: Invalid _phi: min={np.min(_phi):.3e}, "
            f"max={np.max(_phi):.3e}, invalid indices: {invalid_idx}"
        )    
        
    if iteration in slice_iters:
        save_phi_snapshot(_phi, iteration, phi_star_G, phi_star_L)

    #Inamuro eq(3): calculation of the predicted velocity of the two phase fluid        
    if iteration > 0:
        _gi_c = gi_c(u_ckl, rho, tau_g, Kg, iteration)
    if RAISE_NaN_ERROR and np.any(np.isnan(_gi_c)):
        raise ValueError(f"Iter {iteration}: NaN in _gi_c")        

    rho, mu = density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L)
    if RAISE_NaN_ERROR and np.any(np.isnan(rho)):
        raise ValueError(f"Iter {iteration}: NaN in rho")
    if ADD_METRICS_PRINT: print(f"Iter {iteration}: rho min={np.min(rho):.3e}, max={np.max(rho):.3e}")    

    ################### 3. Clip and Smooth Rho/Mu/Phi #####################################################
    rho = np.clip(rho, 0.5, 75)  # Limit ratio effective 1:150, avoid tau_h extremes
    mu = np.clip(mu, mu_G*0.5, mu_L*2)
    _phi = np.clip(_phi, phi_star_G*1.1, phi_star_L*0.9)  # Confine to stable bulk
    # Smooth interfaces if Gab spikes
    from scipy.ndimage import gaussian_filter
    _phi = gaussian_filter(_phi, sigma=0.5)  # Diffuse sharp steps slightly
    ########################################################################################################

    if iteration in slice_iters:
        rho_min = np.min(rho)
        rho_max = np.max(rho)
        title = "Density map"
        density_map_standalone(rho, rho_min, rho_max, title, iteration)
        rho_slice = rho[density_profile_x_position, :].copy()
        density_slices.append((iteration, rho_slice))

    _gi = gi(_gi, _gi_c, u_ckl, rho, mu, iteration) 
    if RAISE_NaN_ERROR and np.any(np.isnan(_gi)):
        raise ValueError(f"Iter {iteration}: NaN in _gi")    
    #_gi = giExt(_gi, _gi_c, u_ckl, rho, mu) 


    '''
    #update ghost nodes
    # Before bounceBackTopBottom2
    # In the main loop, before bounceBackTopBottom2
    for i in [2, 5, 6]:  # Upward directions (c[i,1] > 0)
        _fi[i, :, 0] = _fi[i+2, :, 1]  # Mirror to opposite (2->4, 5->7, 6->8)
        _gi[i, :, 0] = _gi[i+2, :, 1]
    for i in [4, 7, 8]:  # Downward directions (c[i,1] < 0)
        _fi[i, :, Yn+1] = _fi[i-2, :, Yn]  # Mirror to opposite (4->2, 7->5, 8->6)
        _gi[i, :, Yn+1] = _gi[i-2, :, Yn]
    for i in [0, 1, 3]:  # Zero y-velocity directions
        _fi[i, :, 0] = _fi[i, :, 1]
        _fi[i, :, Yn+1] = _fi[i, :, Yn]
        _gi[i, :, 0] = _gi[i, :, 1]
        _gi[i, :, Yn+1] = _gi[i, :, Yn]
    '''
    update_ghost_nodes_top_bottom(_fi, _gi, Yn)

    #=> here the boundary conditions
    #Bounce-Back Top and Bottom
    _fi = bounceBackTopBottom2(_fi, Xn, Yn)
    _gi = bounceBackTopBottom2(_gi, Xn, Yn)

    #4.1b. assign inlet boundary values -> B)
    apply_periodic_boundary_conditions(_fi, _gi)        


    #Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    #Inamuro eq(5): Compute u(x,t+n_dt)
    #Kürger et al, p. 241 eq. (6.29) & Table 6.1
    #1. Shan-Chen - A=tau*n_dt
    A = n_dt*n_dt

    forcing_term = A*force_(F_lattice, rho)*n_dt
    u_ckl_star = np.einsum('ia,ijk->ajk', c, _gi) + forcing_term * ADD_FORCING_TERM

    #Step2a. calculate h, p
    '''
    epsilon0 = epsilon_cutoff * 10.0
    epsilon = np.full_like((Xn+2, Yn+2), epsilon0, dtype=np.float64)
    ph_iter = 0
    max_ph_iters = 100
    while np.all(epsilon > epsilon_cutoff) and ph_iter < max_ph_iters:
        p, h = ph(h, rho, u_ckl_star)
        if p.shape != _p0.shape:
            print(f"Warning: p shape {p.shape} != _p0 {_p0.shape} at iter {iteration}")
        epsilon = np.abs(p-_p0) / rho
        ph_iter += 1
    '''
    epsilon0 = epsilon_cutoff * 10.0
    epsilon = np.full_like(rho, epsilon0)  # Reuse shape from rho (assumes matches (Xn+2, Yn+2))

    ############################ 2. Add Successive Over-Relaxation (SOR) with Residual Monitoring (Accelerate/Damp) #################
    # Outside loop
    ph_iter = 0
    max_ph_iters = 500  # Generous but cap to avoid infinite
    omega_sor = 1.5  # 1.2-1.8 optimal for LBM Poisson; start 1.5
    # Inside main while
    while np.max(epsilon) > epsilon_cutoff and ph_iter < max_ph_iters:
        p_old = _p0.copy()
        p, h = ph(h, rho, u_ckl_star, iteration)  # Use damped div_u if added above
        delta_p = p - p_old
        p = p_old + omega_sor * delta_p  # Over-relax
        epsilon = np.abs(delta_p) / np.maximum(rho, 0.1)  # Avoid /small rho
        _p0 = p
        ph_iter += 1
        if ph_iter % 100 == 0:
            print(f"Global {iteration} ph_iter {ph_iter}: max_eps={np.max(epsilon):.3e}")
    if np.max(epsilon) > epsilon_cutoff:
        print(f"Warning: ph stalled at global {iteration}, eps={np.max(epsilon):.3e}, using approx p")
        # Fallback: smooth p to break stall
        from scipy.ndimage import gaussian_filter
        _p0 = gaussian_filter(_p0, sigma=1.0)  # Light smoothing 
    #################################################################################################################################

    #Inamuro eq(22 & 24): assign resultant p to _p0 for next iteration
    _p0 = p

    #Step 3: Compute u(x,t+n_dt) using eq. (20)
    #Inamuro eq(20): corrected current velocity u which satisfies the continuity equation div.u=0
    u_ckl = -gradient_p(p)*n_dt/(rho*Sh) + u_ckl_star
    
    u_ckl_x_min = np.min(u_ckl[0])
    u_ckl_x_max = np.max(np.abs(u_ckl[0]))
    u_ckl_y_min = np.min(u_ckl[1])
    u_ckl_y_max = np.max(np.abs(u_ckl[1]))
    # store tuple
    # In main loop, after u_ckl update:total_mom_x = np.sum(rho * u_ckl[0])  # Add this
    if iteration == 0 or iteration==(TOTAL_ITERATION - 1) or iteration % 100 == 0:
        invariant = np.sum(rho*u_ckl[0])
        MomentumBounds.append((iteration, u_ckl_x_min, u_ckl_x_max, invariant))
        if ADD_METRICS: u_x_bounds.append((iteration, u_ckl_x_min, u_ckl_x_max))
        if ADD_METRICS: u_y_bounds.append((iteration, u_ckl_y_min, u_ckl_y_max))
        rho_min = np.min(rho)
        rho_max = np.max(rho)    
        rho_bounds.append((iteration, rho_min, rho_max))
        Invariants.append((iteration, invariant))
        #max_abs_u_ckl_x = np.max(np.abs(u_ckl[0]))
        #max_abs_u_ckl_y = np.max(np.abs(u_ckl[1]))
        if ADD_METRICS: print(f"Iteration {iteration}: max|u_x|={u_ckl_x_max:.2e}, invariant={invariant:.2e}")    
        GrowthMetric_uckl_x.append((iteration, u_ckl_x_max))
        GrowthMetric_uckl_y.append((iteration, u_ckl_y_max))

        uckl_star_y = np.max(np.abs(u_ckl_star[1]))
        GrowthMetric_uckl_star_y.append((iteration, uckl_star_y))
        if ADD_METRICS: grad_p_y = np.max(np.abs(gradient_p(p)[1]))
        if ADD_METRICS: GrowthMetric_grad_p_y.append((iteration, grad_p_y))
        if ADD_METRICS: forcing_term_y = np.max(np.abs(forcing_term[1]))
        if ADD_METRICS: GrowthMetric_forcing_term_y.append((iteration, forcing_term_y))
        du_dx, du_dy = c_first_derivative(u_ckl_star[0])
        dv_dx, dv_dy = c_first_derivative(u_ckl_star[1])
        div_u_raw = du_dx + dv_dy
    
        GrowthMetric_du_dv_div_u.append((iteration, 
            np.max(np.abs(du_dx)), 
            np.max(np.abs(du_dy)), 
            np.max(np.abs(dv_dx)), 
            np.max(np.abs(dv_dy)), 
            np.max(np.abs(div_u_raw)))) 
            
        gi_y_contribution = np.einsum('i,ijk->jk', c[:,1], _gi)
        GrowthMetric_gi_y_contribution.append((iteration, np.max(np.abs(gi_y_contribution))))

        if ADD_METRICS: 
            item = np.max(np.abs(u_ckl[1, :, 1]))
            GrowthMetric_bounceBackTopBottom2_u_ckl_bottom.append((iteration, item))
            item = np.max(np.abs(u_ckl[1, :, Yn]))    
            GrowthMetric_bounceBackTopBottom2_u_ckl_top.append((iteration, item))

        if ADD_METRICS: 
            item = np.max(np.abs(_gi[2, :, 1]))        
            GrowthMetric_bounceBackTopBottom2_gi_up.append((iteration, item))
            item = np.max(np.abs(_gi[4, :, 1]))    
            GrowthMetric_bounceBackTopBottom2_gi_down.append((iteration, item))

        spuriousField1 = np.max(np.abs(np.gradient(p)))
        laplacian_phi = c_second_derivative(_phi)
        spuriousField2 = np.max(np.abs(laplacian_phi))
        AuxFields.append((iteration, spuriousField1, spuriousField2))
        PhIters.append((iteration, ph_iter))

        ################################# 6. Additional Diagnostics and Rollbacks #######################################################
        DivU_max.append((iteration, np.max(np.abs(div_u_raw))))
        PhEps_max.append((iteration, np.max(epsilon)))
        ##################################################################################################################################

    #streaming has commenced

    # Get the maximum density and its location
    max_density = np.max(rho)
    max_location = np.unravel_index(np.argmax(rho), rho.shape)
    if VERBOSE_MAX_RHO:
        if np.any(rho > 1):
            print(f"Instability detected at iteration {iteration + 1}")
        print(f"Maximum density: {max_density} at location {max_location}")


    # Update plots and parameters
    _rho_full_range = rho
    list_avg_velocities_x[iteration] = u_ckl[0, 1:-1, :]
    list_avg_velocities_y[iteration] = u_ckl[1, 1:-1, :]

    if iteration == 0 or iteration==(TOTAL_ITERATION - 1) or iteration % 100 == 0:
        epsilon_u_ckl = np.abs(u_ckl[0,int(Xn/2),int(Yn/2)] - u_ckl_midpoint0)
        epsilon_u_ckl_list.append((iteration, epsilon_u_ckl))    
        u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2)]

    #Step 4: re-iterate

    iteration += 1
    if iteration % 100 == 0:
        print(f"Simulation Execution -> TOTAL_ITERATION: {TOTAL_ITERATION}; iteration: {iteration}; {((iteration/TOTAL_ITERATION)*100.0):.1f} %")


end = time.perf_counter()
iterationsOfInterest = [0, 10, 50, 100, 200, 500, 1000, 5000, 10000, 12000]
diff = end - start
rho_in, rho_out, cs_2 = _rho_full_range[1, Yn // 2], _rho_full_range[Xn, Yn // 2], 1/3
dp = (rho_in - rho_out) * cs_2
rho_min = np.min(_rho_full_range)
rho_max = np.max(_rho_full_range)
print(f"Debug: _rho_full_range min = {rho_min:.6f}, max = {rho_max:.6f}")
paneLabel = f"LB: 2D Poiseuille with pressure difference; Lattice [{Xn},{Yn}]; Single processor"

filtered_u_ckl_dict_x = filter_u_ckl_fullrange(list_avg_velocities_x, iterationsOfInterest)
filtered_u_ckl_list_x = list(filtered_u_ckl_dict_x.values())

filtered_u_ckl_dict_y = filter_u_ckl_fullrange(list_avg_velocities_y, iterationsOfInterest)
filtered_u_ckl_list_y = list(filtered_u_ckl_dict_y.values())

# height ratios based on lattice dimensions
aspect_ratio = Xn / Yn
top_row_height = 1 #1.5
bottom_row_height = 1 #max(top_row_height * (Yn / Xn) * 2, 0.5)
# Define height_ratios for 3 rows
height_ratios = [top_row_height, bottom_row_height, bottom_row_height] if 'top_row_height' in globals() else [1, 1, 1]


# 3x2 multi-plot grid
fig1, ax1 = plt.subplots(
    3, 2,
    figsize=(15, 10),
    gridspec_kw={
        'width_ratios': [2, 4],  # Narrower first column, wider second column
        'height_ratios': height_ratios,
        'left': 0.15, 'right': 0.85, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)


# Row 0, Col 0: Amplitude plot (now in narrower column, width 2)
sectionPosition = int(Xn/2)
U_max_x = np.max(filtered_u_ckl_list_x[-1][sectionPosition, 1:Yn+1])
#amplitude_plot(ax1, u_full_range, listIterations, axis, xlabel, ylabel, title, nx=Xn, ny=Yn, angle=45, font_size=10):
amplitude_plot(ax1[0, 0], filtered_u_ckl_dict_x, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_x$", f"Amplitude u$_x$ at x={Xn}", sectionPosition, Yn)
amplitude_plot(ax1[1, 0], filtered_u_ckl_dict_y, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_y$", f"Amplitude u$_y$ at x={Xn}", sectionPosition, Yn)

# Row 0, Col 1: Velocity map (now in wider column, width 4)
_iteration = -1
velocity_map(ax1[0, 1], filtered_u_ckl_list_x[-1][1:-1, 1:Yn+1], _iteration, "Velocity [u$_x$] map") # "velocity_map-u_ckl_list_x_")
velocity_map(ax1[1, 1], filtered_u_ckl_list_y[-1][1:-1, 1:Yn+1], _iteration, "Velocity [u$_y$] map") # "velocity_map-u_ckl_list_y_")

# Row 1, Col 0: Density profile (now in narrower column, width 2)
#density_profile(ax[2, 0], _rho_full_range, Xn, Yn, iteration)
#profile position
#density_profile_transition(ax1[2, 0], _rho_full_range, density_profile_x_position, density_profile_y_position, Xn, Yn, 20, iteration)
density_profiles(ax1[2, 0], density_slices, density_profile_x_position,  Xn, Yn)
#density_mapExt(ax[1, 1], _rho_full_range, rho_min, rho_max, "density_map", iteration)

# Row 1, Col 1: Density map (now in wider column, width 4)
if PRESSURE_IN_DENSITY_MAP:
    min_value = 0 #np.min(_rho_full_range)
    _pressure_full_range = (_rho_full_range - min_value) * Cs**2 
    _pressure_out = (rho_min - min_value) * Cs**2
    _pressure_in = (rho_max - min_value) * Cs**2
    title = "Pressure map"
    density_mapExt(ax1[2, 1], _pressure_full_range, _pressure_out, _pressure_in, title, iteration)
else:
    title = "Density map"
    density_mapExt(ax1[2, 1], _rho_full_range, rho_min, rho_max, title, iteration)

# Add figure text
text = f"Run-time: {diff:.1f} s; rho_in: {rho_in:.6f}; rho_out: {rho_out:.6f}; dp: {dp:.6f}"
fig1.text(0.5, 0.98, text, ha='center', va='top', fontsize=12)

# Adjust layout
fig1.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

# Save the figure
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "Milestone-Images")
os.makedirs(images_dir, exist_ok=True)
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_channel_parameters.png")
fig1.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved 3x2 grid: {save_path}")
plt.show(block=True)


# 3 rows, 4 columns
fig2, ax2 = plt.subplots(
    3, 4,  # 3 rows, 4 columns
    figsize=(18, 10),  # Wider figure to fit 4 columns nicely
    gridspec_kw={
        'width_ratios': [1, 1, 1, 1],  # Equal widths for all 4 columns
        'height_ratios': height_ratios,  # Keep your height_ratios
        'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

plot_bounds_ext(GrowthMetric_uckl_x, "GrowthMetric_uckl_x", ax2[0, 0])
plot_bounds_ext(GrowthMetric_uckl_y, "GrowthMetric_uckl_y", ax2[0, 1])
plot_bounds_ext(GrowthMetric_uckl_star_y, "GrowthMetric_uckl_star_y", ax2[0, 2])

plot_bounds_ext(rho_bounds, "rho_bounds", ax2[0, 3])
#plot_bounds_ext(AuxFields, "AuxFields", ax2[1, 1])
plot_bounds_ext(epsilon_u_ckl_list, "epsilon_u_ckl growth", ax2[1, 0])
plot_bounds_ext(Invariants, "Invariants", ax2[1, 1])
plot_momentum_bounds(MomentumBounds, "MomentumBounds", ax2[1, 2])
plot_bounds_ext(PhIters, "PhIters", ax2[1, 3])    

if False:
    
    plot_bounds(u_x_bounds, "u_ckl_x_bounds", Kf)
    plot_bounds(u_y_bounds, "u_ckl_y_bounds", Kf)
    plot_bounds_ext(GrowthMetric_grad_p_y, "GrowthMetric_grad_p_y")

    plot_bounds_ext(GrowthMetric_grad_p_y, "GrowthMetric_grad_p_y")
    plot_bounds_ext(GrowthMetric_forcing_term_y, "GrowthMetric_forcing_term_y")
    plot_bounds_ext(GrowthMetric_gi_y_contribution, "GrowthMetric_gi_y_contribution")

    plot_bounds_ext(GrowthMetric_gi_c_term6_Gab, "GrowthMetric_gi_c_term6_Gab")
    plot_bounds_ext(GrowthMetric_gi_c_term2, "GrowthMetric_gi_c_term2")

    series_labels = ["du_dx","du_dy","dv_dx","dv_dy","div_u"]
    plot_bounds_ext(GrowthMetric_du_dv_div_u, "GrowthMetric_du_dv_div_u", series_labels, ax2[1, 3])    

    plot_bounds_ext(GrowthMetric_bounceBackTopBottom2_u_ckl_bottom, "GrowthMetric_bounceBackTopBottom2_u_ckl_bottom")
    plot_bounds_ext(GrowthMetric_bounceBackTopBottom2_u_ckl_top, "GrowthMetric_bounceBackTopBottom2_u_ckl_top")

    plot_bounds_ext(GrowthMetric_bounceBackTopBottom2_gi_up, "GrowthMetric_bounceBackTopBottom2_gi_up")
    plot_bounds_ext(GrowthMetric_bounceBackTopBottom2_gi_down, "GrowthMetric_bounceBackTopBottom2_gi_down")    

plot_bounds_ext(GrowthMetric_gi_div_sigma_x, "GrowthMetric_gi_div_sigma_x", ax2[2, 0])
plot_bounds_ext(GrowthMetric_gi_div_sigma_y, "GrowthMetric_gi_div_sigma_y", ax2[2, 1])   

plot_bounds_ext(DivU_max, "DivU_max", ax2[2, 2])
plot_bounds_ext(PhEps_max, "PhEps_max", ax2[2, 3])


plt.tight_layout()
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_channel_metrics.png")
fig2.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved 3x2 grid: {save_path}")
plt.show(block=True)plt.show(block=True)