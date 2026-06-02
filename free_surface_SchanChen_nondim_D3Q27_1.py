#this simulation a microchannel flow is modelled along a plane inserted vertically and symetrically in the channel center#aligned to the z-axis vertically and in the x-axis direction along hte channel length
#the axes in the plane are y-axis in the vertical direction and x-axis in the channel horizontal direction
#The simulation is based on the paper by Inamuro et al, 2003, Journal of Computational Physics 198
# Add at top of script (Python 3.7+ for forward refs in annotations)
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

import json
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Optional, Tuple  # Import these for proper type hints

import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from github_uploader import GitHubUploader
from lbm_plotter import LBMPlotter
from profiler_class import Profiler
from vtk_data_dump import VTKDataDumper

profiler = Profiler()

#flags
RAISE_LESS_THAN_ZERO_ERROR = False
RAISE_NaN_ERROR = False
PRESSURE_IN_DENSITY_MAP = False
ADD_FORCING_TERM = 1
TOTAL_ITERATIONS = 4000 #12001 * 4
FILENAME_PADDING_WIDTH = int(np.ceil(np.log10(TOTAL_ITERATIONS + 1)))
NO_DATA_DUMP_SLICES = 51  # 51 slices for 3D model
PLOTREALTIME = False  
ADD_METRICS = False 
ADD_METRICS_PRINT = False
UPLOAD_TO_GITHUB = False
DUMP_TO_VTK = False
MOTION_TERM = False

# --- Simulation use-cases ---
USE_CASES = {
    "nonlinear": {"PHI_NONLINEAR": True, "alpha": 0.0},
    "linear": {"PHI_NONLINEAR": False, "alpha": 30.0},
}
ACTIVE_CASE = "nonlinear" # "nonlinear"   # or "linear"
PHI_NONLINEAR = USE_CASES[ACTIVE_CASE]["PHI_NONLINEAR"]
alpha = USE_CASES[ACTIVE_CASE]["alpha"]
USE_CASE_TAG = f"{ACTIVE_CASE}_a{alpha:g}"

# Constants
SCRIPT_FILENAME = os.path.splitext(os.path.basename(__file__))[0] 
SCRIPT_FULL_PATH = os.path.abspath(__file__) 
SCRIPTS_PATH = "scripts/freesurface/"
PLOTS_PATH = "results/freesurface/"  # GitHub path prefix
IMAGES_SUBDIR = "FreesurfaceImages"  # Local subdir
script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory

images_dir = os.path.join(script_dir, IMAGES_SUBDIR)
os.makedirs(images_dir, exist_ok=True)  # create folder if it doesn't exist  
LOG_FILE = 'lbm3D_debug.log'

######### logging ####################################################################################################
# Global debug level (set once at init, e.g., based on flags like VERBOSE1, ADD_METRICS_PRINT)
# 0: none (suppress all)
# 1: init (startup params only)
# 2: iter (iteration progress, e.g., %100 summaries)
# 3: fields (detailed field stats like min/max per component)
# Global DEBUG_LEVEL (unchanged)
# -------------------- Config --------------------
DEBUG_LEVEL = 0  # Or whatever; controls prefixed categories only

# Improved logging config: Root at WARNING to suppress most noise, logger at DEBUG
logging.getLogger().setLevel(logging.WARNING)  # Root: Silence everything by default

logger = logging.getLogger(LOG_FILE)
logger.setLevel(logging.DEBUG)  # logger: Always DEBUG internally

# Custom filter to enforce DEBUG_LEVEL (attached to logger)
# Custom filter to show only main loop progress
class DebugLevelFilter(logging.Filter):
    def filter(self, record):
        msg = record.msg
        # Allow only 'ITER' messages containing 'Simulation Execution'
        if DEBUG_LEVEL == 0 and 'ITER: Simulation Execution' in msg:
            return True
        return False

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(message)s'))  # Simplified format: show only message
console_handler.addFilter(DebugLevelFilter())
logger.addHandler(console_handler)

# File handler (optional, for full debug logs if needed)
file_handler = logging.FileHandler('lbm_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
logger.addHandler(file_handler)  # No filter, logs everything to file

# Silence noisy external loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('kiwisolver').setLevel(logging.WARNING)

# Debug log function (unchanged)
def debug_log(category: str, message: str, *args: Any, **kwargs: Any) -> None:
    prefixed_msg = f"{category}: {message}"
    log_func = logger.debug
    if category == 'WARN':
        log_func = logger.warning
    elif category == 'ERROR':
        log_func = logger.error
    log_func(prefixed_msg, *args, **kwargs)


def validate_field(field: np.ndarray, name: str, iter: Optional[int] = None, 
                  allow_neg: bool = False, allow_range: Optional[Tuple[float, float]] = None) -> None:
    """
    Centralized validator for NaN, negatives, and custom ranges.
    Logs via debug_log('ERROR', ...) before raising.
    
    Args:
        field: np.ndarray to check.
        name: Field name (e.g., '_phi').
        iter: Optional iteration for context (use -1 for non-iter calls like init).
        allow_neg: If False, flags negatives.
        allow_range: (min_excl, max_excl) open interval; e.g., (0, 1/b) for _phi.
    """
    if np.any(np.isnan(field)):
        min_val, max_val = np.min(field), np.max(field)  # Still compute for context
        debug_log('ERROR', 'NaN in {name} at iter {iter}: min={min:.3e}, max={max:.3e}', 
                  name=name, iter=iter if iter is not None else 'N/A', min=min_val, max=max_val)
        raise ValueError(f'NaN in {name}')
    
    if not allow_neg and np.any(field < 0):
        min_val = np.min(field)
        debug_log('ERROR', 'Negative in {name} at iter {iter}: min={min:.3e}', 
                  name=name, iter=iter if iter is not None else 'N/A', min=min_val)
        raise ValueError(f'Negative in {name}')
    
    if allow_range:
        min_req, max_req = allow_range
        invalid = (field <= min_req) | (field >= max_req)
        if np.any(invalid):
            min_val, max_val = np.min(field), np.max(field)
            invalid_idx = np.where(invalid)
            debug_log('ERROR', 'Out-of-range {name} at iter {iter}: min={min:.3e}, max={max:.3e}, '
                      'must be in ({min_req:.3e}, {max_req:.3e}). Invalid indices: {idx}', 
                      name=name, iter=iter if iter is not None else 'N/A', min=min_val, max=max_val, 
                      min_req=min_req, max_req=max_req, idx=invalid_idx)
            raise ValueError(
                f"Invalid {name}: min={min_val:.3e}, max={max_val:.3e}, "
                f"must be in ({min_req:.3e}, {max_req:.3e}). Invalid indices: {invalid_idx}"
            ) 
######################################################################################################################

Cs=np.sqrt(1/3)
D=1e-3 #m
L=1 #m

D_nd=50 #100

Yn=int(D_nd) #+1
Xn=200 #int(Yn*L/D)
Zn=int(D_nd)

dx=D/D_nd #old->5*10**(-5)
dy = dx
dz = dx
#relaxation time n_tau, should be > 0,5
n_tau = 0.6

dP=0 #Pa
rho_0=1e3 #kg/m^3
dRho=dP/Cs**2

nu=2.9e-6 #m^2/s => in OLB this is 1/Re, with Re=148. So Re must become Re= in order to conform with this simulation
dt = Cs**2*(n_tau-0.5)*(dx**2/nu)

debug_log('INIT', 'Yn={Yn}, Xn={Xn}, dx={dx:.3e}, dy={dy:.3e}, dt={dt:.3e}')

#Assume U at centerline (max) velocity
U=1.0
Re=D*U/nu
Ma=U/Cs 
Kn=U*D/nu
debug_log('INIT', 'U=%(U).2f, Re=%(Re).2f, Ma=%(Ma).3e, Kn=%(Kn).3e', extra=dict(U=U, Re=Re, Ma=Ma, Kn=Kn))

#we need Cl, Crho, Ct
# 1. Conversion factor Cl for length
Cl = dx #freely chosen
n_dx = dx/Cl #-> dx_nd=1
n_dy = n_dx
n_dz = n_dx
debug_log('INIT', 'Cl=%(Cl).2f, n_dx=%(n_dx).2f',  extra=dict(Cl=Cl, n_dx=n_dx))

#2. Conversion factor Crho for density
Crho = rho_0
rho_nd = rho_0/Crho #-> rho_nd=1
debug_log('INIT', 'Crho=%(Crho).2f, rho_nd=%(rho_nd).2f', extra=dict(Crho=Crho, rho_nd=rho_nd))

#3. Conversion factor Ct for time
Ct=dt
n_dt = dt/Ct #-> dt_nd=1
debug_log('INIT', 'Ct=%(Ct).2f, dt_nd=%(n_dt).2f', extra=dict(Ct=Ct, n_dt=n_dt))

#4. Conversion factor Cu for velocity
Cu=Cl/Ct
U_nd = U/Cu #-> limit U_nd=0.1
U_nd=0.1

debug_log('INIT', 'Cu=%(Cu).2f, dt_nd=%(U_nd).2f', extra=dict(Cu=Cu, U_nd=U_nd))

#5. Conversion factor CF for Force
CF=Crho*Cl/(Ct**2)
debug_log('INIT', 'CF=%(CF).2f', extra=dict(CF=CF))

#6. Conversion factor Cf for frequency
Cf=1/Ct
debug_log('INIT', 'CF=%(Cf).2f', extra=dict(Cf=Cf))

#change nu_nd in order to achieve U_nd=0,1
nu_nd=((D_nd*U_nd)/(D*U))*nu
debug_log('INIT', 'nu_nd=%(nu_nd).2f', extra=dict(nu_nd=nu_nd))


tau_nd=(nu_nd/Cs**2)+1./2
debug_log('INIT', 'tau_nd=%(tau_nd).2f', extra=dict(tau_nd=tau_nd))
omega = dt/n_tau
debug_log('INIT', 'omega=%(omega).2f', extra=dict(omega=omega))
omega_nd = n_dt/tau_nd
debug_log('INIT', 'omega_nd=%(omega_nd).2f', extra=dict(omega_nd=omega_nd))

#discrete velocity channels for D2Q26
channel_range_26 = 26
channel_range_27 = 27
c = np.array([
    # i=0, rest particle
    [ 0,  0,  0],   
    # 6 axis-aligned
    [ 1,  0,  0],   # i=1
    [-1,  0,  0],   # i=2
    [ 0,  1,  0],   # i=3
    [ 0, -1,  0],   # i=4
    [ 0,  0,  1],   # i=5
    [ 0,  0, -1],   # i=6
    # 12 edge-aligned
    [ 1,  1,  0],   # i=7
    [-1,  1,  0],   # i=8
    [-1, -1,  0],   # i=9
    [ 1, -1,  0],   # i=10
    [ 1,  0,  1],   # i=11
    [-1,  0,  1],   # i=12
    [-1,  0, -1],   # i=13
    [ 1,  0, -1],   # i=14
    [ 0,  1,  1],   # i=15
    [ 0, -1,  1],   # i=16
    [ 0, -1, -1],   # i=17
    [ 0,  1, -1],   # i=18
    # 8 corner-aligned
    [ 1,  1,  1],   # i=19
    [-1,  1,  1],   # i=20
    [-1, -1,  1],   # i=21
    [ 1, -1,  1],   # i=22
    [ 1,  1, -1],   # i=23
    [-1,  1, -1],   # i=24
    [-1, -1, -1],   # i=25
    [ 1, -1, -1]    # i=26
])

#Inamuro eq(8): constant E & #Krüger: force weights -> expand to D3Q27
# E: main weights
E = np.array([
    8/27,        # i=0 rest particle
    *(2/27 for _ in range(6)),  # 6 axis-aligned
    *(1/54 for _ in range(12)),  # 12 edge-aligned
    *(1/216 for _ in range(8))  # 8 corner-aligned
])

# H: only rest particle has 1
H = np.array([
    1,  # i=0
] + [0]*channel_range_26)  # i=1..26

# F: force weights, F0 = -5/3, others = 3*E_i
F = 3*E
F[0] = -5/3

# Precompute constants (in global scope, near c definition, line ~290)
H_exp = H[:, np.newaxis, np.newaxis, np.newaxis]  # (27,1,1,1)
F_exp = F[:, np.newaxis, np.newaxis, np.newaxis]  # (27,1,1,1)
# Equilibrium weights for h_i
E_exp = E[:, np.newaxis, np.newaxis, np.newaxis]  # Shape (27, 1, 1, 1) for broadcasting
neigh_c = c[1:channel_range_27]  # (26, 3)

debug_log('INIT', 'c=%(c).2f', extra=dict(c=c))

#Inamuro eq(4): particle velocity distribution
def phi(_f):
    __phi = np.sum(_f, axis=0)

    return __phi


#Inamuro eq(11): bulk free-energy density
def psi(_phi, iteration=None):
    """
    Inamuro eq(11): Compute bulk free-energy density for 3D D3Q27.
    _phi: Order parameter (numpy array, shape (Xn+2, Yn+2, Zn+2)).
    Returns: psi (same shape as _phi).
    Raises: ValueError if _phi is out of valid range (0, 1/b).
    """

    profiler.start("psi")  

    # Check for valid phi range
    validate_field(_phi, '_phi', iter=iteration, allow_range=(0, 1/b))
    _psi = _phi * T * np.log(_phi / (1 - b * _phi)) - a * _phi**2

    profiler.stop("psi")  

    return _psi


#Inamuro eq(10): p0 from eq(6)F[i]*
def p0(_phi):
    """
    Inamuro eq(10): Compute p0 from free-energy for 3D D3Q27.
    _phi: Order parameter (numpy array, shape (Xn+2, Yn+2, Zn+2)).
    Returns: p0 (same shape as _phi).
    """    
    profiler.start("p0")  

    _p0 = ((_phi * T) / (1 - b * _phi)) - a * _phi**2

    profiler.stop("p0")  

    return _p0


#Inamuro eq(24): pressure
def gradient_p(p):
    """
    Compute pressure gradient for 3D D3Q27 using c_first_derivative with bounce-back at y=0, Yn+1, z=0, Zn+1.
    p: Pressure field (shape (Xn+2, Yn+2, Zn+2)).
    n_dx, n_dy, n_dz: Grid spacings.
    Returns: Gradient (shape (3, Xn+2, Yn+2, Zn+2)), [dp/dx, dp/dy, dp/dz].
    """

    profiler.start("gradient_p")  

    if np.any(np.isnan(p)):
        raise ValueError(f"NaN in pressure field p: min={np.min(p):.3e}, max={np.max(p):.3e}")
    
    dp_dx, dp_dy, dp_dz = c_first_derivative(p, n_dx, n_dy, n_dz)
    
    # Apply bounce-back for dp_dy at y=0, Yn+1
    dp_dy[:, 0, :] = 0.0  # No y-gradient at bottom wall
    dp_dy[:, Yn+1, :] = 0.0  # No y-gradient at top wall
    
    # Apply bounce-back for dp_dz at z=0, Zn+1
    dp_dz[:, :, 0] = 0.0  # No z-gradient at bottom wall
    dp_dz[:, :, Zn+1] = 0.0  # No z-gradient at top wall

    profiler.stop("gradient_p")  
    
    return np.stack([dp_dx, dp_dy, dp_dz], axis=0)


def c_first_derivative(lamda, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Fast central differences for 3D scalar field.
    Computes derivatives only for interior points (avoids np.roll).
    """
    profiler.start("c_first_derivative")

    dlamda_dx = np.zeros_like(lamda)
    dlamda_dy = np.zeros_like(lamda)
    dlamda_dz = np.zeros_like(lamda)

    # interior points only
    dlamda_dx[1:-1,:,:] = (lamda[2:,:,:] - lamda[:-2,:,:]) / (2 * n_dx)
    dlamda_dy[:,1:-1,:] = (lamda[:,2:,:] - lamda[:,:-2,:]) / (2 * n_dy)
    dlamda_dz[:,:,1:-1] = (lamda[:,:,2:] - lamda[:,:,:-2]) / (2 * n_dz)

    profiler.stop("c_first_derivative")

    return dlamda_dx, dlamda_dy, dlamda_dz


#Inamuro eq(12): first derivatives - partial dphi/dx_a, du_b/dx_a, drho/dx_a
def c_first_derivative_0(lamda, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Compute first derivatives of a 3D scalar field using D3Q27 lattice directions.
    lamda: Scalar field (shape (nx, ny, nz), e.g., (202, 52, 52)).
    n_dx, n_dy, n_dz: Grid spacings.
    Returns: (dlamda_dx, dlamda_dy, dlamda_dz), each shape (nx, ny, nz).
    """
    if len(lamda.shape) != 3:
        raise ValueError(f"Expected lamda shape (nx, ny, nz) like (202, 52, 52), got {lamda.shape}")
    
    profiler.start("c_first_derivative")

    # Lattice speed of sound squared for D3Q27
    cs2 = 1./3
    scale_x = 1. / (cs2 * n_dx)
    scale_y = 1. / (cs2 * n_dy)
    scale_z = 1. / (cs2 * n_dz)

    neigh_c_x = c[1:channel_range_27, 0]  # x-components of neighbor velocities
    neigh_c_y = c[1:channel_range_27, 1]  # y-components
    neigh_c_z = c[1:channel_range_27, 2]  # z-components

    roll_axes = (0, 1, 2)  # 3D axes: x, y, z

    # Shift lamda in all 26 neighbor directions
    shifted_neighbors = [
        np.roll(lamda, shift=(c[k+1, 0], c[k+1, 1], c[k+1, 2]), axis=roll_axes)
        for k in range(channel_range_26)
    ]
    neighbors_stack = np.stack(shifted_neighbors, axis=0)  # Shape (26, nx, ny, nz)

    # Compute derivatives using weighted sum over neighbors
    dlamda_dx = np.einsum('h,hijk->ijk', neigh_c_x, neighbors_stack) * scale_x
    dlamda_dy = np.einsum('h,hijk->ijk', neigh_c_y, neighbors_stack) * scale_y
    dlamda_dz = np.einsum('h,hijk->ijk', neigh_c_z, neighbors_stack) * scale_z

    profiler.stop("c_first_derivative")

    return dlamda_dx, dlamda_dy, dlamda_dz


#Inamuro eq(13): second derivatives partial - partial dlambda²/dx_a²
def c_second_derivative(lamda, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Vectorized version of eq(13) in Inamuro for 3D D3Q27 - second derivatives.
    Compute ∂²λ/∂x², ∂²λ/∂y², ∂²λ/∂z² using weighted sum over pre-shifted neighbors.
    lamda: Scalar field (shape (Xn+2, Yn+2, Zn+2)).
    n_dx, n_dy, n_dz: Grid spacings.
    Returns: (d2lamda_dx2, d2lamda_dy2, d2lamda_dz2) (each same shape as lamda).
    """
    if len(lamda.shape) != 3:
        raise ValueError(f"Expected lamda shape (nx, ny, nz) like (202, 52, 52), got {lamda.shape}")
    
    profiler.start("c_second_derivative")    

    # Lattice speed of sound squared for D3Q27
    cs2 = 1/3
    scale_x = 2 / (cs2 * n_dx**2)
    scale_y = 2 / (cs2 * n_dy**2)
    scale_z = 2 / (cs2 * n_dz**2)

    # Non-zero velocities (i=1 to 26) and weights
    neigh_E = E[1:27]  # Weights for i=1 to 26
    sum_w = np.sum(neigh_E)  # Sum of weights = 19/27    

    roll_axes = (0, 1, 2)

    shifted_neighbors = [
        np.roll(lamda, shift=(c[k+1, 0], c[k+1, 1], c[k+1, 2]), axis=roll_axes)
        for k in range(channel_range_26)
    ]
    neighbors_stack = np.stack(shifted_neighbors, axis=0)  # Shape (26, nx, ny, nz)

    # Weighted sum over neighbors
    sum_neighbors = np.einsum('h,hijk->ijk', neigh_E, neighbors_stack)

    # Apply formula: scale * (sum - sum_w * lamda)
    d2lamda_dx2 = scale_x * (sum_neighbors - sum_w * lamda)
    d2lamda_dy2 = scale_y * (sum_neighbors - sum_w * lamda)
    d2lamda_dz2 = scale_z * (sum_neighbors - sum_w * lamda)

    profiler.stop("c_second_derivative")    

    return d2lamda_dx2, d2lamda_dy2, d2lamda_dz2


#Inamuro eq(9): Gab - shear terms
def Gab(_phi):
    profiler.start("Gab")    

    dphi_x, dphi_y, dphi_z = c_first_derivative(_phi)

    Nx, Ny, Nz = _phi.shape
    G = np.zeros((Nx, Ny, Nz, 3, 3))

    # |grad phi|^2
    mag2 = dphi_x**2 + dphi_y**2 + dphi_z**2

    # (9/2) * dphi_a * dphi_b
    G[:, :, :, 0, 0] = (9/2) * dphi_x * dphi_x - (3/2) * mag2
    G[:, :, :, 0, 1] = (9/2) * dphi_x * dphi_y
    G[:, :, :, 0, 2] = (9/2) * dphi_x * dphi_z

    G[:, :, :, 1, 0] = (9/2) * dphi_y * dphi_x
    G[:, :, :, 1, 1] = (9/2) * dphi_y * dphi_y - (3/2) * mag2
    G[:, :, :, 1, 2] = (9/2) * dphi_y * dphi_z

    G[:, :, :, 2, 0] = (9/2) * dphi_z * dphi_x
    G[:, :, :, 2, 1] = (9/2) * dphi_z * dphi_y
    G[:, :, :, 2, 2] = (9/2) * dphi_z * dphi_z - (3/2) * mag2

    profiler.stop("Gab")   

    return G


#Inamuro eq(14 & 15): density rho and viscosity mu
def density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L):
    profiler.start("density_and_viscosity")   

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

    profiler.stop("density_and_viscosity")   

    return _rho, _mu


#Inamuro eq(23): relaxation time tau_h
def tau_h(_rho):
    """
    Inamuro eq(23): Compute relaxation time tau_h for 3D D3Q27.
    _rho: Density field (shape (Xn+2, Yn+2, Zn+2)).
    Returns: _tau_h (same shape as _rho).
    """    
    _tau_h = 1/_rho + 1./2

    return _tau_h


#support function for g[i] and ph()
def vectorized_stream_old(neigh_collision, neigh_c):
    """
    Vectorized streaming for multiple directions.
    neigh_collision: (N, nx, ny, nz) - N directions (excluding rest)
    neigh_c: (N, 3) - shifts along x, y, z for each direction
    Returns streamed array of same shape.
    """
    N, nx, ny, nz = neigh_collision.shape
    # Create indices for x, y, z axes
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    
    # Broadcast and add shifts per direction
    x_idx = (x + neigh_c[:, 0][:, None, None, None]) % nx  # shape (N, nx, 1, 1)
    y_idx = (y + neigh_c[:, 1][:, None, None, None]) % ny  # shape (N, 1, ny, 1)
    z_idx = (z + neigh_c[:, 2][:, None, None, None]) % nz  # shape (N, 1, 1, nz)
    
    # Advanced indexing to fetch rolled arrays in batch
    streamed = neigh_collision[
        np.arange(N)[:, None, None, None],
        x_idx,
        y_idx,
        z_idx
    ]
    return streamed


#Inamuro eq(22,24): evolution equation of the velocity distribution function h(i) and pressure
def ph(hn, _rho, u_ckl_star, iteration, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Inamuro eq(22,24): Vectorized evolution for h_i in 3D D3Q27.
    hn: Pressure distribution (shape (27, Xn+2, Yn+2, Zn+2)).
    _rho: Density field (shape (Xn+2, Yn+2, Zn+2)).
    u_ckl_star: Velocity field (shape (3, Xn+2, Yn+2, Zn+2)).
    iteration: Current iteration for debugging.
    n_dx, n_dy, n_dz: Grid spacings.
    Returns: (_p_new, hn_plus1).
    """

    profiler.start("ph")   

    # -----------------------------
    # 1. Compute pressure from hn
    # -----------------------------
    _p = np.sum(hn, axis=0)  # (nx, ny, nz)

    # Equilibrium distribution
    h_eq = E_exp * _p  # Broadcast to (27, nx, ny, nz)

    # BGK relaxation
    tau_h = 1.0 / _rho + 0.5  # (nx, ny, nz)
    omega = 1.0 / tau_h        # (nx, ny, nz), will broadcast automatically

    # -----------------------------
    # 2. Compute divergence term
    # -----------------------------
    du_dx, _, _ = c_first_derivative(u_ckl_star[0], n_dx, n_dy, n_dz)
    _, dv_dy, _ = c_first_derivative(u_ckl_star[1], n_dx, n_dy, n_dz)
    _, _, dw_dz = c_first_derivative(u_ckl_star[2], n_dx, n_dy, n_dz)

    div_u_raw = du_dx + dv_dy + dw_dz
    max_div_u = np.max(np.abs(div_u_raw))
    if iteration % 100 == 0:
        debug_log('PH', 'Iter %d ph_start: max|div_u_raw|=%.3e', iteration, max_div_u)

    # Normalize if too large
    div_u = div_u_raw / (1.0 + max_div_u) if max_div_u > 1.0 else div_u_raw

    # -----------------------------
    # 3. Enforce zero at boundaries
    # -----------------------------
    div_u[0, :, :] = div_u[-1, :, :] = 0.0
    div_u[:, 0, :] = div_u[:, -1, :] = 0.0
    div_u[:, :, 0] = div_u[:, :, -1] = 0.0

    # Expand div_u for broadcasting
    div_u_exp = div_u[np.newaxis, :, :, :]  # (1, nx, ny, nz)
    #forcing_term = Sh * (1.0 / 3.0) * E_exp * div_u_exp * n_dx
    # In D2Q9, the coefficient 1/3 comes from the lattice moment analysis for 2D. 
    # It ensures that when you sum over all directions, the contribution of 
    # ∇⋅𝑢∗∇⋅u∗ gives the correct pressure change.
    #forcing = (1.0/3.0) * E[:, None, None, None] * div_u    
    # change 1/3 -> 1/9 since     
    forcing_term = Sh * (1.0 / 9.0) * E_exp * div_u_exp * n_dx

    # -----------------------------
    # 4. Collision step
    # -----------------------------
    collision = hn - omega * (hn - h_eq) - forcing_term  # broadcasting omega

    # -----------------------------
    # 5. Streaming step
    # -----------------------------
    hn_plus1 = np.empty_like(hn)
    hn_plus1[0] = collision[0]  # rest particle (no streaming)

    neigh_collision = collision[1:27]  # moving directions
    neigh_c = c[1:27]                  # shifts for each direction

    USE_OLD_STREAMING = False

    if USE_OLD_STREAMING:
        # ->this is too slow
        # Triple np.roll for each direction
        streamed_neighbors = [
            np.roll(
                np.roll(
                    np.roll(neigh_collision[i], shift=neigh_c[i, 0], axis=0),
                    shift=neigh_c[i, 1], axis=1
                ),
                shift=neigh_c[i, 2], axis=2
            )
            for i in range(26)
        ]
        hn_plus1[1:27] = np.stack(streamed_neighbors, axis=0)
    else:
        #try this
        # --- ULTRA-FAST & CORRECT STREAMING (transpose trick) ---
        shift_x = neigh_c[:, 0]  # (26,)
        shift_y = neigh_c[:, 1]  # (26,)
        shift_z = neigh_c[:, 2]  # (26,)

        # Move population index to end: (nx, ny, nz, 26)
        streamed = np.moveaxis(neigh_collision, 0, -1)

        # Now roll in spatial dimensions (axis 0,1,2), shift broadcasts over last axis
        streamed = np.roll(streamed, shift=shift_x, axis=0)  # X
        streamed = np.roll(streamed, shift=shift_y, axis=1)  # Y
        streamed = np.roll(streamed, shift=shift_z, axis=2)  # Z

        # Move population index back to front: (26, nx, ny, nz)
        streamed = np.moveaxis(streamed, -1, 0)

        hn_plus1[1:27] = streamed


    # -----------------------------
    # 6. Update pressure
    # -----------------------------
    _p_new = np.sum(hn_plus1, axis=0)

    profiler.stop("ph")   

    return _p_new, hn_plus1


def velocity_gradient(u_ckl, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Compute ∂u_beta/∂x_alpha for a 2D velocity field.
    Returns array with shape (2, 2, nx, ny).
    """
    u_x, u_y, u_z = u_ckl  # unpack

    # Compute derivatives (np.gradient returns [∂/∂x0, ∂/∂x1])
    du_x_dx, du_x_dy, du_x_dz = np.gradient(u_x, n_dx, n_dy, n_dz, edge_order=3)
    du_y_dx, du_y_dy, du_y_dz = np.gradient(u_y, n_dx, n_dy, n_dz, edge_order=3)
    du_z_dx, du_z_dy, du_z_dz = np.gradient(u_z, n_dx, n_dy, n_dz, edge_order=3)

    # Stack properly into 4D tensor (β, α, nx, ny)
    du_b_dx_a = np.stack([
        np.stack([du_x_dx, du_x_dy, du_x_dz], axis=0),   # β = 0
        np.stack([du_y_dx, du_y_dy, du_y_dz], axis=0),   # β = 1
        np.stack([du_z_dx, du_z_dy, du_z_dz], axis=0)    # β = 2
    ], axis=0)

    return du_b_dx_a


#Inamuro eq(3): calculation of the predicted velocity of the two phase fluid
def gi(_gi, _gi_c, u_ckl, rho, mu, iteration):
    """
    Vectorized Inamuro eq(3): predicted velocity evolution for gi in two-phase LBM.
    Full batch operations on (27, nx, ny, nz) - assumes padded grids (e.g., 202x52x52).
    Collision via broadcasting, streaming via batched rolls.
    """

    profiler.start("gi")   

    # 1. Velocity gradients on 3D grid
    grads = [c_first_derivative(u_ckl[i], n_dx, n_dy, n_dz) for i in range(3)]
    du_x_dx, du_x_dy, du_x_dz = grads[0]
    du_y_dx, du_y_dy, du_y_dz = grads[1]
    du_z_dx, du_z_dy, du_z_dz = grads[2]

    # 2. Strain rate tensor S_ab = du_a/dx_b + du_b/dx_a
    S_xx = 2 * du_x_dx
    S_yy = 2 * du_y_dy
    S_zz = 2 * du_z_dz
    S_xy = du_y_dx + du_x_dy
    S_xz = du_x_dz + du_z_dx
    S_yz = du_y_dz + du_z_dy

    # 3. Stress sigma = mu * S
    sigma_xx = mu * S_xx
    sigma_yy = mu * S_yy
    sigma_zz = mu * S_zz
    sigma_xy = mu * S_xy
    sigma_xz = mu * S_xz
    sigma_yz = mu * S_yz

    # 4. Divergence of stress: div_sigma_a = ∂_b σ_ab
    div_sigma_x = (np.roll(sigma_xx, -1, axis=0) - np.roll(sigma_xx, 1, axis=0)) / (2 * n_dx) + \
                  (np.roll(sigma_xy, -1, axis=1) - np.roll(sigma_xy, 1, axis=1)) / (2 * n_dy) + \
                  (np.roll(sigma_xz, -1, axis=2) - np.roll(sigma_xz, 1, axis=2)) / (2 * n_dz)
    
    div_sigma_y = (np.roll(sigma_xy, -1, axis=0) - np.roll(sigma_xy, 1, axis=0)) / (2 * n_dx) + \
                  (np.roll(sigma_yy, -1, axis=1) - np.roll(sigma_yy, 1, axis=1)) / (2 * n_dy) + \
                  (np.roll(sigma_yz, -1, axis=2) - np.roll(sigma_yz, 1, axis=2)) / (2 * n_dz)

    div_sigma_z = (np.roll(sigma_xz, -1, axis=0) - np.roll(sigma_xz, 1, axis=0)) / (2 * n_dx) + \
                  (np.roll(sigma_yz, -1, axis=1) - np.roll(sigma_yz, 1, axis=1)) / (2 * n_dy) + \
                  (np.roll(sigma_zz, -1, axis=2) - np.roll(sigma_zz, 1, axis=2)) / (2 * n_dz)

    # 5. Expand for viscous term broadcasting
    div_sigma_x_exp = div_sigma_x[np.newaxis, :, :, :]  # (1, nx+2, ny+2, nz+2)
    div_sigma_y_exp = div_sigma_y[np.newaxis, :, :, :]
    div_sigma_z_exp = div_sigma_z[np.newaxis, :, :, :]
    rho_exp = rho[np.newaxis, :, :, :]  # (1, nx+2, ny+2, nz+2)
    c_x_exp = c[:, 0][:, np.newaxis, np.newaxis, np.newaxis]  # (27, 1, 1, 1)
    c_y_exp = c[:, 1][:, np.newaxis, np.newaxis, np.newaxis]
    c_z_exp = c[:, 2][:, np.newaxis, np.newaxis, np.newaxis]

    # 6. Viscous term: 3 * E_i * c_a[i] * div_sigma_a / rho
    dot_term = c_x_exp * (div_sigma_x_exp / rho_exp) + \
               c_y_exp * (div_sigma_y_exp / rho_exp) + \
               c_z_exp * (div_sigma_z_exp / rho_exp)
    viscous_term = 3.0 * E_exp * dot_term * n_dx  # (27, nx+2, ny+2, nz+2)

    # 7. Collision: BGK + viscous forcing
    _gi_star = _gi  - (1.0 / tau_g) * (_gi - _gi_c) + viscous_term

    # 8. Streaming: vectorized rolls for all directions
    streamed = [
        np.roll(_gi_star[i], shift=(c[i, 0], c[i, 1], c[i, 2]), axis=(0, 1, 2)) for i in range(27)
    ]
    _gi[:] = np.stack(streamed, axis=0)  # In-place overwrite

    profiler.stop("gi")   

    return _gi


# Force density Fg formulation acc. Krüger et al. §6.1
def force_(F_lattice, rho):
    _force = F_lattice[:, None, None, None]* rho 

    return _force


#Inamuro eq(7): calculation of predicted velocity of the two phase fluid - collision term
def gi_c(u, rho, tau_g, Kg, iteration):
    """
    Vectorized gi_c equilibrium (Inamuro) on padded grid (27, Xn+2, Yn+2, Zn+2).
    Matches original loop exactly: all ops on padded shapes, no slicing.
    Assumes inputs u (3, Xn+2, Yn+2, Zn+2), rho (Xn+2, Yn+2, Zn+2), _phi (Xn+2, Yn+2, Zn+2) are padded.
    Globals: c(27,2), E(27), F(27), n_dx, Gab func.
    """

    profiler.start("gi_c")   

    # Validate inputs
    if tau_g != globals()['tau_g'] or Kg != globals()['Kg']:
        raise ValueError("Input tau_g, Kg must match global values")
    if _phi is None:
        raise ValueError("Global _phi must be defined for Gab")
    
    # Padded shapes
    nx_pad, ny_pad, nz_pad  = rho.shape  # e.g., (202,52,52)
    
    # Shared terms
    grad_rho_x, grad_rho_y, grad_rho_z = c_first_derivative(rho, n_dx, n_dy, n_dz)  # (nx_pad, ny_pad, nz_pad)
    u_dot_u = np.sum(u**2, axis=0)  # (nx_pad, ny_pad, nz_pad)
    
    # Velocity gradients
    du_a_dx, du_a_dy, du_a_dz = c_first_derivative(u[0], n_dx, n_dy, n_dz)
    du_b_dx, du_b_dy, du_b_dz = c_first_derivative(u[1], n_dx, n_dy, n_dz)
    du_g_dx, du_g_dy, du_g_dz = c_first_derivative(u[2], n_dx, n_dy, n_dz)
    
    # Gab on full
    Gab_phi = Gab(_phi)  # (nx_pad, ny_pad, nz_pad, 3, 3)
    
    # Expansions for i-broadcast
    ones = np.ones((nx_pad, ny_pad, nz_pad))
    
    # term 1: E_i * 1
    term1 = E_exp * ones  # (27,nx, ny, nz)
    
    # term2: E[i] * 3 * (c_i . u)
    c_dot_u = np.einsum('ia,axyz->ixyz', c, u)  # (27,nx,ny,nz)
    term2 = E_exp * 3.0 * c_dot_u
    
    # term3: E[i] * (3/2) * u_dot_u
    term3 = E_exp * (3.0 / 2.0) * u_dot_u
    
    # term4: E[i] * (9/2) * (c_dot_u)^2
    c_dot_u_tensor = c_dot_u ** 2  
    term4 = E_exp * (9.0 / 2.0) * c_dot_u_tensor
    
    # term5: E[i] * (3/2) * (tau_g - 1/2) * n_dx * velocity_gradient_term
    c_x = c[:, 0][:, np.newaxis, np.newaxis, np.newaxis]  # (27,1,1,1)
    c_y = c[:, 1][:, np.newaxis, np.newaxis, np.newaxis]
    c_z = c[:, 2][:, np.newaxis, np.newaxis, np.newaxis]
    
    # Strain rate terms: (du_b/dx_a + du_a/dx_b) * c_i,a * c_i,b
    term_xxx = 2.0 * du_a_dx * c_x**2  
    term_yyy = 2.0 * du_b_dy * c_y**2
    term_zzz = 2.0 * du_g_dz * c_z**2    
    term_xy = (du_a_dy + du_b_dx) * c_x * c_x
    term_xz = (du_a_dz + du_g_dx) * c_y * c_y
    term_yz = (du_b_dz + du_g_dy) * c_z * c_z

    velocity_gradient_term = term_xxx  + term_yyy  + term_zzz  + term_xy + term_xz + term_yz  # (27,nx,ny,nz)
    term5 = E_exp * (3.0 / 2.0) * (tau_g - 0.5) * n_dx * velocity_gradient_term
    
    # term6: E[i] * (Kg/rho) * (c_i . Gab . c_i)
    _Gab = np.einsum('ia,ib,xyzab->ixyz', c, c, Gab_phi)  # (27,nx,ny,nz)
    term6 = E_exp * (Kg / rho) * _Gab
    #GrowthMetric_gi_c_term6_Gab.append((iteration, abs_term6))
    
    # term7: (2/3) * F_i * (Kg/rho) * (grad_rho)^2
    grad_rho_sq = grad_rho_x**2 + grad_rho_y**2 + grad_rho_z**2  # Sum over all directions
    term7 = (2.0 / 3.0) * F_exp * (Kg / rho) * grad_rho_sq

    # Assemble
    _gi_c = term1 + term2 - term3 + term4 + term5 + term6 - term7  # (27,nx_pad,ny_pad,nz_pad)

    profiler.stop("gi_c")   
    
    return _gi_c


#Inamuro eq(2): calculation of the order parameter which distiguishes the two phases
def fi(_fi, _fi_c, tau_f):
    """
    Collision and streaming for fi distribution (D3Q27).
    No Python loops anywhere - negative debug uses vectorized argmin/where.
    Assumes globals: RAISE_LESS_THAN_ZERO_ERROR, iteration, c(27,2).
    """

    profiler.start("fi")   

    # Validate input
    if tau_f != globals()['tau_f']:
        raise ValueError("Input tau_f must match global value")

    # Collision:  BGK
    omega_f = 1.0 / tau_f
    _fi_star = _fi - omega_f * (_fi - _fi_c)   # Broadcast all

    # Vectorized negative check on _fi_star
    if RAISE_LESS_THAN_ZERO_ERROR:
        neg_mask = _fi_star < 0  # (27,nx+2,ny+2,nz+2)
        if np.any(neg_mask):
            # Find directions with negatives
            neg_dirs = np.where(np.any(neg_mask, axis=(1,2,3)))[0] 
            for i in neg_dirs:
                debug_log('WARN', 'Iter %d: Negative _fi_star[%d] at z=50: min=%.3e', iteration, i, np.min(_fi_star[i, :, :, 50]))

    # Streaming: batched rolls
    streamed = [
        np.roll(_fi_star[i], shift=(c[i, 0], c[i, 1], c[i, 2]), axis=(0, 1, 2)) for i in range(channel_range_27)
    ]
    _fi[:] = np.stack(streamed, axis=0)

    # Post-streaming check
    if RAISE_LESS_THAN_ZERO_ERROR and np.any(_fi < 0):
        raise ValueError(f"Iter {iteration}: Negative _fi: min={np.min(_fi):.3e}")
    
    profiler.stop("fi")       

    return _fi


#support function for fi_c
def c_laplacian(lamda, n_dx=1.0, n_dy=1.0, n_dz=1.0):
    """
    Inamuro eq(13): Compute ∇²λ = ∂²λ/∂x² + ∂²λ/∂y² + ∂²λ/∂z² for 3D D3Q27.
    lamda: Scalar field (shape (Xn+2, Yn+2, Zn+2)).
    n_dx, n_dy, n_dz: Grid spacings.
    Returns: Laplacian (same shape as lamda).
    """

    profiler.start("c_laplacian")  

    d2lamda_dx2, d2lamda_dy2, d2lamda_dz2 = c_second_derivative(lamda, n_dx, n_dy, n_dz)
    laplacian = d2lamda_dx2 + d2lamda_dy2 + d2lamda_dz2

    profiler.stop("c_laplacian")  

    return laplacian


#Inamuro eq(6): calculation of the order parameter which distiguishes the two phases - collision term
def fi_c(u, Kf, F, _phi, C=0.0):
    """
    Inamuro eq(6): Equilibrium distribution f_i^eq for 3D D3Q27.
    u: Velocity field (shape (3, Xn+2, Yn+2, Zn+2)).
    Kf: Gradient parameter (scalar).
    F: Weights (array of length 27).
    _phi: Order parameter (shape (Xn+2, Yn+2, Zn+2)).
    Returns: _fi_c (shape (27, Xn+2, Yn+2, Zn+2)).
    """

    profiler.start("fi_c")   

    # Validate inputs
    if Kf != globals()['Kf']:
        raise ValueError("Input Kf must match global value")
    if not np.array_equal(F, globals()['F']):
        raise ValueError("Input F must match global F")
    if _phi is None:
        raise ValueError("Global _phi must be defined")
    
    # Padded shapes
    nx_pad, ny_pad, nz_pad = _phi.shape    
    
    # Shared terms
    dphi_dx, dphi_dy, dphi_dz = c_first_derivative(_phi, n_dx, n_dy, n_dz)
    if RAISE_NaN_ERROR and (np.any(np.isnan(dphi_dx)) or np.any(np.isnan(dphi_dy)) or np.any(np.isnan(dphi_dz))):
        raise ValueError("NaN in dphi_dx, dphi_dy, or dphi_dz")

    laplacian_phi = c_laplacian(_phi, n_dx, n_dy, n_dz)
    if RAISE_NaN_ERROR and np.any(np.isnan(laplacian_phi)):
        raise ValueError("NaN in laplacian_phi")

    G = Gab(_phi)  # (nx_pad, ny_pad, nz_pad, 3, 3)
    if RAISE_NaN_ERROR and np.any(np.isnan(G)):
        raise ValueError("NaN in Gab")

    p0_val = p0(_phi)  # (nx_pad, ny_pad, nz_pad)
    if RAISE_NaN_ERROR and np.any(np.isnan(p0_val)):
        raise ValueError("NaN in p0")
    
    # term1: H[i] * _phi
    term1 = H_exp * _phi  # (27, nx_pad, ny_pad, nz_pad)
    
    # term2: F[i] * p0_val
    term2 = F_exp * p0_val
    
    # term3: F[i] * Kf * _phi * laplacian_phi
    term3 = F_exp * Kf * _phi * laplacian_phi
    
    # term4: F_i * (Kf/6) * (dphi_dx^2 + dphi_dy^2 + dphi_dz^2)
    grad_phi_sq = dphi_dx**2 + dphi_dy**2 + dphi_dz**2
    term4 = F_exp * (Kf / 6.0) * grad_phi_sq
    
    # term5: 3 * E[i] * _phi * c_dot_u
    c_dot_u = np.einsum('ia,axyz->ixyz', c, u)  # (27, nx_pad, ny_pad, nz_pad) - c_i . u
    term5 = 3.0 * E_exp * _phi * c_dot_u
    
    # term6: E[i] * Kf * _Gab (c^T G c)
    _Gab = np.einsum('ia,ib,xyzab->ixyz', c, c, G)  # (27, nx_pad, ny_pad, nz_pad)
    term6 = E_exp * Kf * _Gab
    
    # Assemble _fi_c
    _fi_c = term1 + term2 - term3 - term4 + term5 + term6  # (27,nx_pad,ny_pad,nz_pad)

    # --------------------  <<<  ADD THE EXTRA TERM  >>> --------------------
    if MOTION_TERM and C != 0.0:
        _fi_c += mobility_term(_phi, Kf, p0_val, C)    
    
    # Negative check
    if RAISE_LESS_THAN_ZERO_ERROR:
        neg_mask = _fi_c < 0
        if np.any(neg_mask):
            neg_dirs = np.where(np.any(neg_mask, axis=(1, 2, 3)))[0]
            for i in neg_dirs:
                min_val = np.min(_fi_c[i, :, :, 50])
                t1_min = np.min(term1[i, :, :, 50])
                t2_min = np.min(term2[i, :, :, 50])
                t3_min = np.min(term3[i, :, :, 50])
                t4_min = np.min(term4[i, :, :, 50])
                t5_min = np.min(term5[i, :, :, 50])
                t6_min = np.min(term6[i, :, :, 50])
                debug_log('FIELD', 'Iter %d: _fi_c[%d] at z=50: min=%.3e, terms at z=50: t1=%.3e, t2=%.3e, t3=%.3e, t4=%.3e, t5=%.3e, t6=%.3e',
                          iteration, i, min_val, t1_min, t2_min, t3_min, t4_min, t5_min, t6_min)
            raise ValueError(f"Iter {iteration}: Negative _fi_c: min={np.min(_fi_c):.3e}")
        
    profiler.stop("fi_c")   
    
    return _fi_c


#auxiliary code for Press tensor
def grad(arr, dim):
    """Central difference: ∂arr/∂x_dim"""
    return (np.roll(arr, -1, axis=dim) - np.roll(arr, +1, axis=dim)) / (2.0 * dx)


def pressure_tensor(phi, Kf, p0_val):
    """
    Returns P_ab[α,β]  (α,β = 0,1,2)
    Summation over γ is performed inside.
    """
    profiler.start("pressure_tensor")  

    # ----- gradients of phi ------------------------------------------------
    dphi = np.stack([grad(phi, d) for d in range(3)], axis=0)   # shape (3,nx,ny,nz)

    # ----- Laplacian (sum of second derivatives) -------------------------
    lap = grad(dphi[0],0) + grad(dphi[1],1) + grad(dphi[2],2)

    # ----- |∇φ|² -----------------------------------------------------------
    grad2 = np.sum(dphi**2, axis=0)          # (nx,ny,nz)

    # ----- bulk part -------------------------------------------------------
    bulk = p0_val - Kf * phi * lap - (Kf/2.0) * grad2

    # ----- P_ab = bulk * δ_ab + Kf * (∂φ/∂x_a)(∂φ/∂x_b) --------------------
    P = np.zeros((3,3,*phi.shape))
    for a in range(3):
        P[a,a] = bulk                     # diagonal Kronecker part
        for b in range(3):
            if a != b:
                P[a,b] = 0.0
            P[a,b] += Kf * dphi[a] * dphi[b]   # off-diagonal + diagonal correction

    profiler.stop("pressure_tensor")  

    return P          # shape (3,3,nx,ny,nz)


def mobility_term(phi, Kf, p0_val, C):
    """
    Returns: (Q, nx, ny, nz)
    Adds: E[i] * C * (∂P_{αβ}/∂x_β) * c_{iα} * Δx  → reduces mobility
    """
    
    profiler.start("mobility_term")  

    # --- pressure tensor ---
    P = pressure_tensor(phi, Kf, p0_val)                # (3,3,nx,ny,nz)

    # --- divergence of P: ∂P_{αβ}/∂x_β ---
    divP = np.zeros((3, *phi.shape))
    for a in range(3):
        divP[a] = grad(P[a,0], 0) + grad(P[a,1], 1) + grad(P[a,2], 2)

    # --- build term for all 27 directions ---
    term = np.zeros((0, *phi.shape))
    for i in range(0):
        # dot product: c_i · divP
        scalar = (c[i,0] * divP[0] + 
                  c[i,1] * divP[1] + 
                  c[i,2] * divP[2])
        term[i] = E[i] * C * scalar * dx

    profiler.stop("mobility_term")          

    return term


def bounceBackTopBottom2(f):
    """
    Bounce-back at top/bottom y-walls and z-walls for D3Q27.
    f: Distribution function (shape (27, nx, ny, nz)).
    Returns: f with bounce-back applied.
    """

    profiler.start("bounceBackTopBottom2")  

    nz = f.shape[3]

    # Precompute opposite directions
    opposite = np.zeros(channel_range_27, dtype=int)
    for i in range(channel_range_27):
        for j in range(channel_range_27):
            if np.all(c[j] == -c[i]):
                opposite[i] = j
                break

    # Bottom y-wall (y=0)
    f[:, :, 0, :] = f[opposite, :, 1, :]

    # Top y-wall (y=ny-1)
    f[:, :, Yn-1, :] = f[opposite, :, Yn-2, :]

    # Bottom z-wall (z=0)
    f[:, :, :, 0] = f[opposite, :, :, 1]

    # Top z-wall (z=nz-1)
    f[:, :, :, nz-1] = f[opposite, :, :, nz-2]

    profiler.stop("bounceBackTopBottom2")  

    return f


def update_ghost_nodes_top_bottom(_fi, _gi, c):
    """
    Update ghost nodes at top/bottom y- and z-boundaries for D3Q27.
    _fi: Order parameter distribution (27, nx, ny, nz).
    _gi: Velocity distribution (27, nx, ny, nz).
    c: D3Q27 velocity vectors (27, 3).
    Returns: None (in-place modification).
    """

    profiler.start("update_ghost_nodes_top_bottom")  

    ny, nz = _fi.shape[2], _fi.shape[3]

    # Precompute opposite directions
    opp = np.zeros(channel_range_27, dtype=int)
    for i in range(channel_range_27):
        for j in range(channel_range_27):
            if np.all(c[i] == -c[j]):
                opp[i] = j
                break

    # Direction groups
    up_y_dirs = np.where(c[:, 1] > 0)[0]      # Moving upward in y
    down_y_dirs = np.where(c[:, 1] < 0)[0]    # Moving downward in y
    zero_y_dirs = np.where(c[:, 1] == 0)[0]   # Zero y-velocity
    up_z_dirs = np.where(c[:, 2] > 0)[0]      # Moving upward in z
    down_z_dirs = np.where(c[:, 2] < 0)[0]    # Moving downward in z
    zero_z_dirs = np.where(c[:, 2] == 0)[0]   # Zero z-velocity

    # Bottom y-ghost (y=0): reflect from y=1
    _fi[up_y_dirs, :, 0, :] = _fi[opp[up_y_dirs], :, 1, :]
    _gi[up_y_dirs, :, 0, :] = _gi[opp[up_y_dirs], :, 1, :]

    # Top y-ghost (y=ny-1): reflect from y=ny-2
    _fi[down_y_dirs, :, ny-1, :] = _fi[opp[down_y_dirs], :, ny-2, :]
    _gi[down_y_dirs, :, ny-1, :] = _gi[opp[down_y_dirs], :, ny-2, :]

    # Zero y-velocity directions: copy from nearest interior
    _fi[zero_y_dirs, :, 0, :] = _fi[zero_y_dirs, :, 1, :]
    _fi[zero_y_dirs, :, ny-1, :] = _fi[zero_y_dirs, :, ny-2, :]
    _gi[zero_y_dirs, :, 0, :] = _gi[zero_y_dirs, :, 1, :]
    _gi[zero_y_dirs, :, ny-1, :] = _gi[zero_y_dirs, :, ny-2, :]

    # Bottom z-ghost (z=0): reflect from z=1
    _fi[up_z_dirs, :, :, 0] = _fi[opp[up_z_dirs], :, :, 1]
    _gi[up_z_dirs, :, :, 0] = _gi[opp[up_z_dirs], :, :, 1]

    # Top z-ghost (z=nz-1): reflect from z=nz-2
    _fi[down_z_dirs, :, :, nz-1] = _fi[opp[down_z_dirs], :, :, nz-2]
    _gi[down_z_dirs, :, :, nz-1] = _gi[opp[down_z_dirs], :, :, nz-2]

    # Zero z-velocity directions: copy from nearest interior
    _fi[zero_z_dirs, :, :, 0] = _fi[zero_z_dirs, :, :, 1]
    _fi[zero_z_dirs, :, :, nz-1] = _fi[zero_z_dirs, :, :, nz-2]
    _gi[zero_z_dirs, :, :, 0] = _gi[zero_z_dirs, :, :, 1]
    _gi[zero_z_dirs, :, :, nz-1] = _gi[zero_z_dirs, :, :, nz-2]

    profiler.stop("update_ghost_nodes_top_bottom")  


def apply_periodic_boundary_conditions(_fi, _gi, Xn):
    """
    Apply periodic boundary conditions in x-direction for D3Q27.
    _fi, _gi: Distributions (shape (27, nx+2, ny+2, nz+2)).
    Xn: Interior grid size in x-direction.
    Returns: None (in-place modification).
    """

    profiler.start("apply_periodic_boundary_conditions")  

    # Left boundary x=0 ← copy from right interior x=Xn
    _fi[:, 0, :, :] = _fi[:, Xn, :, :]
    _gi[:, 0, :, :] = _gi[:, Xn, :, :]

    # Right boundary x=Xn+1 ← copy from left interior x=1
    _fi[:, Xn+1, :, :] = _fi[:, 1, :, :]
    _gi[:, Xn+1, :, :] = _gi[:, 1, :, :]

    profiler.stop("apply_periodic_boundary_conditions")  
    

def initialize_step_phi(xn, yn, zn, phi_star_g, phi_star_l, xi=2.0, smooth_sigma=None):
    """
    Initialize _phi for 3D domain using nonlinear step geometry (as before)
    and apply Gaussian smoothing to remove sharp corners/singularities.

    Args:
        xn, yn, zn : int
            Domain dimensions.
        phi_star_g, phi_star_l : float
            Gas and liquid phase phi values.
        xi : float
            Characteristic transition width (used in erf).
        smooth_sigma : float or None
            Gaussian smoothing sigma (grid units). If None, uses xi / 1.5 (recommended).
    Returns:
        _phi : np.ndarray (xn, yn, zn)
    """
    # --- coordinate grid ---
    x, y, z = np.meshgrid(np.arange(xn), np.arange(yn), np.arange(zn), indexing='ij')
    
    # --- key interface positions ---
    x_mid = 0.5 * (xn - 1)
    x_34  = 0.75 * (xn - 1)
    y_mid = 0.5 * (yn - 1)
    y_23  = (2.0 / 3.0) * (yn - 1)

    # --- base liquid/gas pattern (same logic as original nonlinear 3D init) ---
    _phi = np.where(
        ((x <= x_mid) & (y < y_mid)) |                     # Left block
        (((x > x_mid) & (x <= x_34)) & (y < y_23)) |       # Middle block
        ((x > x_34) & (y < y_mid)),                        # Right block
        phi_star_l, phi_star_g
    )

    # --- smooth erf transitions (same as before) ---
    phi_mid  = 0.5 * (phi_star_l + phi_star_g)
    phi_diff = 0.5 * (phi_star_l - phi_star_g)
    sigma    = np.sqrt(2.0) * xi
    trans_width = 3.0 * xi

    # y = y_mid transitions (left/right)
    erf_factor_y_mid = phi_diff * erf((y_mid - y) / sigma)
    mask_y_mid = (((x <= x_mid) | (x > x_34)) & (np.abs(y - y_mid) <= trans_width))
    _phi[mask_y_mid] = phi_mid + erf_factor_y_mid[mask_y_mid]

    # y = y_23 transition (middle)
    erf_factor_y_23 = phi_diff * erf((y_23 - y) / sigma)
    mask_y_23 = ((x > x_mid) & (x <= x_34) & (np.abs(y - y_23) <= trans_width))
    _phi[mask_y_23] = phi_mid + erf_factor_y_23[mask_y_23]

    # --- apply Gaussian smoothing in 3D to remove corner singularities ---
    if smooth_sigma is None:
        smooth_sigma = xi / 1.5   # empirically stable default
    _phi = gaussian_filter(_phi, sigma=smooth_sigma, mode='nearest')

    # --- enforce physical bounds ---
    _phi = np.clip(_phi, min(phi_star_g, phi_star_l), max(phi_star_g, phi_star_l))

    return _phi


def initialize_phi_line(xn, yn, zn, phi_star_g, phi_star_l, height=None, xi=2.0):
    """
    Initialize _phi along a line at height y in D3Q27, constant along z.
    Args:
        xn, yn, zn: Grid points (Xn+2, Yn+2, Zn+2).
        phi_star_g, phi_star_l: Gas and liquid order parameters.
        height: Transition height (default yn/2).
        xi: Transition width (default 2.0).
    Returns: _phi (xn, yn, zn).
    """
    y0 = (yn - 1) / 2 if height is None else height
    x, y, z = np.meshgrid(np.arange(xn), np.arange(yn), np.arange(zn), indexing='ij')
    _phi = np.where(y < y0, phi_star_l, phi_star_g)
    sigma = np.sqrt(2) * xi
    transition = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * erf((y0 - y) / sigma)
    mask = np.abs(y - y0) <= 3 * xi
    _phi[mask] = transition[mask]
    return _phi


def get_iterations_of_interest(total_iterations, no_slices=51, early_fraction=0.2, exp_factor=3.0):
    """
    Generate iteration indices for 3D reconstruction.
    Args:
        total_iterations: Total simulation iterations.
        no_slices: Number of iterations to sample (default 51).
        early_fraction: Fraction for early sampling (default 0.2).
        exp_factor: Exponential sampling factor (default 3.0).
    Returns: List of iteration indices.
    """
    if total_iterations <= 0 or no_slices <= 0:
        return []

    fixed = [0, 50, 100, 200, 500]
    early_end = int(total_iterations * early_fraction)
    n_rem = no_slices - len(fixed)
    
    early_dense = np.linspace(early_end//4, early_end, 15, dtype=int).tolist()
    mid_end = int(total_iterations * 0.7)
    exp_samples = 20
    exp = np.linspace(0, 1, exp_samples + 1)[1:]
    mid_samples = np.floor((np.exp(exp * exp_factor) - 1) / (np.exp(exp_factor) - 1) * 
                          (mid_end - early_end)).astype(int) + early_end
    mid_samples = mid_samples.tolist()
    late_samples = np.linspace(mid_end, total_iterations - 1, 16, dtype=int).tolist()
    
    all_iters = sorted(set(fixed + early_dense + mid_samples + late_samples))
    return all_iters[:no_slices]


#preliminary
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)

#initialise
#average velocity, cartesion x,y-directions, k is y-position, l is x-position
u_ckl = np.zeros((3, Xn+2, Yn+2, Zn+2), dtype=np.float64)
INIT_RHO = 1 #0.001
rho = np.full((Xn+2, Yn+2, Zn+2), INIT_RHO, dtype=np.float64)

# Simulation parameters
R = D / 2  # Radius of the pipe

iteration = 0

list_avg_velocities_x = {}
phi_3d_data = {}  # ← ADD THIS
list_avg_velocities_y = {}

start = time.perf_counter()
epsilon_threshold = 1e-5
tau_f = 2.5
tau_g = 1.
a=1
b=6.7
T=3.5e-2

#Kf = 0.0001 * n_dx**2  # Reduce by 5x from current value
#Kf = 1e-6 * n_dx**2  # Reduce by 5x from current value
Kf = 1e-6 * n_dx**2
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
g = 9.81
alpha_rad = np.radians(alpha)
g_x = g * np.sin(alpha_rad)
g_y = -g * np.cos(alpha_rad)
g_z = 0.
F_body = np.array([g_x, g_y, g_z])
F_lattice = F_body / CF
C    = 0.0          # start with 0, later increase to 0.1 … 1.0

#initial conditions
y0 = (Yn-1)/2
#xi = 0.75
xi = 2.0  # Increase for smoother transition (test 4, 8, 12)
x, y, z = np.meshgrid(np.arange(Xn+2), np.arange(Yn+2), np.arange(Zn+2), indexing='ij')

#phi initialisation
# Corrected order parameter: phi decreases from phi_star_L to phi_star_G as y > y0
_phi = (phi_star_L + phi_star_G) / 2 + (phi_star_L - phi_star_G) / 2 * erf((y0 - y) / (np.sqrt(2) * xi))

# Example usage during initialization
# Replace the original _phi initialization with a call to the method
_phi = initialize_phi_line(Xn+2, Yn+2, Zn+2, phi_star_G, phi_star_L, height=(Yn+1)/2, xi=2.0)
density_profile_x_position = Xn//2
density_profile_y_position = Yn//2
density_profile_z_position = Zn//2

if PHI_NONLINEAR:
    # Example usage during initialization
    # Replace the original _phi initialization with a call to the method
    _phi = initialize_step_phi(Xn+2, Yn+2, Zn+2, phi_star_G, phi_star_L, xi=2.0) 
    density_profile_x_position = int(2/3*Xn)
    density_profile_y_position = int(2/3*Yn)   
    density_profile_z_position = Zn//2

h0 = np.zeros((channel_range_27,Xn+2, Yn+2, Zn+2),dtype=np.float64)
h = np.zeros((channel_range_27, Xn+2, Yn+2, Zn+2), dtype=np.float64)
_p0 = np.zeros((Xn+2, Yn+2, Zn+2),dtype=np.float64)
_fi_c = np.zeros((channel_range_27,Xn+2, Yn+2, Zn+2),dtype=np.float64)
_fi = np.zeros((channel_range_27,Xn+2, Yn+2, Zn+2),dtype=np.float64)
# Before the main loop
_fi = fi_c(np.zeros_like(u_ckl), Kf, F, _phi)
_gi_c = np.zeros((channel_range_27, Xn+2, Yn+2, Zn+2),dtype=np.float64)
_gi = np.zeros((channel_range_27, Xn+2, Yn+2, Zn+2),dtype=np.float64)

rho, mu = density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L)

#Bootstrap h and _p0 to equilibrium (prevents ph divergence on iter 0)
# FIXED: Bootstrap h and _p0 to equilibrium (prevents ph divergence on iter 0)
_p0 = rho / 3.0 # Lattice p_eq = rho * Cs^2 = rho / 3;  Initial _p0 = p_eq for ph loop
debug_log('INIT', 'Initial h_eq check: sum h[1:] mean=%.3f, should = p_eq mean=%.3f', np.mean(np.sum(h[1:], axis=0)), np.mean(_p0))

ADD_METRICS = False
ADD_METRICS_PRINT = False

rho_bounds = []
Invariants = []
MomentumBounds = []
GrowthMetric_uckl_x = []
GrowthMetric_uckl_y = []

GrowthMetric_uckl_star_y = []

GrowthMetric_div_u_raw = []
GrowthMetric_u_ckl_star_du_dy = []

DivU_max = []
PhEps_max = []
PhIters = []
AuxFields = []

iterationsOfInterest = get_iterations_of_interest(TOTAL_ITERATIONS, no_slices=NO_DATA_DUMP_SLICES, early_fraction=0.3, exp_factor=4.0)
density_slices = []

u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2),int(Zn/2)]
epsilon_u_ckl = 0
epsilon_u_ckl_list = []


plotter = LBMPlotter(
    script_dir=script_dir,
    script_filename=SCRIPT_FILENAME,
    use_case_tag=USE_CASE_TAG,
    images_subdir=IMAGES_SUBDIR,
    total_iterations=TOTAL_ITERATIONS,
    filename_padding_width=FILENAME_PADDING_WIDTH,
    debug_log=debug_log
)

if ADD_METRICS:
    rho_min = np.min(rho)
    rho_max = np.max(rho)
    title = "Density map"
    plotter.density_map_standalone(rho, rho_min, rho_max, title, iteration)
    plotter.save_phi_snapshot(_phi, iteration, phi_star_G, phi_star_L)

if DUMP_TO_VTK:
    vtkdumper = VTKDataDumper(Xn+2, Yn+2, Zn+2, output_dir="paraview", spacing=(1.0, 1.0, 1.0))

inner_loop_log_file = 'ph_sor_log.txt'      

while iteration < TOTAL_ITERATIONS:
    profiler = Profiler()
    profiler.start("main_loop")

    if iteration % 100 == 0:
        debug_log('ITER', 'Iter %d: phi min=%.3e, max=%.3e', iteration, np.min(_phi), np.max(_phi))
    #Inamuro §2.3 Algorithm of computation:
    #Step 1. Using eqs (1) and (2), compute (fi(x, t+n_dt) and g(x, t+n_dt), and then compute phi(x, t+n_dt) and _u(x, t+n_dt)= with eqs (4) and (5).
    #Also rho(x, t+n_dt) is calculated with eq (4)
    if iteration == 0:
        u_zero = np.zeros_like(u_ckl)
        _fi_c = fi_c(u_zero, Kf, F, _phi)
        debug_log('ITER', 'Iter 0: _fi_c with u=0 (no advection)')
    else:
        _fi_c = fi_c(u_ckl, Kf, F, _phi)
    if RAISE_NaN_ERROR and np.any(np.isnan(_fi_c)):
        raise ValueError(f"Iter {iteration}: NaN in _fi_c")
    if ADD_METRICS_PRINT: debug_log('ITER', 'Iter %d: fi_c min=%.3e, max=%.3e | fi min=%.3e, max=%.3e', 
          iteration, np.min(_fi_c), np.max(_fi_c), np.min(_fi), np.max(_fi))  

    #Inamuro eq(2): calculation of the order parameter which distiguishes the two phases
    _fi = fi(_fi, _fi_c, tau_f)
    if RAISE_NaN_ERROR and np.any(np.isnan(_fi)):
        raise ValueError(f"Iter {iteration}: NaN in _fi")
    if ADD_METRICS_PRINT: debug_log('ITER', ' advisory Iter %d: fi min=%.3e, max=%.3e', 
          iteration, np.min(_fi), np.max(_fi))

    #Calculation of order parameter to distiguish the 2 phases
    _phi = phi(_fi)
    # In the main loop, after _phi = phi(_fi)
    if ADD_METRICS_PRINT: debug_log('ITER', 'Iter %d: phi at y=0: %.3e, y=50: %.3e, y=51: %.3e', iteration, np.mean(_phi[:,1]), np.mean(_phi[:,50]), np.mean(_phi[:,51])) 
    if RAISE_NaN_ERROR and np.any(np.isnan(_phi)):
        raise ValueError(f"Iter {iteration}: NaN in _phi")
    if RAISE_LESS_THAN_ZERO_ERROR and (np.any(_phi <= 0) or np.any(_phi >= 1/b)):
        invalid_idx = np.where((_phi <= 0) | (_phi >= 1/b))
        raise ValueError(
            f"Iter {iteration}: Invalid _phi: min={np.min(_phi):.3e}, "
            f"max={np.max(_phi):.3e}, invalid indices: {invalid_idx}"
        )    
        
    if iteration in iterationsOfInterest:
        #phi mapping
        plotter.save_phi_snapshot(_phi, iteration, phi_star_G, phi_star_L)

    #Inamuro eq(3): calculation of the predicted velocity of the two phase fluid        
    if iteration > 0:
        _gi_c = gi_c(u_ckl, rho, tau_g, Kg, iteration)
    if RAISE_NaN_ERROR and np.any(np.isnan(_gi_c)):
        raise ValueError(f"Iter {iteration}: NaN in _gi_c")        

    rho, mu = density_and_viscosity(_phi, rho_G, rho_L, phi_star_G, phi_star_L, mu_G, mu_L)
    if RAISE_NaN_ERROR and np.any(np.isnan(rho)):
        raise ValueError(f"Iter {iteration}: NaN in rho")
    if ADD_METRICS_PRINT: debug_log('ITER', 'Iter %d: rho min=%.3e, max=%.3e', iteration, np.min(rho), np.max(rho))

    if iteration in iterationsOfInterest:
        # Store 2D data (existing)
        list_avg_velocities_x[iteration] = u_ckl[0, 1:-1, :, :].copy()
        list_avg_velocities_y[iteration] = u_ckl[1, 1:-1, :, :].copy()
        
        # density mapping
        rho_min = np.min(rho)
        rho_max = np.max(rho)
        title = "Density map"
        plotter.density_map_standalone(rho, rho_min, rho_max, title, iteration)
        rho_slice = rho[:, :, density_profile_z_position].copy()
        density_slices.append((iteration, rho_slice))

    _gi = gi(_gi, _gi_c, u_ckl, rho, mu, iteration) 
    if RAISE_NaN_ERROR and np.any(np.isnan(_gi)):
        raise ValueError(f"Iter {iteration}: NaN in _gi")    
    #_gi = giExt(_gi, _gi_c, u_ckl, rho, mu) 

    update_ghost_nodes_top_bottom(_fi, _gi, c)

    #=> here the boundary conditions
    #Bounce-Back Top and Bottom
    _fi = bounceBackTopBottom2(_fi)
    _gi = bounceBackTopBottom2(_gi)

    #4.1b. assign inlet boundary values -> B)
    apply_periodic_boundary_conditions(_fi, _gi, Xn)        


    #Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    #Inamuro eq(5): Compute u(x,t+n_dt)
    #Kürger et al, p. 241 eq. (6.29) & Table 6.1
    #1. Shan-Chen - A=tau*n_dt
    A = n_dt*n_dt

    forcing_term = A*force_(F_lattice, rho)*n_dt
    u_ckl_star = np.einsum('ia,ijkh->ajkh', c, _gi) + forcing_term * ADD_FORCING_TERM

    ############################ Step 2a. Calculate h, p, Pressure Correction #################
    profiler.start("ph_sor")

    # --- Configuration ---
    MAX_PH_ITERATIONS = 500
    OMEGA_SOR = 1.5
    TOLERANCE_INITIAL = epsilon_threshold * 10.0
    TOLERANCE_MIN = 1e-8
    TOLERANCE_DECAY_RATE = 0.99  # Reduce tolerance every 10 iterations

    # --- Initialize ---
    _p_prev = _p0.copy()
    max_eps = float('inf')        # Will be updated in first iteration
    ph_iter = 0
    current_tolerance = TOLERANCE_INITIAL

    # --- Main PH-SOR Loop ---
    while ph_iter < MAX_PH_ITERATIONS and max_eps > current_tolerance:
        ph_iter += 1

        # Solve for p and h
        p, h = ph(h, rho, u_ckl_star, iteration)
        delta_p = p - _p_prev

        # --- Adaptive SOR to avoid overshoot ---
        omega = OMEGA_SOR
        while omega >= 1e-4:
            p_trial = _p_prev + omega * delta_p
            eps_trial = np.abs(p_trial - _p_prev) / np.maximum(rho, 0.1)
            max_eps_trial = np.max(eps_trial)

            if max_eps_trial <= current_tolerance:
                _p0 = p_trial
                break
            else:
                omega *= 0.5  # Reduce step size

        else:
            # If omega became too small: take safe fractional step
            scale = current_tolerance / max(max_eps_trial, 1e-12)
            _p0 = _p_prev + delta_p * min(1.0, scale)

        # --- Update error ---
        max_eps = np.max(np.abs(_p0 - _p_prev) / np.maximum(rho, 0.1))
        _p_prev = _p0.copy()

        # --- Log progress ---
        progress_pct = (ph_iter / MAX_PH_ITERATIONS) * 100
        print(f"ITER: Simulation Execution -> ph_iter: {ph_iter}; "
            f"max_eps: {max_eps:.8e}; "
            f"MAX_PH_ITERATIONS: {MAX_PH_ITERATIONS}; "
            f"iteration: {iteration}; "
            f"{progress_pct:.3f} %")

        # --- Adaptive tolerance tightening ---
        if ph_iter % 10 == 0:
            current_tolerance = max(current_tolerance * TOLERANCE_DECAY_RATE, TOLERANCE_MIN)

    # --- Final log to file ---
    with open(inner_loop_log_file, 'a') as f_log:
        f_log.write(f"{iteration};{ph_iter};{max_eps:.6e};{current_tolerance:.6e}\n")

    # --- Convergence check ---
    if max_eps > current_tolerance:
        debug_log(
            'ITER',
            'ph stalled at global %d, ph_iter=%d, eps=%.3e > tol=%.3e, using approx p',
            iteration, ph_iter, max_eps, current_tolerance
        )
    else:
        print(f"Converged: ph_iter={ph_iter}, final max_eps={max_eps:.3e}")

    profiler.stop("ph_sor")
    ####################################################################

    #Inamuro eq(22 & 24): assign resultant p to _p0 for next iteration
    _p0 = p

    #Step 3: Compute u(x,t+n_dt) using eq. (20)
    #Inamuro eq(20): corrected current velocity u which satisfies the continuity equation div.u=0
    u_ckl = -gradient_p(p)*n_dt/(rho*Sh) + u_ckl_star
    
    # In main loop, after u_ckl update:total_mom_x = np.sum(rho * u_ckl[0])  # Add this
    if ADD_METRICS:
        if iteration == 0 or iteration==(TOTAL_ITERATIONS - 1) or iteration % 100 == 0:

            vtkdumper.dump_to_vti(_phi, iteration, field_name="_phi")
            vtkdumper.dump_to_vti(rho, iteration, field_name="rho")
            vtkdumper.dump_to_vti(u_ckl, iteration, field_name="u_ckl")

            u_ckl_x_min = np.min(u_ckl[0])
            u_ckl_x_max = np.max(np.abs(u_ckl[0]))
            u_ckl_y_min = np.min(u_ckl[1])
            u_ckl_y_max = np.max(np.abs(u_ckl[1]))

            u_ckl_z_min = np.min(u_ckl[2])
            u_ckl_z_max = np.max(np.abs(u_ckl[2]))

            invariant = np.sum(rho*u_ckl[0])
            MomentumBounds.append((iteration, u_ckl_x_min, u_ckl_x_max, invariant))
            rho_min = np.min(rho)
            rho_max = np.max(rho)    
            rho_bounds.append((iteration, rho_min, rho_max))
            Invariants.append((iteration, invariant))
            if ADD_METRICS: 
                debug_log('FIELD','Iteration={iteration}', 'max|u_x|={u_ckl_x_max:.2e}, invariant={invariant:.2e}')
            GrowthMetric_uckl_x.append((iteration, u_ckl_x_max))
            GrowthMetric_uckl_y.append((iteration, u_ckl_y_max))

            uckl_star_y = np.max(np.abs(u_ckl_star[1]))
            GrowthMetric_uckl_star_y.append((iteration, uckl_star_y))
            du_dx, du_dy, du_dz = c_first_derivative(u_ckl_star[0])
            dv_dx, dv_dy, dv_dz = c_first_derivative(u_ckl_star[1])
            div_u_raw = du_dx + dv_dy

            GrowthMetric_div_u_raw.append((iteration, np.max(np.abs(du_dx)), np.max(np.abs(du_dy)), np.max(np.abs(div_u_raw))))
            GrowthMetric_u_ckl_star_du_dy.append((iteration, du_dy))            

            PhIters.append((iteration, ph_iter))
            spuriousField1 = np.max(np.abs(np.gradient(p)))
            laplacian_phi = c_second_derivative(_phi)
            spuriousField2 = np.max(np.abs(laplacian_phi))
            AuxFields.append((iteration, spuriousField1, spuriousField2))

            ################################# 6. Additional Diagnostics and Rollbacks #######################################################
            DivU_max.append((iteration, np.max(np.abs(div_u_raw))))
            PhEps_max.append((iteration, np.max(epsilon)))
            ##################################################################################################################################

    #streaming has commenced

    # Get the maximum density and its location
    max_density = np.max(rho)
    max_location = np.unravel_index(np.argmax(rho), rho.shape)
    
    # Update plots and parameters
    _rho_full_range = rho
    if iteration in iterationsOfInterest:
        list_avg_velocities_x[iteration] = u_ckl[0, 1:-1, :, :]
        list_avg_velocities_y[iteration] = u_ckl[1, 1:-1, :, :]

    if iteration == 0 or iteration==(TOTAL_ITERATIONS - 1) or iteration % 100 == 0:
        epsilon_u_ckl = np.abs(u_ckl[0,int(Xn/2),int(Yn/2),int(Zn/2)] - u_ckl_midpoint0)
        epsilon_u_ckl_list.append((iteration, epsilon_u_ckl))    
        u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2),int(Zn/2)]


    profiler.stop("main_loop")
    # Every 100 iterations, print or save profiler info
    if iteration % 100 == 0:
        profile_filename = (f'{SCRIPT_FILENAME}_profile_Iteration_{iteration}.json')
        profile_str = profiler.report()
        print(profile_str)        
        profile_str = profiler.to_json(profile_filename)        
        profile_barchart_filename = (f'{SCRIPT_FILENAME}_profile_Iteration_{iteration}.png')
        profiler.plot(title=f"{SCRIPT_FILENAME}: Iteration {iteration}", filename=profile_barchart_filename)

    #Step 4: re-iterate
    iteration += 1
    if iteration % 1 == 0:
        progress = (iteration / TOTAL_ITERATIONS) * 100.0
        debug_log('ITER', 'Simulation Execution -> ph_iter: %d; max_eps:%.8e; TOTAL_ITERATIONS: %d; iteration: %d; %.3f %%', 
            ph_iter, max_eps, 
            TOTAL_ITERATIONS, iteration, progress)
        
    # Step 5: slowly ramp C
    if iteration % 100 == 0 and C < 1.0:
        C = min(C + 0.05, 1.0)
        debug_log('ITER', "step %s: C =  %.3f %%", iteration, C)
        

end = time.perf_counter()
diff = end - start
rho_in, rho_out = _rho_full_range[1, Yn // 2, Zn // 2], _rho_full_range[Xn, Yn // 2, Zn // 2]
rho_min = np.min(_rho_full_range)
rho_max = np.max(_rho_full_range)
debug_log('FIELD', '_rho_full_range min = %(min).6f, max = %(max).6f', extra=dict(min=rho_min, max=rho_max))

filtered_u_ckl_dict_x = plotter.filter_u_ckl_fullrange(list_avg_velocities_x, iterationsOfInterest)
filtered_u_ckl_list_x = list(filtered_u_ckl_dict_x.values())

filtered_u_ckl_dict_y = plotter.filter_u_ckl_fullrange(list_avg_velocities_y, iterationsOfInterest)
filtered_u_ckl_list_y = list(filtered_u_ckl_dict_y.values())

# height ratios based on lattice dimensions
aspect_ratio = Xn / Yn
top_row_height = 1 #1.5
bottom_row_height = 1
height_ratios = [top_row_height, bottom_row_height, bottom_row_height] if 'top_row_height' in globals() else [1, 1, 1]


# 3x2 multi-plot grid
paneLabel = f"Dashboard D2Q9 LB method for incompressible two-phase flows Inamuro et al 2004 Lattice [{Xn} {Yn}] Single processor"
fig1, ax1 = plt.subplots(
    3, 2,
    figsize=(15, 10),
    gridspec_kw={
        'width_ratios': [2, 4], 
        'height_ratios': height_ratios,
        'left': 0.15, 'right': 0.85, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

# In the fig1, ax1 section
sectionPosition = int(Xn/2)
U_max_x = np.max(filtered_u_ckl_list_x[-1][sectionPosition, 1:Yn+1])
plotter.amplitude_plot(ax1[0, 0], filtered_u_ckl_dict_x, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_x$", f"Amplitude u$_x$ at x={Xn}", sectionPosition, Yn)
plotter.amplitude_plot(ax1[1, 0], filtered_u_ckl_dict_y, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_y$", f"Amplitude u$_y$ at x={Xn}", sectionPosition, Yn)

_iteration = TOTAL_ITERATIONS
plotter.velocity_map(ax1[0, 1], filtered_u_ckl_list_x[-1][1:-1, 1:Yn+1, Zn//2], _iteration, "Velocity [u$_x$] map")
plotter.velocity_map(ax1[1, 1], filtered_u_ckl_list_y[-1][1:-1, 1:Yn+1, Zn//2], _iteration, "Velocity [u$_y$] map")

plotter.density_profiles(ax1[2, 0], density_slices, density_profile_x_position, Xn, Yn)

if PRESSURE_IN_DENSITY_MAP:
    min_value = 0 
    _pressure_full_range = (_rho_full_range - min_value) * Cs**2 
    _pressure_out = (rho_min - min_value) * Cs**2
    _pressure_in = (rho_max - min_value) * Cs**2
    title = "Pressure map"
    plotter.density_mapExt(ax1[2, 1], _pressure_full_range, _pressure_out, _pressure_in, title, iteration)
else:
    title = "Density map"
    plotter.density_mapExt(ax1[2, 1], _rho_full_range, rho_min, rho_max, title, iteration)

text = f"Run-time: {diff:.1f} s"
fig1.text(0.5, 0.98, text, ha='center', va='top', fontsize=12)
fig1.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreesurfaceImages")
os.makedirs(images_dir, exist_ok=True)
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_{USE_CASE_TAG}_channel_parameters.png")
fig1.savefig(save_path, dpi=300, bbox_inches='tight')
debug_log('INIT', 'Saved 3x2 grid: %s', save_path)
plt.close(fig1)


# 3 rows, 4 columns
paneLabel = f"Metrics - D2Q9 LB method for incompressible two-phase ﬂows Inamuro et al 2004 Lattice [{Xn} {Yn}] Single processor"
fig2, ax2 = plt.subplots(
    3, 4,  
    figsize=(18, 10), 
    gridspec_kw={
        'width_ratios': [1, 1, 1, 1],
        'height_ratios': height_ratios,
        'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

plotter.plot_bounds_ext(GrowthMetric_uckl_x, "GrowthMetric_uckl_x", ax2[0, 0])
plotter.plot_bounds_ext(GrowthMetric_uckl_y, "GrowthMetric_uckl_y", ax2[0, 1])
plotter.plot_bounds_ext(GrowthMetric_uckl_star_y, "GrowthMetric_uckl_star_y", ax2[0, 2])
plotter.plot_bounds_ext(rho_bounds, "rho_bounds", ax2[0, 3])

plotter.plot_bounds_ext(epsilon_u_ckl_list, "epsilon_u_ckl growth", ax2[1, 0])
plotter.plot_bounds_ext(Invariants, "Invariants", ax2[1, 1])
plotter.plot_momentum_bounds(MomentumBounds, "MomentumBounds", ax2[1, 2])
plotter.plot_bounds_ext(PhIters, "PhIters", ax2[1, 3])    

series_labels = ["du_dx","du_dy","dv_dx","dv_dy","div_u"]
plotter.plot_bounds_ext(GrowthMetric_div_u_raw, "GrowthMetric_div_u_raw", ax2[2, 0], series_labels)
series_labels = ["np.gradient(p)","laplacian_phi"]
plotter.plot_bounds_ext(AuxFields, "AuxFields", ax2[2, 1], series_labels)
plotter.plot_bounds_ext(DivU_max, "DivU_max", ax2[2, 2])
plotter.plot_bounds_ext(PhEps_max, "PhEps_max", ax2[2, 3])

fig2.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreesurfaceImages")
os.makedirs(images_dir, exist_ok=True)
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_{USE_CASE_TAG}_Metrics_{TOTAL_ITERATIONS:0{FILENAME_PADDING_WIDTH}d}.png")
fig2.savefig(save_path, dpi=300, bbox_inches='tight')
debug_log('INIT', 'Saved 3x4 grid: %s', save_path)
plt.close(fig2)


########### upload this file and results to GitHub repo
if UPLOAD_TO_GITHUB:
    uploader = GitHubUploader(
        debug_log=debug_log,
        script_filename=SCRIPT_FILENAME,
        script_full_path=SCRIPT_FULL_PATH,
        scripts_path=SCRIPTS_PATH,
        plots_path=PLOTS_PATH,
        images_subdir=IMAGES_SUBDIR,
        log_file=LOG_FILE,
        token_file='github-repo-token.txt'
    )
    try:
        uploader.upload_results(upload_log=True)
        debug_log('INIT', f'Upload complete: Script at root, results in https://github.com/faircm2/Lb-Python/tree/main/{uploader.results_folder}')
    except Exception as e:
        debug_log('ERROR', f'Upload failed: {e}')

    # Example usage at the end of your script:
    # Assuming SCRIPT_FILENAME = 'your_script.py'
    # uploader = GitHubUploader(SCRIPT_FILENAME, repo_name='yourusername/your-repo-name')
    # uploader.upload_results()