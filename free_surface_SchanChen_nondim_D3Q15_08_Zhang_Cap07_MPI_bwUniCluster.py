#this simulation a microchannel flow is modelled along a plane inserted vertically and symetrically in the channel center#aligned to the z-axis vertically and in the x-axis direction along hte channel length
#the axes in the plane are y-axis in the vertical direction and x-axis in the channel horizontal direction
#The simulation is based on the paper by Zhang et al, 2020, Physics of Fluids 32, 103301 (2020)
# Add at top of script (Python 3.7+ for forward refs in annotations)
from __future__ import annotations

from mpi4py import MPI

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple  # Import these for proper type hints

import numpy as np

import sys
import logging

# Force stdout and stderr to use UTF-8 (Windows fix)
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


@dataclass
class FlowConfig:
    # --- Required per-scenario fields ---
    n_dx: float
    dx: float

    # --- Scenario-specific parameters ---
    tau_f: float #relaxation time tau_f, should be > 0,5
    tau_g: float
    Kf: float

    # --- viscous forces ---
    # Zhang interface thickness    
    '''W - the measurement of the interface thickness'''
    vf_W: int
    '''σ - surface tension coefficient'''
    # Zhang surface tension coefficient, A. Wetting behavior of a particle on the liquid–gas interface
    vf_sigma: float
    '''θ - equilibrium contact angle'''
    vf_theta: float

    velocitySetSize: int = 15 #velocity set size, for D3Q15 ==15

    # --- Defaults ---
    alpha: float = 0.0                  # inclination of channel (degrees)
        
    # --- Default physical constants ---
    g: float = 9.81
    rho_G: float = 0.01
    rho_L: float = 10.0
    phi_star_G: float = 0
    phi_star_L: float = 1

    # --- Iteration / simulation control (with defaults) ---
    MULTIPLES: int = 1
    CORE_TOTAL_ITERATIONS: int = 12001
    ENFORCE_MASS_CONSERVATION: bool = True
    ENABLE_PHI_CLIPPING: bool = True

    # --- Derived quantities ---
    @property
    def dt(self) -> float:
        """eq.25 with the physical nu: dt follows tau_f."""
        return Cs2 * (self.tau_f - 0.5) * self.dx**2 / nu

    @property
    def TOTAL_ITERATIONS(self) -> int:
        """Total number of simulation iterations."""
        return int(self.CORE_TOTAL_ITERATIONS * self.MULTIPLES)

    @property
    def alpha_rad(self) -> float:
        """Inclination angle in radians."""
        return np.radians(self.alpha)
    
    @property 
    def g_lattice(self) -> float:
        return self.g * (self.dt**2 / self.dx)

    @property
    def g_x(self) -> float:
        """x-component of gravitational acceleration."""
        return self.g_lattice * np.sin(self.alpha_rad)

    @property
    def g_y(self) -> float:
        """y-component of gravitational acceleration."""
        return 0
    
    @property
    def g_z(self) -> float:
        """z-component of gravitational acceleration."""
        return -self.g_lattice * np.cos(self.alpha_rad)    

    @property
    def F_body(self) -> np.ndarray:
        """Body force vector in world coordinates."""
        return np.array([self.g_x, self.g_y, self.g_z])

    @property
    def vf_kappa(self) -> float:
        '''κ - relaxation factor'''
        kappa = (3.0 * self.vf_sigma * self.vf_W) / 2.0    
        return kappa

    @property
    def vf_beta(self) -> np.ndarray:
        '''β - parameter to quantify the position of the contact line'''
        beta = (12.0 * self.vf_sigma) / self.vf_W
        return beta   
    
    @property
    def vf_theta_rad(self) -> int:
        '''theta - contact angle'''
        return np.deg2rad(self.vf_theta)


#flags
ADD_METRICS = True

# Constants
DEFAULT_D_ND = 384
SCRIPT_FILENAME = os.path.splitext(os.path.basename(__file__))[0] 
IMAGES_SUBDIR = "FreesurfaceImages"  # Local subdir
script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
images_subdir = os.path.join(script_dir, IMAGES_SUBDIR)
os.makedirs(images_subdir, exist_ok=True)  # create folder if it doesn't exist  
# Create the specific subdirectory
LOG_FILE = 'lbm_debug.log'


######### logging ####################################################################################################
# Global debug level (set once at init, e.g., based on flags like VERBOSE1, ADD_METRICS_PRINT)
# 0: none (suppress all)
# 1: init (startup params only)
# 2: iter (iteration progress, e.g., %100 summaries)
# 3: fields (detailed field stats like min/max per component)
# Global DEBUG_LEVEL (unchanged)
DEBUG_LEVEL = 0  # Or whatever; controls prefixed categories only

# Improved logging config: Root at WARNING to suppress most noise, logger at DEBUG
logging.getLogger().setLevel(logging.WARNING)  # Root: Silence everything by default

logger = logging.getLogger(LOG_FILE)
logger.setLevel(logging.DEBUG)  # logger: Always DEBUG internally

# Custom filter to enforce DEBUG_LEVEL (attached to logger)
# Custom filter to show only main loop progress
class DebugLevelFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:   # let WARN and ERROR (e.g. from validate_field) through
            return True        
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

# Silence noisy external loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('kiwisolver').setLevel(logging.WARNING)

# Debug log function (unchanged)
_IS_ROOT = MPI.COMM_WORLD.Get_rank() == 0
      
######################################################################################################################

# ──────────────────────────────────────────────────────────────────────────────────────────
# lattice parameters
# ──────────────────────────────────────────────────────────────────────────────────────────

Cs=np.sqrt(1/3)
Cs2 = Cs**2
Cs4 = Cs**4

Yn=int(DEFAULT_D_ND) #+1
Xn=int(DEFAULT_D_ND) * 2 #200 #int(Yn*L/D)
Zn=int(DEFAULT_D_ND) #200 #int(Zn*L/D)

#relaxation time tau_f (eq.25): default 0.6, takes the same --tau_f the parameter study passes
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--tau_f', type=float, default=1.0)
tau_f = _pre_parser.parse_known_args()[0].tau_f

_pre_parser.add_argument('--periodic_x', type=int, default=1)
PERIODIC_X = bool(_pre_parser.parse_known_args()[0].periodic_x)   # 1: x periodic (channel), 0: closed x walls
_pre_parser.add_argument('--rho_L', type=float, default=10.0)   # lattice density of the liquid, same --rho_L as read again below
rho_L_lat = _pre_parser.parse_known_args()[0].rho_L

rho_0=1e3 #kg/m^3

_pre_parser.add_argument('--nu', type=float, default=2.9e-6)   # kinematic viscosity of the liquid [m^2/s]; water: 1.0e-6
nu = _pre_parser.parse_known_args()[0].nu
_pre_parser.add_argument('--D', type=float, default=1e-3)   # channel width [m]; the experiment: 1.2e-3
D = _pre_parser.parse_known_args()[0].D
dx = D/DEFAULT_D_ND
_pre_parser.add_argument('--fill', type=float, default=38.0)   # initial liquid depth [cells above the floor]; the interface centre sits at fill + 0.5
FILL = _pre_parser.parse_known_args()[0].fill
dt = Cs2*(tau_f-0.5)*(dx**2/nu)


#we need Cl, Crho, Ct
# 1. Conversion factor Cl for length
Cl = dx #freely chosen
n_dx = dx/Cl #-> dx_nd=1
n_dy = n_dx
n_dz = n_dx

#2. Conversion factor Crho for density
Crho = rho_0 / rho_L_lat   # kg/m^3 per lattice density unit (lattice liquid density rho_L_lat <-> rho_0)

#3. Conversion factor Ct for time
Ct=dt
n_dt = dt/Ct #-> dt_nd=1

#4. Conversion factor Cu for velocity

#5. Conversion factor CF for Force

#6. Conversion factor Cf for frequency

#change nu_nd in order to achieve U_nd=0,1
nu_nd = Cs2 * (tau_f - 0.5)

#discrete velocity channels for D3Q15
_AXIS = [
    [ 0,  0,  0],
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
]
# 8 corner directions (all three components ±1) 
_CORNER = [
    [ 1,  1,  1], [-1, -1, -1],
    [ 1,  1, -1], [-1, -1,  1],
    [ 1, -1,  1], [-1,  1, -1],
    [-1,  1,  1], [ 1, -1, -1],
]

LATTICE_SETS = {
    "D3Q15": {
        "c": np.array(_AXIS + _CORNER),
        "E": np.array([2/9] + [1/9]*6 + [1/72]*8),
    }
}        

# Zhang bottom p. 32, 2nd column constant w0 & Krüger: force weights
LATTICE_MODEL = "D3Q15"   # "D3Q15"
c = LATTICE_SETS[LATTICE_MODEL]["c"]
E = LATTICE_SETS[LATTICE_MODEL]["E"]

# To match original exactly (including the buggy du+du), broadcast per-component on full grid
# Expansions for broadcasting over i
E_exp = E[:, np.newaxis, np.newaxis, np.newaxis]  # (15,1,1,1)


# ──────────────────────────────────────────────────────────────────────────────────────────
# MPI configuration
# ──────────────────────────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD

def _abort_all_ranks_on_error(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = _abort_all_ranks_on_error

assert comm.Get_size() in (512, 1024, 2048), f"need 512, 1024 or 2048 ranks, got {comm.Get_size()}"
dims = {512: [8, 8, 8], 1024: [16, 8, 8], 2048: [16, 16, 8]}[comm.Get_size()]
cart = comm.Create_cart(dims, periods=[PERIODIC_X, False, False], reorder=True)
rank_coords = cart.Get_coords(cart.Get_rank())

rank = comm.Get_rank()
size = comm.Get_size()

# neigbour ranks along each axis
x_lo, x_hi = cart.Shift(0, 1)
y_lo, y_hi = cart.Shift(1, 1)
z_lo, z_hi = cart.Shift(2, 1)

def exchange_ghosts_x(_phi, x_lo, x_hi, cart):
    send_to_hi = np.ascontiguousarray(_phi[-2, :, :])   # last real x-layer -> goes to the +x neighbor
    send_to_lo = np.ascontiguousarray(_phi[1, :, :])    # first real x-layer -> goes to the -x neighbor
    recv_from_lo = np.empty_like(send_to_lo)
    recv_from_hi = np.empty_like(send_to_hi)

    req1 = cart.Isend(send_to_hi, dest=x_hi)
    req2 = cart.Irecv(recv_from_lo, source=x_lo)
    req3 = cart.Isend(send_to_lo, dest=x_lo)
    req4 = cart.Irecv(recv_from_hi, source=x_hi)
    MPI.Request.Waitall([req1, req2, req3, req4])

    if x_lo != MPI.PROC_NULL:
        _phi[0, :, :] = recv_from_lo
    if x_hi != MPI.PROC_NULL:
        _phi[-1, :, :] = recv_from_hi

def exchange_ghosts_y(_phi, y_lo, y_hi, cart):
    send_to_hi = np.ascontiguousarray(_phi[:, -2, :])   # last real y-layer -> goes to the +y neighbor
    send_to_lo = np.ascontiguousarray(_phi[:, 1, :])    # first real y-layer -> goes to the -y neighbor
    recv_from_lo = np.empty_like(send_to_lo)
    recv_from_hi = np.empty_like(send_to_hi)

    req1 = cart.Isend(send_to_hi, dest=y_hi)
    req2 = cart.Irecv(recv_from_lo, source=y_lo)
    req3 = cart.Isend(send_to_lo, dest=y_lo)
    req4 = cart.Irecv(recv_from_hi, source=y_hi)
    MPI.Request.Waitall([req1, req2, req3, req4])

    if y_lo != MPI.PROC_NULL:
        _phi[:, 0, :] = recv_from_lo
    if y_hi != MPI.PROC_NULL:
        _phi[:, -1, :] = recv_from_hi        

def exchange_ghosts_z(_phi, z_lo, z_hi, cart):
    send_to_hi = np.ascontiguousarray(_phi[:, :, -2])   # last real z-layer -> goes to the +z neighbor
    send_to_lo = np.ascontiguousarray(_phi[:, :, 1])    # first real z-layer -> goes to the -z neighbor
    recv_from_lo = np.empty_like(send_to_lo)
    recv_from_hi = np.empty_like(send_to_hi)

    req1 = cart.Isend(send_to_hi, dest=z_hi)
    req2 = cart.Irecv(recv_from_lo, source=z_lo)
    req3 = cart.Isend(send_to_lo, dest=z_lo)
    req4 = cart.Irecv(recv_from_hi, source=z_hi)
    MPI.Request.Waitall([req1, req2, req3, req4])

    if z_lo != MPI.PROC_NULL:
        _phi[:, :, 0] = recv_from_lo
    if z_hi != MPI.PROC_NULL:
        _phi[:, :, -1] = recv_from_hi      


def gather_full_volume(local_field, x_offset, y_offset, z_offset, local_Xn, local_Yn, local_Zn, Xn, Yn, Zn, dtype=None):
    piece = local_field[1:-1, 1:-1, 1:-1]
    if dtype is not None:
        piece = piece.astype(dtype)
    payload = (x_offset, y_offset, z_offset, piece)
    gathered = comm.gather(payload, root=0)
    if rank != 0:
        return None
    full_field = np.zeros((Xn+2, Yn+2, Zn+2), dtype=dtype if dtype is not None else np.float64)
    for x_off, y_off, z_off, piece in gathered:
        full_field[x_off+1:x_off+1+local_Xn, y_off+1:y_off+1+local_Yn, z_off+1:z_off+1+local_Zn] = piece
    return full_field          


local_Xn = Xn // dims[0]
local_Yn = Yn // dims[1]
local_Zn = Zn // dims[2]

x_offset = rank_coords[0] * local_Xn
y_offset = rank_coords[1] * local_Yn
z_offset = rank_coords[2] * local_Zn


# ──────────────────────────────────────────────────────────────────────────────────────────
# Configurations
# ──────────────────────────────────────────────────────────────────────────────────────────
# 0.2 Benchmark - inclined channel
BENCHMARK_INCLINED = FlowConfig(
    dx=dx,
    n_dx=n_dx, alpha=45,
    tau_g=0.65, 
    tau_f=tau_f, #tau_f=1.25, #see eq(25) + Example A , t_g start with 0.6, increase as necessary -> was tau_f=0.65
    Kf=0.002,
    #Increase interface smoothness: Set vf_W = 6 or 8 in FlowConfig to widen the diffuse interface, reducing sharp edges.
    vf_W = 4, #was 6
    vf_sigma = 0.01, #0.072
    vf_theta = 60.0, #60
    MULTIPLES=1,
    ENFORCE_MASS_CONSERVATION = True
)

USE_CASES = {
    "proof_inclined": {"PHI_DISTRIBUTION": "HORIZONTAL", "fc": BENCHMARK_INCLINED}
}

ACTIVE_CASE = "proof_inclined"
PHI_DISTRIBUTION = USE_CASES[ACTIVE_CASE]["PHI_DISTRIBUTION"]
fc = USE_CASES[ACTIVE_CASE]["fc"]  # current FlowConfig

# CLI overrides for param_study.py sweeps (tau_g, tau_f, vf_theta)
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument('--tau_g', type=float, default=None)
_arg_parser.add_argument('--tau_f', type=float, default=None)
_arg_parser.add_argument('--vf_theta', type=float, default=None)
_arg_parser.add_argument('--vf_sigma', type=float, default=None)
_arg_parser.add_argument('--alpha', type=float, default=None)
_arg_parser.add_argument('--multiples', type=int, default=None)
_arg_parser.add_argument('--rho_G', type=float, default=None)
_arg_parser.add_argument('--rho_L', type=float, default=None)
_arg_parser.add_argument('--vf_W', type=int, default=None)

_args, _ = _arg_parser.parse_known_args()
if _args.tau_g is not None:
    fc.tau_g = _args.tau_g
if _args.tau_f is not None:
    fc.tau_f = _args.tau_f
if _args.vf_theta is not None:
    fc.vf_theta = _args.vf_theta
if _args.vf_sigma is not None:
    fc.vf_sigma = _args.vf_sigma    
if _args.alpha is not None:
    fc.alpha = _args.alpha    
if _args.multiples is not None:
    fc.MULTIPLES = _args.multiples
if _args.rho_G is not None:
    fc.rho_G = _args.rho_G    
if _args.rho_L is not None:
    fc.rho_L = _args.rho_L    
if _args.vf_W is not None:
    fc.vf_W = _args.vf_W    


# ──────────────────────────────────────────────────────────────────────────────────────────
# methods
# ──────────────────────────────────────────────────────────────────────────────────────────

# Zhang eq(26): Compute the phase field phi
def phi(fc, z_g):
    """
    Compute the phase field phi from the particle distribution _f.
    Optionally clip phi to the cut-off values.

    Parameters
    ----------
    _f : np.ndarray
        Order parameter distribution function (Lattice Boltzmann populations)
    fc : object
        Contains cut-off values: fc.phi_star_G, fc.phi_star_L
    ENABLE_PHI_CLIPPING : bool
        If True, clip phi to [fc.phi_star_G, fc.phi_star_L]

    Returns
    -------
    __phi : np.ndarray
        Phase field values
    """
    __phi = np.sum(z_g, axis=0)

    if fc.ENABLE_PHI_CLIPPING:
        __phi = np.clip(__phi, fc.phi_star_G, fc.phi_star_L)

    return __phi


def zhang_gradient(fc, __phi):
    Nx, Ny, Nz = __phi.shape
    grad_x = np.zeros_like(__phi)
    grad_y = np.zeros_like(__phi)
    grad_z = np.zeros_like(__phi)

    for i in range(fc.velocitySetSize):
        cx, cy, cz = c[i]
        phi_shifted = np.roll(np.roll(np.roll(__phi, -cx, axis=0), -cy, axis=1), -cz, axis=2)
        grad_x += E[i] * cx * phi_shifted
        grad_y += E[i] * cy * phi_shifted
        grad_z += E[i] * cz * phi_shifted

    prefactor = 1.0 / Cs2
    grad_x *= prefactor
    grad_y *= prefactor
    grad_z *= prefactor

    # eq(22) is only meaningful at fluid nodes - zero it at the ghost/solid
    # layer so it can never inject a spurious force there, independent of
    # whatever the stencil read across the wrap.
    grad_x[0, :, :] = 0.0; grad_x[-1, :, :] = 0.0
    grad_x[:, 0, :] = 0.0; grad_x[:, -1, :] = 0.0
    grad_x[:, :, 0] = 0.0; grad_x[:, :, -1] = 0.0

    grad_y[0, :, :] = 0.0; grad_y[-1, :, :] = 0.0
    grad_y[:, 0, :] = 0.0; grad_y[:, -1, :] = 0.0
    grad_y[:, :, 0] = 0.0; grad_y[:, :, -1] = 0.0

    grad_z[0, :, :] = 0.0; grad_z[-1, :, :] = 0.0
    grad_z[:, 0, :] = 0.0; grad_z[:, -1, :] = 0.0
    grad_z[:, :, 0] = 0.0; grad_z[:, :, -1] = 0.0

    return grad_x, grad_y, grad_z


def zhang_laplacian(fc, __phi):
    Ny, Nx, Nz = __phi.shape
    laplacian = np.zeros((Ny, Nx, Nz))

    for i in range(1, fc.velocitySetSize):  # i=0 term cancels
        cx, cy, cz = c[i]
        phi_shifted = np.roll(np.roll(np.roll(__phi, cx, axis=0), cy, axis=1), cz, axis=2)
        laplacian += E[i] * (phi_shifted - __phi)

    laplacian *= (2.0 / Cs2)     # usually 6.0

    # eq(22)'s Laplacian is only meaningful at fluid nodes - zero it at the
    # ghost/solid layer for the same reason as zhang_gradient: nothing
    # downstream (chemical_potential -> Fs) needs a real value there, and this makes
    # the wrap-vs-clamp question moot.
    laplacian[0, :, :] = 0.0; laplacian[-1, :, :] = 0.0
    laplacian[:, 0, :] = 0.0; laplacian[:, -1, :] = 0.0
    laplacian[:, :, 0] = 0.0; laplacian[:, :, -1] = 0.0

    return laplacian


# Zhang text below eq(4): 
# where ... ρ = ϕρL + (1 − ϕ)ρG is the density of the fluid mixture; 
# similarly, η = ϕηL + (1 − ϕ)ηG is the dynamic viscosity of the fluid 
# mixture. In this study, we use the subscripts “L” and “G” to denote 
# liquid and gas, respectively.
# Zhang: ρ = φρ_L + (1-φ)ρ_G   eq.(4)
def density(fc, __phi):
    _rho = __phi * fc.rho_L + (1.0 - __phi) * fc.rho_G
    
    return _rho 


#Zhang eq(27):  the hydrodynamic pressure -> first-order moment
def zp(fc, _z_fi, _rho, u_ckl, out):
    """
    Vectorized evolution for p (hydrodynamic pressur) - D2Q9 streaming/collision.
    p =[i=0..8] ∑fi + δt/2 u⋅∇ρCs²
    """
    # channel velocity components
    np.sum(_z_fi, axis=0, out=out)  # out = sum_i fi, written in place

    # force term
    drho_dx, drho_dy, drho_dz = zhang_gradient(fc,_rho)
    grad_rho = np.stack([drho_dx, drho_dy, drho_dz], axis=0)   # (3, nx, ny, nz)
    u_dot_grad_rho = np.einsum('cijk,cijk->ijk', u_ckl, grad_rho)

    # hydrodynamic pressure
    out += (n_dt / 2.0) * Cs2 * u_dot_grad_rho

    return out


# lambda - anti-diffusion term in Allen–Cahn equation
def z_lambda(fc, __phi):
    _z_lambda = 4. * __phi * (1 - __phi) / fc.vf_W

    return _z_lambda


def zu_ckl(fc, _z_fi, rho, body_force, _capillary_force):
    _u_ckl = np.einsum('ia,ijkl->ajkl', c, _z_fi) / (Cs2 * rho)  \
        + 1/(2*rho) * _capillary_force * n_dt \
        + 1/(2*rho) * body_force

    return _u_ckl


# Zhang eq(13): collision function of pressure distribution function 
def zfi(fc, z_fi, z_fi_c, u_ckl, rho, Fs, G):
    """
    Zhang eq(13) collision + eq(15) streaming for fi in D2Q9 two-phase LBM.
    """
    global _z_star_buf

    # Zhang eq(20)/(21): forcing term Fi, shape (9, nx, ny)
    _Fi = Fi(fc, Fs, G, u_ckl, rho,  out=_Fi_buf)

    # eq(13): collision
    omega_f = 1.0 / fc.tau_f
    np.subtract(z_fi_c, z_fi, out=_z_star_buf)  #-> use the buffer
    _z_star_buf *= omega_f             # in-place
    _z_star_buf += z_fi                # in-place  (now equals z_fi - omega_f*(z_fi-z_fi_c))
    _z_star_buf += n_dt * _Fi          # n_dt*_Fi is 1 new array; += is in-place

    # eq(15): streaming
    for i in range(fc.velocitySetSize):
        exchange_ghosts_x(_z_star_buf[i], x_lo, x_hi, cart)
        exchange_ghosts_y(_z_star_buf[i], y_lo, y_hi, cart)
        exchange_ghosts_z(_z_star_buf[i], z_lo, z_hi, cart)
        z_fi[i] = np.roll(_z_star_buf[i], shift=(c[i,0],c[i,1],c[i,2]), axis=(0,1,2))

    return z_fi


# Zhang eq(6): Chemical potential
def chemical_potential(fc, __phi):
    """
    Zhang eq(6): Compute checmial potential μϕ
    μϕ = 4βϕ(ϕ - 1)(ϕ - 0.5) - κ∇2ϕ
    """
    term1 = 4. * fc.vf_beta * __phi*(__phi - 1.0) * (__phi - 0.5)
    term2 = zhang_laplacian(fc, __phi)
    _mu_phi =  term1 -  fc.vf_kappa * term2

    return _mu_phi    


# Zhang eq(5): Fs is the surface tension force, expressed in a potential form
def Fs(fc, __phi, out):
    drho_dx, drho_dy, drho_dz = zhang_gradient(fc,__phi)
    mu = chemical_potential(fc, __phi) 
    np.multiply(mu, drho_dx, out=out[0])
    np.multiply(mu, drho_dy, out=out[1])
    np.multiply(mu, drho_dz, out=out[2])

    return out


# Zhang eq(20): Fi discrete forcing term for pressure distribution function in eq(13)
def Fi(fc, Fs, G, u_ckl, rho, out):
    u_sq = np.einsum('cjkl,cjkl->jkl', u_ckl, u_ckl)                      # (X,Y,Z) - once

    term1 = 1 - 1 / (2 * fc.tau_f)
    combined_force = Fs + G    # (3,X,Y,Z) - once

    drho_dx, drho_dy, drho_dz = zhang_gradient(fc, rho * Cs2)
    stack_grad_rho = np.stack([drho_dx, drho_dy, drho_dz], axis=0)        # (3,X,Y,Z) - once

    #_Fi = np.zeros((fc.velocitySetSize, Xn+2, Yn+2, Zn+2)) -> use central buffer to prevent malloc in each method call
    for i in range(fc.velocitySetSize):
        ei_dot_u_i = c[i,0]*u_ckl[0] + c[i,1]*u_ckl[1] + c[i,2]*u_ckl[2]  # (X,Y,Z)
        s0_i = 1 + ei_dot_u_i / Cs2 + ei_dot_u_i**2 / (2 * Cs4) - u_sq / (2 * Cs2)
        s_i = E[i] * s0_i                                                 # (X,Y,Z)

        ei_u_i = c[i][:, None, None, None] - u_ckl                        # (3,X,Y,Z)
        forces_s_i = combined_force * s_i                                 # (3,X,Y,Z)
        second_term_i = (s_i - E[i]) * stack_grad_rho                     # (3,X,Y,Z)

        out[i] = term1 * np.sum(ei_u_i * (forces_s_i + second_term_i), axis=0)  # (X,Y,Z)

    return out


# Zhang eq(18): equilibrium function for pressure disrtibution function
def zfi_c(fc, u, rho, p, out):
    u_dot_u = np.sum(u**2, axis=0)  # (X,Y,Z) - once

    #-> use buffer to prevent malloc
    for i in range(fc.velocitySetSize):
        c_dot_u_i = c[i,0]*u[0] + c[i,1]*u[1] + c[i,2]*u[2]
        out[i] = E[i] * (p + rho*c_dot_u_i + rho*1.5*c_dot_u_i**2 - rho*0.5*u_dot_u)

    return out    


# Zhang eq(12): collision function of order parameter distribution function 
def zgi(fc, z_gi, z_gi_c, __phi_old, _u_ckl_old, __phi, _u_ckl):
    global _z_star_buf

    omega_g = 1.0 / fc.tau_g
    _Gi = Gi(fc, __phi_old, _u_ckl_old, __phi, _u_ckl, out=_Fi_buf)

    np.subtract(z_gi_c, z_gi, out=_z_star_buf)      # -> use buffer to prevent malloc
    _z_star_buf *= omega_g             # in-place
    _z_star_buf += z_gi                # in-place  (now equals z_gi - omega_g*(z_gi-z_gi_c))
    _z_star_buf += n_dt * _Gi          # n_dt*_Gi is 1 new array; += is in-place

    for i in range(fc.velocitySetSize):
        exchange_ghosts_x(_z_star_buf[i], x_lo, x_hi, cart)
        exchange_ghosts_y(_z_star_buf[i], y_lo, y_hi, cart)
        exchange_ghosts_z(_z_star_buf[i], z_lo, z_hi, cart)
        z_gi[i] = np.roll(_z_star_buf[i], shift=(c[i, 0], c[i, 1], c[i, 2]), axis=(0, 1, 2))

    return z_gi


# Zhang bottom p 32, 2nd column:
def n(fc, __phi):
    # n = ∇ϕ/∣∇ϕ∣ is the unit vector normal to the interface
    dphi_dx, dphi_dy, dphi_dz = zhang_gradient(fc, __phi)
    grad_phi = np.stack([dphi_dx, dphi_dy, dphi_dz], axis=0) 
    mag = np.sqrt(dphi_dx**2 + dphi_dy**2 + dphi_dz**2)
    mag = np.where(mag < 1e-12, 1.0, mag)   # avoid div-by-zero
    _n = grad_phi / mag[np.newaxis]

    return _n


# Zhang eq(19): Gi discrete forcing term for order parameter in eq(12)
def Gi(fc, __phi_old, _u_ckl_old, __phi, _u_ckl, out):
    term1 = 1 - 1 / (2 * fc.tau_g)
    # ∂t(ϕu) - Note that in the above equations, the time derivative term ∂t(ϕu) 
    # is explicitly computed by a difference between two consecutive time steps
    dphi_u_dt = __phi[np.newaxis] * _u_ckl - __phi_old[np.newaxis] * _u_ckl_old
    cs2_lambda_n = Cs2 * z_lambda(fc, __phi) * n(fc, __phi)
    vec = dphi_u_dt  + cs2_lambda_n
    ei_dot_vec = np.einsum('ic,cjkl->ijkl', c, vec)

    out[:] = term1 * E_exp * ei_dot_vec / Cs2

    return out


# Zhang eq(17): equilibrium distribution function for pressure distribution
def zgi_c(fc, u, _phi, out):
    #-> replace with buffer to prevent malloc
    for i in range(fc.velocitySetSize):
        c_dot_u_i = c[i,0]*u[0] + c[i,1]*u[1] + c[i,2]*u[2]
        out[i] = E[i] * _phi * (1 + 3.0 * c_dot_u_i)

    return out 


def bounceBackTopBottom_conservation(__gi, __fi, nx, ny, nz):
    # channel i  = 0,1,2,3,4,5,6,7,8
    # anti-channel i_ = 0,3,4,1,2,7,8,5,6

    # bottom wall (y=1) - no-slip halfway bounce-back
    # straight vertical reflection
    if z_lo == MPI.PROC_NULL:
        __gi[5, :, :, 1] = __gi[6, :, :, 0]                                 # 6-> 5, die GhostNodes spielen keine Rolle
        # diagonal reflections with horizontal shift
        __gi[ 7, 1:nx-1, 1:ny-1, 1] = __gi[ 8, 0:nx-2, 0:ny-2, 0]           # 8 -> 7
        __gi[10, 1:nx-1, 1:ny-1, 1] = __gi[ 9,  2:nx,    2:ny, 0]           # 10 -> 9
        __gi[11, 1:nx-1, 1:ny-1, 1] = __gi[12,  0:nx-2,  2:ny, 0]           # 11 -> 12
        __gi[13, 1:nx-1, 1:ny-1, 1] = __gi[14,  2:nx,  0:ny-2, 0]           # 14 -> 13

    # top wall (y=ny) - no-slip halfway bounce-back
    # straight vertical reflection
    if z_hi == MPI.PROC_NULL:
        __gi[6, :, :, nz-2] = __gi[5, :, :, nz-1]                           # 5-> 6, die GhostNodes spielen keine Rolle
        # diagonal reflections with horizontal shift
        __gi[ 8, 1:nx-1, 1:ny-1, nz-2] = __gi[7,  2:nx,    2:ny,  nz-1]     # 8 -> 7
        __gi[ 9, 1:nx-1, 1:ny-1, nz-2] = __gi[10, 0:nx-2, 0:ny-2, nz-1]     # 10 -> 9
        __gi[12, 1:nx-1, 1:ny-1, nz-2] = __gi[11, 2:nx,   0:ny-2, nz-1]     # 11 -> 12
        __gi[14, 1:nx-1, 1:ny-1, nz-2] = __gi[13, 0:nx-2,   2:ny, nz-1]     # 14 -> 13

    # Zhang eq(30) for fi:
    # where, 
    # xf is the fluid node nearest to the solid boundary
    # "i∗" represents the opposite direction of "i,"" i.e., ei∗ = −ei
    # xff = xf + ei is the fluid node next to xf
    # xs = xf + ei* is the solid node within the particle
    # xf - position of boundary fluid node  
    # xw - position of solid surface 
    # uw is the velocity of the solid boundary
    # fi(xf) = [1/(1+q)] * [q·fi(xff,t) + (1-q)·fi*(xf,t) + q·fi*(xs,t) + 2·rho·wi·(ei·uw)/cs2]
    # with q = |xf-xw|/|xf-xs| = |0.5-xw|/|xf-xs|
    # xf = 0.5
    # xw = 0
    # xs = -0.5
    # => q = |xf-xw|/|xf-xs| = |0.5-0|/|0.5--0.5| = |0.5|/|1| = 0.5
    # => uw = 0  since there is not velocity normal to the walls
    # with vertical wall geometry, q=0.5, uw=0
    # => fi(xf) = [1/(1+0.5)] * [0.5·fi(xff,t) + (1-0.5)·fi*(xf,t) + 0.5·fi*(xs,t) + 2·rho·wi·(ei·0)/cs2]
    # => fi(xf) = [1/1.5] * [0.5·fi(xff,t) + (0.5)·fi*(xf,t) + 0.5·fi*(xs,t)]
    # => fi(xf) = [2/3]·[0.5·fi(xff,t) + 0.5·fi*(xf,t) + 0.5·fi*(xs,t)]
    # => fi(xf) = (1/3)·fi(xff) + (1/3)·fi*(xf,t) + (1/3)·fi*(xs)
    # => fi(xf) = (1/3)·fi(xff) + (1/3)·fi*(xf) + (1/3)·fi*(xs)

    # xff: y-index 2
    # xs: y-index 0
    # xf: y-index 1    

    # bottom wall (i=2, 5, 6):
    if z_lo == MPI.PROC_NULL:
        __fi[5, :, :, 1] =  (1/3)*__fi[5, :, :, 2] + (1/3)*__fi[6, :, :, 1] + (1/3)*__fi[6, :, :, 0] # 6-> 5, die GhostNodes spielen keine Rolle
        # diagonal reflections with horizontal shift
        __fi[7, 1:nx-1, 1:ny-1, 1] = (1/3)*__fi[7, 2:nx, 2:ny, 2] + (1/3)*__fi[8, 1:nx-1, 1:ny-1, 1] + (1/3)*__fi[8, 0:nx-2, 0:ny-2, 0]  # 8 -> 7
        __fi[10, 1:nx-1, 1:ny-1, 1] = (1/3)*__fi[10, 0:nx-2, 0:ny-2, 2] + (1/3)*__fi[9, 1:nx-1, 1:ny-1, 1] + (1/3)*__fi[9, 2:nx, 2:ny, 0]  # 9 -> 10
        __fi[11, 1:nx-1, 1:ny-1, 1] = (1/3)*__fi[11, 2:nx, 0:ny-2, 2] + (1/3)*__fi[12, 1:nx-1, 1:ny-1, 1] + (1/3)*__fi[12, 0:nx-2, 2:ny, 0]  # 12 -> 11
        __fi[13, 1:nx-1, 1:ny-1, 1] = (1/3)*__fi[13, 0:nx-2, 2:ny, 2] + (1/3)*__fi[14, 1:nx-1, 1:ny-1, 1] + (1/3)*__fi[14, 2:nx, 0:ny-2, 0]    # 14 -> 13
    
    # top wall (i pointing S into fluid, cz=-1): 6, 8, 9, 12, 14
    if z_hi == MPI.PROC_NULL:
        __fi[6, :, :, nz-2] = (1/3)*__fi[6, :, :, nz-3] + (1/3)*__fi[5, :, :, nz-2] + (1/3)*__fi[5, :, :, nz-1]                              # 5 -> 6
        # diagonal reflections with horizontal shift
        __fi[8, 1:nx-1, 1:ny-1, nz-2]  = (1/3)*__fi[8,  0:nx-2, 0:ny-2, nz-3] + (1/3)*__fi[7, 1:nx-1, 1:ny-1, nz-2] + (1/3)*__fi[7,  2:nx, 2:ny, nz-1]  # 7 -> 8
        __fi[9, 1:nx-1, 1:ny-1, nz-2]  = (1/3)*__fi[9,  2:nx, 2:ny, nz-3] + (1/3)*__fi[10, 1:nx-1, 1:ny-1, nz-2] + (1/3)*__fi[10, 0:nx-2, 0:ny-2, nz-1]  # 10 -> 9
        __fi[12,1:nx-1, 1:ny-1, nz-2]  = (1/3)*__fi[12, 0:nx-2, 2:ny, nz-3] + (1/3)*__fi[11, 1:nx-1, 1:ny-1, nz-2] + (1/3)*__fi[11, 2:nx, 0:ny-2, nz-1]  # 11 -> 12
        __fi[14,1:nx-1, 1:ny-1, nz-2]  = (1/3)*__fi[14, 2:nx, 0:ny-2, nz-3] + (1/3)*__fi[13, 1:nx-1, 1:ny-1, nz-2] + (1/3)*__fi[13, 0:nx-2, 2:ny, nz-1]  # 13 -> 14    

    return __gi, __fi 


def bounceBackFrontBack_conservation(__gi, __fi, nx, ny, nz):

    # front wall (y=1) - no-slip halfway bounce-back
    if y_lo == MPI.PROC_NULL:
        __gi[3, :, 1, :]           = __gi[4, :, 0, :]                             # 4 -> 3 (axis, no shift)
        __gi[7, 1:nx-1, 1, 1:nz-1] = __gi[8, 0:nx-2, 0, 0:nz-2]                    # 8 -> 7
        __gi[9, 1:nx-1, 1, 1:nz-1] = __gi[10,0:nx-2, 0, 2:nz]                      # 10 -> 9
        __gi[12,1:nx-1, 1, 1:nz-1] = __gi[11,2:nx,   0, 2:nz]                      # 11 -> 12
        __gi[13,1:nx-1, 1, 1:nz-1] = __gi[14,2:nx,   0, 0:nz-2]                    # 14 -> 13

    # back wall (y=ny-2) - no-slip halfway bounce-back
    if y_hi == MPI.PROC_NULL:
        __gi[4, :, ny-2, :]            = __gi[3, :, ny-1, :]                       # 3 -> 4 (axis, no shift)
        __gi[8, 1:nx-1, ny-2, 1:nz-1]  = __gi[7, 2:nx,   ny-1, 2:nz]               # 7 -> 8
        __gi[10,1:nx-1, ny-2, 1:nz-1]  = __gi[9, 2:nx,   ny-1, 0:nz-2]             # 9 -> 10
        __gi[11,1:nx-1, ny-2, 1:nz-1]  = __gi[12,0:nx-2, ny-1, 0:nz-2]             # 12 -> 11
        __gi[14,1:nx-1, ny-2, 1:nz-1]  = __gi[13,0:nx-2, ny-1, 2:nz]               # 13 -> 14

    # front wall (y=1) - Zhang eq(30)
    if y_lo == MPI.PROC_NULL:
        __fi[3, :, 1, :] = (1/3)*__fi[3, :, 2, :] + (1/3)*__fi[4, :, 1, :] + (1/3)*__fi[4, :, 0, :]                                          # 4 -> 3
        __fi[7, 1:nx-1, 1, 1:nz-1]  = (1/3)*__fi[7, 2:nx, 2, 2:nz]  + (1/3)*__fi[8, 1:nx-1, 1, 1:nz-1] + (1/3)*__fi[8, 0:nx-2, 0, 0:nz-2]          # 8 -> 7
        __fi[9, 1:nx-1, 1, 1:nz-1]  = (1/3)*__fi[9, 2:nx, 2, 0:nz-2]  + (1/3)*__fi[10, 1:nx-1, 1, 1:nz-1] + (1/3)*__fi[10,0:nx-2, 0, 2:nz]          # 10 -> 9
        __fi[12,1:nx-1, 1, 1:nz-1]  = (1/3)*__fi[12,0:nx-2, 2, 0:nz-2]  + (1/3)*__fi[11, 1:nx-1, 1, 1:nz-1] + (1/3)*__fi[11,2:nx, 0, 2:nz]          # 11 -> 12
        __fi[13,1:nx-1, 1, 1:nz-1]  = (1/3)*__fi[13,0:nx-2, 2, 2:nz]  + (1/3)*__fi[14, 1:nx-1, 1, 1:nz-1] + (1/3)*__fi[14,2:nx, 0, 0:nz-2]          # 14 -> 13

    # back wall (y=ny-2) - Zhang eq(30)
    if y_hi == MPI.PROC_NULL:    
        __fi[4, :, ny-2, :] = (1/3)*__fi[4, :, ny-3, :] + (1/3)*__fi[3, :, ny-2, :] + (1/3)*__fi[3, :, ny-1, :]                                  # 3 -> 4
        __fi[8, 1:nx-1, ny-2, 1:nz-1]  = (1/3)*__fi[8, 0:nx-2, ny-3, 0:nz-2]  + (1/3)*__fi[7, 1:nx-1, ny-2, 1:nz-1] + (1/3)*__fi[7, 2:nx-0, ny-1, 2:nz-0]  # 7 -> 8
        __fi[10,1:nx-1, ny-2, 1:nz-1]  = (1/3)*__fi[10,0:nx-2, ny-3, 2:nz-0]  + (1/3)*__fi[9, 1:nx-1, ny-2, 1:nz-1] + (1/3)*__fi[9, 2:nx-0, ny-1, 0:nz-2]  # 9 -> 10
        __fi[11,1:nx-1, ny-2, 1:nz-1]  = (1/3)*__fi[11,2:nx-0, ny-3, 2:nz-0]  + (1/3)*__fi[12, 1:nx-1, ny-2, 1:nz-1] + (1/3)*__fi[12,0:nx-2, ny-1, 0:nz-2]  # 12 -> 11
        __fi[14,1:nx-1, ny-2, 1:nz-1]  = (1/3)*__fi[14,2:nx-0, ny-3, 0:nz-2]  + (1/3)*__fi[13, 1:nx-1, ny-2, 1:nz-1] + (1/3)*__fi[13,0:nx-2, ny-1, 2:nz-0]  # 13 -> 14

    return __gi, __fi    


def bounceBackLeftRight_conservation(__gi, __fi, nx, ny, nz):

    # left wall (x=1) - no-slip halfway bounce-back
    if x_lo == MPI.PROC_NULL:
        __gi[1, 1, :, :]            = __gi[2, 0, :, :]                                # 2 -> 1 (axis, no shift)
        # diagonal reflections with horizontal shift
        __gi[7, 1, 1:ny-1, 1:nz-1]  = __gi[8, 0, 0:ny-2, 0:nz-2]                       # 8 -> 7
        __gi[9, 1, 1:ny-1, 1:nz-1]  = __gi[10,0, 0:ny-2, 2:nz]                        # 10 -> 9
        __gi[11,1, 1:ny-1, 1:nz-1]  = __gi[12,0, 2:ny,   0:nz-2]                      # 12 -> 11
        __gi[14,1, 1:ny-1, 1:nz-1]  = __gi[13,0, 2:ny,   2:nz]                        # 13 -> 14

    # right wall (x=nx-2) - no-slip halfway bounce-back
    if x_hi == MPI.PROC_NULL:
        __gi[2, nx-2, :, :]            = __gi[1, nx-1, :, :]                          # 1 -> 2 (axis, no shift)
        # diagonal reflections with horizontal shift
        __gi[8, nx-2, 1:ny-1, 1:nz-1]  = __gi[7, nx-1, 2:ny,   2:nz]                  # 7 -> 8
        __gi[10,nx-2, 1:ny-1, 1:nz-1]  = __gi[9, nx-1, 2:ny,   0:nz-2]                # 9 -> 10
        __gi[12,nx-2, 1:ny-1, 1:nz-1]  = __gi[11,nx-1, 0:ny-2, 2:nz]                  # 11 -> 12
        __gi[13,nx-2, 1:ny-1, 1:nz-1]  = __gi[14,nx-1, 0:ny-2, 0:nz-2]                # 14 -> 13


    # Zhang eq(30) for fi: -> see explanation in bounceBackTopBottom_conservation
    # left wall (x=1) - Zhang eq(30)
    if x_lo == MPI.PROC_NULL:
        __fi[1, 1, :, :] = (1/3)*__fi[1, 2, :, :] + (1/3)*__fi[2, 1, :, :] + (1/3)*__fi[2, 0, :, :]                                        # 2 -> 1
        # diagonal reflections with horizontal shift
        __fi[7, 1, 1:ny-1, 1:nz-1]  = (1/3)*__fi[7, 2, 2:ny-0, 2:nz-0] + (1/3)*__fi[8, 1, 1:ny-1, 1:nz-1] + (1/3)*__fi[8, 0, 0:ny-2, 0:nz-2]          # 8 -> 7
        __fi[9, 1, 1:ny-1, 1:nz-1]  = (1/3)*__fi[9, 2, 2:ny-0, 0:nz-2] + (1/3)*__fi[10, 1, 1:ny-1, 1:nz-1] + (1/3)*__fi[10,0, 0:ny-2, 2:nz-0]          # 10 -> 9
        __fi[11,1, 1:ny-1, 1:nz-1]  = (1/3)*__fi[11,2, 0:ny-2, 2:nz-0] + (1/3)*__fi[12, 1, 1:ny-1, 1:nz-1] + (1/3)*__fi[12,0, 2:ny-0, 0:nz-2]          # 12 -> 11
        __fi[14,1, 1:ny-1, 1:nz-1]  = (1/3)*__fi[14,2, 0:ny-2, 0:nz-2] + (1/3)*__fi[13, 1, 1:ny-1, 1:nz-1] + (1/3)*__fi[13,0, 2:ny-0, 2:nz-0]          # 13 -> 14

    # right wall (x=nx-2) - Zhang eq(30)
    if x_hi == MPI.PROC_NULL:
        __fi[2, nx-2, :, :] = (1/3)*__fi[2, nx-3, :, :] + (1/3)*__fi[1, nx-2, :, :] + (1/3)*__fi[1, nx-1, :, :]                                # 1 -> 2
        # diagonal reflections with horizontal shift
        __fi[8, nx-2, 1:ny-1, 1:nz-1]  = (1/3)*__fi[8, nx-3, 0:ny-2, 0:nz-2] + (1/3)*__fi[7, nx-2, 1:ny-1, 1:nz-1] + (1/3)*__fi[7, nx-1, 2:ny-0, 2:nz-0]  # 7 -> 8
        __fi[10,nx-2, 1:ny-1, 1:nz-1]  = (1/3)*__fi[10,nx-3, 0:ny-2, 2:nz-0] + (1/3)*__fi[9, nx-2, 1:ny-1, 1:nz-1] + (1/3)*__fi[9, nx-1, 2:ny-0, 0:nz-2]  # 9 -> 10
        __fi[12,nx-2, 1:ny-1, 1:nz-1]  = (1/3)*__fi[12,nx-3, 2:ny-0, 0:nz-2] + (1/3)*__fi[11, nx-2, 1:ny-1, 1:nz-1] + (1/3)*__fi[11,nx-1, 0:ny-2, 2:nz-0]  # 11 -> 12
        __fi[13,nx-2, 1:ny-1, 1:nz-1]  = (1/3)*__fi[13,nx-3, 2:ny-0, 2:nz-0] + (1/3)*__fi[14, nx-2, 1:ny-1, 1:nz-1] + (1/3)*__fi[14,nx-1, 0:ny-2, 0:nz-2]  # 14 -> 13

    return __gi, __fi


def init_horizontal_phi(xn, yn, zn, phi_star_g, phi_star_l, height=None, W=4.0):
    """
    Initialize the order parameter _phi as the Zhang eq(9) equilibrium tanh profile:
    a flat horizontal interface at height z0, uniform across x (width) and y (depth).

    Args:
        xn, yn, zn (int): Grid points in width, depth, height directions.
        phi_star_g, phi_star_l (float): Gas/liquid order parameter values.
        height (float, optional): Interface centre z. Defaults to (zn-1)/2.
        W (float): Interface width parameter (vf_W). Controls tanh steepness.

    Returns:
        np.ndarray: 3D array of shape (xn, yn, zn).
    """
    z0 = (zn - 1) / 2 if height is None else height
    x, y, z = np.meshgrid(np.arange(xn), np.arange(yn), np.arange(zn), indexing='ij')
    _phi = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * np.tanh(2 * (z0 - z) / W)

    return _phi


def get_iterations_of_interest(total_iterations, no_slices=51, exp_factor=3.0):
    """
    Generate HIGH granularity iteration indices for 3D reconstruction
    """
    if total_iterations <= 0 or no_slices <= 0:
        return []

    # More frequent early sampling for transient + dense late sampling
    fixed = [0, 50, 100, 200, 250, 300, 350, 400, 450, 500]  # Critical transients
    early_end = int(total_iterations * 0.3)
    
    # Dense linear spacing in early transient
    early_dense = np.linspace(early_end//4, early_end, 15, dtype=int).tolist()
    
    # Exponential spacing in mid-regime
    mid_end = int(total_iterations * 0.7)
    exp_samples = 20
    exp = np.linspace(0, 1, exp_samples + 1)[1:]
    mid_samples = np.floor((np.exp(exp * exp_factor) - 1) / (np.exp(exp_factor) - 1) * 
                          (mid_end - early_end)).astype(int) + early_end
    mid_samples = mid_samples.tolist()
    
    # Dense linear spacing in final convergence
    late_samples = np.linspace(mid_end, total_iterations - 1, 16, dtype=int).tolist()
    
    all_iters = sorted(set(fixed + early_dense + mid_samples + late_samples))
    if len(all_iters) <= no_slices:
        return all_iters
    idx = np.linspace(0, len(all_iters) - 1, no_slices, dtype=int)
    
    return [all_iters[i] for i in idx]


def phi_s(_phi_p, theta, thetaCap, n_dx):
    """
    Compute solid node in wall value __phi_s using Zhang's Eq. (32)
    It's the same as the ghost node
    
    Parameters
    ----------
    phi_f : float or array
        Value(s) at first fluid node(s)
    theta : float
        cos(contact angle) from Zhang Eq. (11)
    dx : float
        lattice spacing
    
    Returns
    -------
    __phi_w : float or array
        Value(s) at wall node
    """
    h = n_dx/2.0
    s = n_dx/2.0
    discriminant = (1.0 + h*thetaCap - np.sqrt( (1.0+h*thetaCap)**2 - 4.0*h*thetaCap*_phi_p) )

    if theta != 90:
        __phi_s = ((s+h)/(2.0*h**2*thetaCap)) * discriminant - (s/h)*_phi_p
    else:
        __phi_s = _phi_p
    
    return __phi_s


def compContactAngle(fc):
    cos_theta = np.cos(fc.vf_theta_rad)
    _compContactAngle = -np.sqrt(2*fc.vf_beta/fc.vf_kappa) * cos_theta
    return _compContactAngle


def set_solid_nodes(fc, _phi):
    #2. viscous force - adaptation of Zhang et al. eq(5), with μ_phi exchanged for μ_c

    # ------- Capillary effect (wetting on bottom, left, and right walls only) -------
    # n = ∇ϕ|∇ϕ∣

    __phi = _phi.copy()

    # The outer ghost ring can hold stale values (periodic np.roll streaming
    # + phi(fc,__gi) reconstruction in bounceBackTopBottom/LeftRight_conservation
    # leaves old data in corners that were never a real phi_p). Mirror each
    # ghost cell from its nearest interior neighbour (zero-gradient/Neumann)
    # so the corner entries phi_s() reads below are physically consistent
    # instead of leftover noise.
    if x_lo == MPI.PROC_NULL:
        __phi[0, :, :]    = __phi[1, :, :]
    if x_hi == MPI.PROC_NULL:
        __phi[local_Xn+1, :, :] = __phi[local_Xn, :, :]
    if y_lo == MPI.PROC_NULL:
        __phi[:, 0, :]    = __phi[:, 1, :]
    if y_hi == MPI.PROC_NULL:
        __phi[:, local_Yn+1, :] = __phi[:, local_Yn, :]
    if z_lo == MPI.PROC_NULL:
        __phi[:, :, 0]    = __phi[:, :, 1]
    if z_hi == MPI.PROC_NULL:
        __phi[:, :, local_Zn+1] = __phi[:, :, local_Zn]
  

    # Contact angle (same for all solid walls — modify later if needed)
    thetaCap = compContactAngle(fc)

    # === Top wall (z=Zn) ===
    _phi_p_n_top = __phi[:, :, local_Zn]                    # 1. fluid node inside
    _phi_s_top = phi_s(_phi_p_n_top, fc.vf_theta, thetaCap, n_dz)

    # === Bottom wall (z=1) ===
    _phi_p_n_bottom = __phi[:, :, 1]                    # 1. fluid node inside
    _phi_s_bottom = phi_s(_phi_p_n_bottom, fc.vf_theta, thetaCap, n_dz)


    # === Left wall (x=1) ===
    _phi_p_n_left = __phi[1, :, :]                    # 1. fluid node inside
    _phi_s_left = phi_s(_phi_p_n_left, fc.vf_theta, thetaCap, n_dx)

    # === Right wall (x=Xn) ===
    _phi_p_n_right = __phi[local_Xn, :, :]                    # 1. fluid node inside
    _phi_s_right = phi_s(_phi_p_n_right, fc.vf_theta, thetaCap, n_dx)


    # === Front wall (x=1) ===
    _phi_p_n_front = __phi[:, 1, :]                    # 1. fluid node inside
    _phi_s_front = phi_s(_phi_p_n_front, fc.vf_theta, thetaCap, n_dy)                   

    # === Back wall (x=Yn) ===
    _phi_p_n_back = __phi[:, local_Yn, :]                    # 1. fluid node inside
    _phi_s_back = phi_s(_phi_p_n_back, fc.vf_theta, thetaCap, n_dy)                               


    # Step 3: Assign to solid wall nodes only
    # Bottom wall
    if z_lo == MPI.PROC_NULL:
        __phi[:, :, 0]  = _phi_s_bottom   # bottom

    # Top wall
    if z_hi == MPI.PROC_NULL:
        __phi[:, :, local_Zn+1] = _phi_s_top  # current x=Xn+1      


    # Left wall
    if x_lo == MPI.PROC_NULL:
        __phi[0, :, :] = _phi_s_left   # current x=0

    # Right wall
    if x_hi == MPI.PROC_NULL:
        __phi[local_Xn+1, :, :] = _phi_s_right  # current x=Xn+1  


    # Front wall
    if y_lo == MPI.PROC_NULL:
        __phi[:, 0, :] = _phi_s_front   # current x=0

    # Back wall
    if y_hi == MPI.PROC_NULL:
        __phi[:, local_Yn+1, :] = _phi_s_back  # current x=Xn+1      

    return __phi


def bodyForce(fc, rho):
    # subtract fc.rho_G from rho in order to model the pressure gradient acting on the air
    _bodyForce = fc.F_body[:, None, None, None] * (rho - fc.rho_G)

    return _bodyForce


def apply_mass_conservation(phi_old_total, __phi):
    # CRITICAL: Restore exact mass conservation after overwrite
    # Compute total ϕ in fluid domain AFTER wetting
    local_phi_total_after = np.sum(__phi[1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1])
    phi_total_after = comm.allreduce(local_phi_total_after, op=MPI.SUM)

    # Scale ONLY the fluid domain back to pre-wetting total (very small correction)
    if fc.ENFORCE_MASS_CONSERVATION and abs(phi_total_after - phi_old_total) > 1e-8:  # avoid div-by-zero or noise
        scale_factor = phi_old_total / phi_total_after
        __phi[1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1] *= scale_factor

        # Optional print to monitor (remove later)
        if iteration % 500 == 0 and rank == 0:
            print(f"          | Wetting overwrote → applied scale {scale_factor:.10f} "
                f"(delta before scale: {phi_total_after - phi_old_total:+.8f})")

    return __phi


# ──────────────────────────────────────────────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────────────────────────────────────────────
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)

#average velocity, cartesion x,y-directions, k is y-position, l is x-position
u_ckl = np.zeros((3, local_Xn+2, local_Yn+2, local_Zn+2), dtype=np.float64)
INIT_RHO = 1 #0.001
rho = np.full((local_Xn+2, local_Yn+2, local_Zn+2), INIT_RHO, dtype=np.float64)

# Simulation parameters
start = time.perf_counter()

PARAMETER_STUB = "__" + ACTIVE_CASE + "__nodes_" + str(DEFAULT_D_ND) + "__tau_f_" + str(fc.tau_f) + "__tau_g_" + str(fc.tau_g) \
    + "__Kf_" + str(fc.Kf) + "__Theta_" + str(fc.vf_theta) + "_alpha" + f"{fc.alpha:g}" + "_sigma" + f"{fc.vf_sigma:g}" \
    + ("_rhoG" + f"{fc.rho_G:g}" if fc.rho_G != 0.01 else "") \
    + ("_rhoL" + f"{fc.rho_L:g}" if fc.rho_L != 10.0 else "") \
    + ("_W" + str(fc.vf_W) if fc.vf_W != 4 else "") \
    + ("_nu" + f"{nu:g}" if nu != 2.9e-6 else "") \
    + ("_D" + f"{D*1e3:g}" if D != 1e-3 else "") \
    + ("_fill" + f"{FILL:g}" if FILL != 38.0 else "") \
    + ("_periodicx" if PERIODIC_X else "") \
    + ("_x" + str(fc.MULTIPLES) if fc.MULTIPLES != 1 else "")

images_dir = os.path.join(images_subdir, SCRIPT_FILENAME + PARAMETER_STUB)
os.makedirs(images_dir, exist_ok=True)

#phi initialisation
if PHI_DISTRIBUTION == "HORIZONTAL":
    _phi = init_horizontal_phi(local_Xn+2, local_Yn+2, local_Zn+2, fc.phi_star_G, fc.phi_star_L, height=FILL + 0.5 - z_offset, W=fc.vf_W)
    density_profile_x_position = Xn//2
    density_profile_y_position = Yn//2
    density_profile_z_position = Zn//2


PhiCollector = []
SurfaceHeight = []

iteration = 0

#gather helper
plane_x = (Xn+1)//2
plane_x_is_local = (x_offset < plane_x <= x_offset + local_Xn)
plane_x_local = plane_x - x_offset

plane_y = (Yn+1)//2
plane_y_is_local = (y_offset < plane_y <= y_offset + local_Yn)
plane_y_local = plane_y - y_offset

# rolling crash buffer: last RING_N iterations of the mid-x yz-plane, overwritten in place
RING_N = 1500
if plane_x_is_local:
    os.makedirs(os.path.join(images_dir, "ring"), exist_ok=True)
    ring = np.lib.format.open_memmap(
        os.path.join(images_dir, "ring", f"ring_y{y_offset}_z{z_offset}.npy"),
        mode="w+", dtype=np.float32, shape=(RING_N, 5, local_Yn, local_Zn))
    ring[:, 4, 0, 0] = -1   # channel 4 holds the iteration number of each slot

# rolling crash buffer of the closed-x corner (high-x wall, front wall y=0, floor z=0): whole local blocks of the ranks touching it
CORNER_N = 1000
corner_is_local = (not PERIODIC_X) and (x_offset + local_Xn == Xn) and (y_offset < 2*local_Yn) and (z_offset == 0)
if corner_is_local:
    os.makedirs(os.path.join(images_dir, "ring"), exist_ok=True)
    corner_ring = np.lib.format.open_memmap(
        os.path.join(images_dir, "ring", f"corner_y{y_offset}.npy"),
        mode="w+", dtype=np.float32, shape=(CORNER_N, 5, local_Xn, local_Yn, local_Zn))
    corner_ring[:, 4, 0, 0, 0] = -1   # channel 4 holds the iteration number of each slot    

u_ckl = np.zeros((3, local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)
p = np.zeros((local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)
z_gi_c = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)
z_gi = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)
# Before the main loop
z_gi = zgi_c(fc, np.zeros_like(u_ckl), _phi, out=z_gi)

z_fi_c = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)
z_fi = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2),dtype=np.float64)

_z_star_buf = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2), dtype=np.float64)
_Fi_buf = np.zeros((15, local_Xn+2, local_Yn+2, local_Zn+2), dtype=np.float64)
_Fs_buf = np.zeros((3, local_Xn+2, local_Yn+2, local_Zn+2), dtype=np.float64)

__phi_old = np.copy(_phi) 
_u_ckl_old = np.copy(u_ckl)

rho = density(fc, _phi)

# floor strip, one frame per iteration, in the windows below: mid-x plane (x = 384), y < 64, z < 8; phi and u; one file per rank block
STRIP_WINDOWS = [(0, 3000), (24000, 27000)]
strip_is_local = plane_x_is_local and (z_offset == 0) and (y_offset < 64)
if strip_is_local:
    strip_ny = min(local_Yn, 64 - y_offset)
    strip_buf = np.zeros((sum(b - a for a, b in STRIP_WINDOWS), 4, strip_ny, 8), dtype=np.float32)
    strip_iter = np.full(strip_buf.shape[0], -1, dtype=np.int32)
    strip_n = 0


iterationsOfInterest = get_iterations_of_interest(fc.TOTAL_ITERATIONS, no_slices=51*fc.MULTIPLES, exp_factor=4.0)
_ioi = sorted(iterationsOfInterest)
snapshot_iterations = set(_ioi[int(k)] for k in np.linspace(0, len(_ioi) - 1, 10))
# ──────────────────────────────────────────────────────────────────────────────────────────
# simulation
# ──────────────────────────────────────────────────────────────────────────────────────────


while iteration < fc.TOTAL_ITERATIONS:
    _t = {}
    def _mark(label, t0):
        _t[label] = _t.get(label, 0) + (time.perf_counter() - t0)

    # ──────────────────────────────────────────────────────────────
    #          Forces: Body and Surface Tension
    # ──────────────────────────────────────────────────────────────
    # 1. Body mass force
    # variable G in eq(20)
    body_force = bodyForce(fc, rho) * n_dt
    assert body_force.shape == (3, local_Xn+2, local_Yn+2, local_Zn+2), f"body_force shape: {body_force.shape}"

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang functions: 2.Equilibrium , 3.Collision/Relaxation, 4.Streaming
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 1. Compute equilibrium distributions (fi_c, gi_c) ===
    # Zhang eq(17):  equilibrium distribution function for order parameter
    _t0 = time.perf_counter(); 
    z_gi_c = zgi_c(fc, u_ckl, _phi, out=z_gi_c); 
    _mark('zgi_c', _t0)

    # Zhang eq(18):  equilibrium distribution function for pressure distribution function
    _t0 = time.perf_counter(); 
    zfi_c(fc, u_ckl, rho, p, out=z_fi_c); 
    _mark('zfi_c', _t0)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang functions: 3.Collision/Relaxation, 4.Streaming (advection)
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 3. COLLISION + STREAMING (inside fi and gi) ===    
    # Zhang eq(2): calculation of the order parameter which distiguishes the two phases
    _t0 = time.perf_counter(); 
    z_gi  = zgi(fc, z_gi, z_gi_c, __phi_old, _u_ckl_old, _phi, u_ckl); 
    _mark('zgi', _t0)

    _t0 = time.perf_counter(); 
    _Fs = Fs(fc, _phi, out=_Fs_buf); 
    _mark('Fs_1', _t0)
    if iteration % 100 == 0:
        assert _Fs.shape == (3, local_Xn+2, local_Yn+2, local_Zn+2), f"_Fs shape: {_Fs.shape}"
        assert _Fs.shape == body_force.shape, f"_Fs {_Fs.shape} != body_force {body_force.shape}"
    # Zhang eq(2): calculation of the pressure distribution function
    _t0 = time.perf_counter(); 
    z_fi = zfi(fc, z_fi, z_fi_c, u_ckl, rho, _Fs, body_force); 
    _mark('zfi', _t0)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # 5. Boundary conditions
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 2. UPDATE GHOST NODES FROM CURRENT POST-COLLISION STATE ===
    # Use _fi_c and _gi_c (post-collision, pre-streaming)

    # 4. top/bottom conservative bounceback
    _t0 = time.perf_counter(); 
    z_gi, z_fi = bounceBackTopBottom_conservation(z_gi, z_fi, local_Xn+2, local_Yn+2, local_Zn+2); 
    _mark('bounceTB', _t0)

    #4.1b. left/right conservative bounceback
    _t0 = time.perf_counter(); 
    z_gi, z_fi = bounceBackLeftRight_conservation(z_gi, z_fi, local_Xn+2, local_Yn+2, local_Zn+2); 
    _mark('bounceLR', _t0)

    #4.1c. front/back conservative bounceback
    _t0 = time.perf_counter(); 
    z_gi, z_fi = bounceBackFrontBack_conservation(z_gi, z_fi, local_Xn+2, local_Yn+2, local_Zn+2); 
    _mark('bounceFB', _t0)


    # ──────────────────────────────
    # NEW: Measure ϕ mass from previous step
    local_phi_old_total = np.sum(_phi[1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1])
    phi_old_total = comm.allreduce(local_phi_old_total, op=MPI.SUM)
    __phi_old = np.copy(_phi) 
    _u_ckl_old = np.copy(u_ckl) 
    # ──────────────────────────────     

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # 1. Moment update
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang eq(26): zeroth-order moment, calculation of order parameter to distiguish the 2 phases
    _t0 = time.perf_counter(); 
    _phi = phi(fc, z_gi); 
    _mark('phi_moment', _t0)

    # ──────────────────────────────
    # Mass conservation diagnostoics
    # ──────────────────────────────   
    _t0 = time.perf_counter(); 
    _phi = apply_mass_conservation(phi_old_total, _phi); 
    _mark('apply_mass_conservation', _t0)
    # ──────────────────────────────  
    exchange_ghosts_x(_phi, x_lo, x_hi, cart)
    exchange_ghosts_y(_phi, y_lo, y_hi, cart)
    exchange_ghosts_z(_phi, z_lo, z_hi, cart)

    # ──────────────────────────────────────────────────────────────
    #          Forces: Surface Tension
    # ──────────────────────────────────────────────────────────────
    # Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    # Zhang  eq(20): Compute u(x,t+n_dt)
    # Kürger et al, p. 241 eq. (6.29) & Table 6.1

    _t0 = time.perf_counter(); 
    _phi  = set_solid_nodes(fc, _phi); 
    _mark('set_solid_nodes', _t0)

    #Calculation of rho, mu
    _t0 = time.perf_counter(); 
    rho = density(fc, _phi); 
    _mark('density', _t0)

    exchange_ghosts_x(rho, x_lo, x_hi, cart)
    exchange_ghosts_y(rho, y_lo, y_hi, cart)
    exchange_ghosts_z(rho, z_lo, z_hi, cart)

    # Zhang eq(5): Fs is the surface tension force, expressed in a potential form
    _t0 = time.perf_counter(); 
    _Fs = Fs(fc, _phi, out=_Fs_buf); 
    _mark('Fs_2', _t0)

    # --> Zhang eq(28) - fluid velocity
    _t0 = time.perf_counter(); 
    u_ckl = zu_ckl(fc, z_fi, rho, body_force, _Fs); 
    _mark('zu_ckl', _t0)
       

    #Step2a. calculate h, p
    # Zhang eq(27) - hydrostatic pressure
    ############################ 2. Add Successive Over-Relaxation (SOR) with Residual Monitoring (Accelerate/Damp) #################
    _t0 = time.perf_counter(); 
    zp(fc, z_fi, rho, u_ckl, out=p); 
    _mark('zp', _t0)  # Use damped div_u if added above
    # Use damped div_u if added above
    #################################################################################################################################

    
    if ADD_METRICS and iteration in iterationsOfInterest:
        #  NEW – collect vertical integrals of φ
        local_phi_total  = np.sum(_phi[1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1])
        phi_total = comm.allreduce(local_phi_total, op=MPI.SUM)
        PhiCollector.append((iteration, phi_total))

        # dump the diagnostic lists to disk every checkpoint (overwrites,
        # always holds the full history so far) - avoids needing to raid
        # the live process's memory to see them mid-run
        np.savez_compressed(
            os.path.join(images_dir, "diagnostic_lists.npz"),
            PhiCollector=np.array(PhiCollector),
        )

        if plane_x_is_local:
            local_plane_piece = _phi[plane_x_local, 1:local_Yn+1, 1:local_Zn+1]
            payload = (y_offset, z_offset, local_plane_piece)
        else:
            payload = None

        gathered = comm.gather(payload, root=0)

        if rank == 0:
            phi_on_plane = np.zeros((Yn, Zn))
            for item in gathered:
                if item is not None:
                    y_off, z_off, piece = item
                    phi_on_plane[y_off:y_off+local_Yn, z_off:z_off+local_Zn] = piece
            _mask = phi_on_plane > 0.5
            _k = Zn - 1 - np.argmax(_mask[:, ::-1], axis=1)
            _i = np.arange(Yn)
            _p0 = phi_on_plane[_i, _k]
            _p1 = phi_on_plane[_i, np.minimum(_k + 1, Zn - 1)]
            _h = _k + (_p0 - 0.5) / np.where(_p0 == _p1, 1.0, _p0 - _p1)
            SurfaceHeight.append(np.concatenate(([iteration], _h)))
            np.savez_compressed(os.path.join(images_dir, "surface_height.npz"), SurfaceHeight=np.array(SurfaceHeight))

        _mu_phi = chemical_potential(fc, _phi)
        if plane_x_is_local:
            payload_u = (y_offset, z_offset,
                         u_ckl[:, plane_x_local, 1:local_Yn+1, 1:local_Zn+1].copy(),
                         _mu_phi[plane_x_local, 1:local_Yn+1, 1:local_Zn+1].copy(),
                         _Fs[:, plane_x_local, 1:local_Yn+1, 1:local_Zn+1].copy())
        else:
            payload_u = None
        gathered_u = comm.gather(payload_u, root=0)

        if rank == 0:
            u_on_plane = np.zeros((3, Yn, Zn), dtype=np.float32)
            mu_on_plane = np.zeros((Yn, Zn), dtype=np.float32)
            Fs_on_plane = np.zeros((3, Yn, Zn), dtype=np.float32)
            for item in gathered_u:
                if item is not None:
                    y_off, z_off, piece, mu_piece, Fs_piece = item
                    u_on_plane[:, y_off:y_off+local_Yn, z_off:z_off+local_Zn] = piece
                    mu_on_plane[y_off:y_off+local_Yn, z_off:z_off+local_Zn] = mu_piece
                    Fs_on_plane[:, y_off:y_off+local_Yn, z_off:z_off+local_Zn] = Fs_piece
            np.savez_compressed(os.path.join(images_dir, f"plane_yz_iter_{iteration:06d}.npz"),
                                phi=phi_on_plane.astype(np.float32), u=u_on_plane,
                                mu=mu_on_plane, Fs=Fs_on_plane)
        if plane_y_is_local:
            payload_xz = (x_offset, z_offset,
                          _phi[1:local_Xn+1, plane_y_local, 1:local_Zn+1].copy(),
                          u_ckl[:, 1:local_Xn+1, plane_y_local, 1:local_Zn+1].copy())
        else:
            payload_xz = None
        gathered_xz = comm.gather(payload_xz, root=0)

        if rank == 0:
            phi_xz = np.zeros((Xn, Zn), dtype=np.float32)
            u_xz = np.zeros((3, Xn, Zn), dtype=np.float32)
            for item in gathered_xz:
                if item is not None:
                    x_off, z_off, phi_piece, u_piece = item
                    phi_xz[x_off:x_off+local_Xn, z_off:z_off+local_Zn] = phi_piece
                    u_xz[:, x_off:x_off+local_Xn, z_off:z_off+local_Zn] = u_piece
            np.savez_compressed(os.path.join(images_dir, f"plane_xz_iter_{iteration:06d}.npz"), phi=phi_xz, u=u_xz)

        if iteration in snapshot_iterations:
            phi_full = gather_full_volume(_phi, x_offset, y_offset, z_offset,
                                          local_Xn, local_Yn, local_Zn, Xn, Yn, Zn, dtype=np.float32)
            if rank == 0:
                np.savez_compressed(os.path.join(images_dir, f"phi3d_iter_{iteration:06d}.npz"),
                                    phi=phi_full[1:Xn+1:2, 1:Yn+1:2, 1:Zn+1:2].astype(np.float16))
                del phi_full            

    #streaming has commenced

    if plane_x_is_local:
        _slot = iteration % RING_N
        ring[_slot, 0] = _phi[plane_x_local, 1:local_Yn+1, 1:local_Zn+1]
        ring[_slot, 1:4] = u_ckl[:, plane_x_local, 1:local_Yn+1, 1:local_Zn+1]
        ring[_slot, 4, 0, 0] = iteration

    if strip_is_local and any(a <= iteration < b for a, b in STRIP_WINDOWS):
        strip_buf[strip_n, 0] = _phi[plane_x_local, 1:strip_ny+1, 1:9]
        strip_buf[strip_n, 1:4] = u_ckl[:, plane_x_local, 1:strip_ny+1, 1:9]
        strip_iter[strip_n] = iteration
        strip_n += 1
        if (iteration + 1) in [b for a, b in STRIP_WINDOWS]:      # a window just ended: write everything collected so far
            np.savez(os.path.join(images_dir, f"floor_strip_y{y_offset}.npz"), frames=strip_buf[:strip_n], iteration=strip_iter[:strip_n])

    if corner_is_local:
        _cslot = iteration % CORNER_N
        corner_ring[_cslot, 0] = _phi[1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1]
        corner_ring[_cslot, 1:4] = u_ckl[:, 1:local_Xn+1, 1:local_Yn+1, 1:local_Zn+1]
        corner_ring[_cslot, 4, 0, 0, 0] = iteration        

    #Step 4: re-iterate
    iteration += 1

    if rank == 0 and iteration % 1000 == 0:
        print(f"--- timings, iteration {iteration} ---")
        for _k, _v in sorted(_t.items(), key=lambda kv: -kv[1]):
            print(f"  {_k}: {_v:.3f}s")

    if ADD_METRICS and iteration in iterationsOfInterest:
        _ui = np.abs(u_ckl[:, 1:-1, 1:-1, 1:-1])
        _c, _ix, _iy, _iz = np.unravel_index(np.argmax(_ui), _ui.shape)
        u_max, _owner = comm.allreduce((float(_ui[_c, _ix, _iy, _iz]), rank), op=MPI.MAXLOC)
        phi_min = comm.allreduce(np.min(_phi[1:-1, 1:-1, 1:-1]), op=MPI.MIN)
        phi_max = comm.allreduce(np.max(_phi[1:-1, 1:-1, 1:-1]), op=MPI.MAX)
        if rank == 0:
            print(f"iter {iteration:6d}  t={iteration*fc.dt*1e3:8.3f} ms  max|u|={u_max:.4e}  Ma={u_max/Cs:.4f}  phi=[{phi_min:.4f}, {phi_max:.4f}]", flush=True)

        if rank == _owner:
            print(f"      max|u| at x={x_offset + _ix}, y={y_offset + _iy}, z={z_offset + _iz} (component {'xyz'[_c]}), phi there = {_phi[_ix + 1, _iy + 1, _iz + 1]:.4f}", flush=True)            

end = time.perf_counter()
diff = end - start

if rank == 0:
    print(f"Run-time: {diff:.1f} s", flush=True)