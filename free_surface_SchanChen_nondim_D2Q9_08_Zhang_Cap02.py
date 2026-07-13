#this simulation a microchannel flow is modelled along a plane inserted vertically and symetrically in the channel center#aligned to the z-axis vertically and in the x-axis direction along hte channel length
#the axes in the plane are y-axis in the vertical direction and x-axis in the channel horizontal direction
#The simulation is based on the paper by Zhang et al, 2020, Physics of Fluids 32, 103301 (2020)
# Add at top of script (Python 3.7+ for forward refs in annotations)
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # or 'Qt5Agg' if you have PyQt installed
#matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple  # Import these for proper type hints

import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from github_uploader import GitHubUploader
from plotter_2d import Plotter2D

import sys
import logging

# Force stdout and stderr to use UTF-8 (Windows fix)
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


@dataclass
class FlowConfig:
    # --- Required per-scenario fields ---
    name: str
    n_dx: float
    dx: float
    dt: float
    CF: float

    # --- Scenario-specific parameters ---
    a: float
    b: float
    T: float
    tau_f: float
    tau_g: float
    Kf: float
    Kg: float

    # --- viscous forces ---
    # Zhang interface thickness    
    '''W - the measurement of the interface thickness'''
    vf_W: int
    '''σ - surface tension coefficient'''
    # Zhang surface tension coefficient, A. Wetting behavior of a particle on the liquid–gas interface
    vf_sigma: float
    '''θ - equilibrium contact angle'''
    vf_theta: float
    '''cap force multiplier'''
    vf_capillaryForceMultiplier: float

    # --- Defaults ---
    epsilon_cutoff: float = 1e-4        # convergence tolerance
    alpha: float = 0.0                  # inclination of channel (degrees)
    # Generous but cap to avoid infinite
    # Successive Over-Relaxation (SOR) factor
    omega_sor: float = 1.2  # 1.2-1.8 optimal for LBM Poisson; start 1.5

    # --- Default physical constants ---
    g: float = 9.81
    rho_G: float = 0.01
    rho_L: float = 10.0
    phi_star_G: float = 0
    phi_star_L: float = 1

    # --- Iteration / simulation control (with defaults) ---
    MULTIPLES: int = 1
    CORE_TOTAL_ITERATIONS: int = 12001
    ADD_BODY_FORCE: int = 1
    ADD_SURFACE_TENSION_FORCE: int = 1
    WRITE_TO_GITHUB: bool = False
    ENFORCE_MASS_CONSERVATION: bool = True
    ENABLE_PHI_CLIPPING: bool = True

    # --- Derived quantities ---
    @property
    def TOTAL_ITERATIONS(self) -> int:
        """Total number of simulation iterations."""
        return int(self.CORE_TOTAL_ITERATIONS * self.MULTIPLES)

    @property
    def FILENAME_PADDING_WIDTH(self) -> int:
        """Digits needed for zero-padded filenames."""
        return int(np.ceil(np.log10(self.TOTAL_ITERATIONS + 1)))

    @property
    def NO_DATA_DUMP_SLICES(self) -> int:
        """Number of data output slices during the run."""
        return int(51 * self.MULTIPLES)

    @property
    def mu_G(self) -> float:
        """Gas-phase dynamic viscosity (lattice units)."""
        return 0.007 * self.n_dx

    @property
    def mu_L(self) -> float:
        """Liquid-phase dynamic viscosity (lattice units)."""
        return 0.5 * self.n_dx

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
        return -self.g_lattice * np.cos(self.alpha_rad)

    @property
    def F_body(self) -> np.ndarray:
        """Body force vector in world coordinates."""
        return np.array([self.g_x, self.g_y])

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
PRESSURE_IN_DENSITY_MAP = False
PLOTREALTIME = False  
ADD_METRICS = True
ZERO_BCs = False

# Constants
DEFAULT_D_ND = 300
SCRIPT_FILENAME = os.path.splitext(os.path.basename(__file__))[0] 
SCRIPT_FULL_PATH = os.path.abspath(__file__) 
SCRIPTS_PATH = "scripts/freesurface/"
PLOTS_PATH = "results/freesurface/"  # GitHub path prefix
IMAGES_SUBDIR = "FreesurfaceImages"  # Local subdir
script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
images_subdir = os.path.join(script_dir, IMAGES_SUBDIR)
os.makedirs(images_subdir, exist_ok=True)  # create folder if it doesn't exist  
# Create the specific subdirectory
zhang_subdir = os.path.join(images_subdir, "zhang_surface_tension_force_new")
os.makedirs(zhang_subdir, exist_ok=True)
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
    iter_label = iter if iter is not None else 'N/A'

    if np.any(np.isinf(field)):
        # Checked before isnan deliberately: overflow produces inf first,
        # and it only becomes nan a step later (e.g. inf-inf, 0*inf) in some
        # downstream op. Catching inf here traps the true origin instead of
        # wherever it first got combined into a nan.
        finite = field[np.isfinite(field)]
        min_val = np.min(finite) if finite.size else float('nan')
        max_val = np.max(finite) if finite.size else float('nan')
        n_pos_inf = np.sum(np.isposinf(field))
        n_neg_inf = np.sum(np.isneginf(field))
        debug_log('ERROR', f'Inf in {name} at iter {iter_label}: +inf count={n_pos_inf}, '
                  f'-inf count={n_neg_inf}, finite min={min_val:.3e}, finite max={max_val:.3e}')
        raise ValueError(f'Inf in {name} (+inf: {n_pos_inf}, -inf: {n_neg_inf})')

    if np.any(np.isnan(field)):
        min_val, max_val = np.min(field), np.max(field)  # Still compute for context
        debug_log('ERROR', f'NaN in {name} at iter {iter_label}: min={min_val:.3e}, max={max_val:.3e}')
        raise ValueError(f'NaN in {name}')

    if not allow_neg and np.any(field < 0):
        min_val = np.min(field)
        debug_log('ERROR', f'Negative in {name} at iter {iter_label}: min={min_val:.3e}')
        raise ValueError(f'Negative in {name}')

    if allow_range:
        min_req, max_req = allow_range
        invalid = (field <= min_req) | (field >= max_req)
        if np.any(invalid):
            min_val, max_val = np.min(field), np.max(field)
            invalid_idx = np.where(invalid)
            debug_log('ERROR', f'Out-of-range {name} at iter {iter_label}: min={min_val:.3e}, '
                      f'max={max_val:.3e}, must be in ({min_req:.3e}, {max_req:.3e}). '
                      f'Invalid indices: {invalid_idx}')
            raise ValueError(
                f"Invalid {name}: min={min_val:.3e}, max={max_val:.3e}, "
                f"must be in ({min_req:.3e}, {max_req:.3e}). Invalid indices: {invalid_idx}"
            )
       
######################################################################################################################

# ──────────────────────────────────────────────────────────────────────────────────────────
# lattice parameters
# ──────────────────────────────────────────────────────────────────────────────────────────

Cs=np.sqrt(1/3)
Cs2 = Cs**2
Cs4 = Cs**4

D=1e-3 #m
L=1 #m

Yn=int(DEFAULT_D_ND ) #+1
Xn=int(DEFAULT_D_ND ) #200 #int(Yn*L/D)

dx=D/DEFAULT_D_ND  #old->5*10**(-5)
dy = dx
#relaxation time n_tau, should be > 0,5
n_tau = 0.6

dP=0 #Pa
rho_0=1e3 #kg/m^3
dRho=dP/Cs2

nu=2.9e-6 #m^2/s => in OLB this is 1/Re, with Re=148. So Re must become Re= in order to conform with this simulation
dt = Cs2*(n_tau-0.5)*(dx**2/nu)

debug_log('INIT', 'Yn={Yn}, Xn={Xn}, dx={dx:.3e}, dy={dy:.3e}, dt={dt:.3e}')

#Assume U at centerline (max) velocity
U=0.0
Re=D*U/nu
Ma=U/Cs 
Kn=U*D/nu
debug_log('INIT', 'U=%(U).2f, Re=%(Re).2f, Ma=%(Ma).3e, Kn=%(Kn).3e', extra=dict(U=U, Re=Re, Ma=Ma, Kn=Kn))

#we need Cl, Crho, Ct
# 1. Conversion factor Cl for length
Cl = dx #freely chosen
n_dx = dx/Cl #-> dx_nd=1
n_dy = n_dx
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
#nu_nd=((DEFAULT_D_ND*U_nd)/(D*U))*nu
#CAPILLARY FLOW -> U=0 -> remove U
nu_nd = Cs2 * (n_tau - 0.5)
nu_nd = 0.0145
debug_log('INIT', 'nu_nd=%(nu_nd).2f', extra=dict(nu_nd=nu_nd))


tau_nd=(nu_nd/Cs2)+1./2
debug_log('INIT', 'tau_nd=%(tau_nd).2f', extra=dict(tau_nd=tau_nd))

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

# Zhang bottom p. 32, 2nd column constant w0 & Krüger: force weights
E = np.array([4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36])

c_x_exp = c[:, 0][:, np.newaxis, np.newaxis]  # (9, 1, 1)
c_y_exp = c[:, 1][:, np.newaxis, np.newaxis]
# To match original exactly (including the buggy du+du), broadcast per-component on full grid
c_x_sq = c[:, 0][:, np.newaxis, np.newaxis] ** 2  # (9,1,1)
c_y_sq = c[:, 1][:, np.newaxis, np.newaxis] ** 2
c_xy = (c[:, 0] * c[:, 1])[:, np.newaxis, np.newaxis]  # (9,1,1)
# Expansions for broadcasting over i
E_exp = E[:, np.newaxis, np.newaxis]  # (9,1,1)
c_exp = c[:, :, np.newaxis, np.newaxis]  # (9,2,1,1)
debug_log('INIT', 'c=%(c).2f', extra=dict(c=c))


# ──────────────────────────────────────────────────────────────────────────────────────────
# Configurations
# ──────────────────────────────────────────────────────────────────────────────────────────
# 0.1 Proof of Principle (POP) - STEP
PROOF_STEP1 = FlowConfig(
    name="Proof: Step1",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=2.5, tau_g=1.0,
    Kf = 1e-6 * n_dx**2, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=1,
    vf_capillaryForceMultiplier=1.0,
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0    
)

PROOF_STEP2 = FlowConfig(
    name="Proof: Step2",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.5, tau_g=1.0,
    #Kf=0.008, Kg=1e-6 * n_dx**2,
    Kf=0.0008, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8*8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP3 = FlowConfig(
    name="Proof: Step3",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=2.5, tau_g=1.0,
    Kf=0.008, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8*8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP4 = FlowConfig(
    name="Proof: Step4",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.0, tau_g=1.0,
    Kf=0.008, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP5 = FlowConfig(
    name="Proof: Step5",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=0.9, tau_g=1.0,
    Kf=0.008, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP6 = FlowConfig(
    name="Proof: Step6",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.5, tau_g=1.0,
    Kf=0.008, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES= 1, #8*4
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 0.0
)

PROOF_STEP7 = FlowConfig(
    name="Proof: Step7",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=2.5, tau_g=1.0,
    Kf=0.05, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8*8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP8 = FlowConfig(
    name="Proof: Step8",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=2., tau_g=1.0,
    Kf=0.05, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8*8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

PROOF_STEP9 = FlowConfig(
    name="Proof: Step9",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.8, tau_g=1.0,
    Kf=0.05, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=8*8,
    vf_capillaryForceMultiplier=1.0,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

# 0.2 Proof of Principle (POP) - inclined channel
PROOF_INCLINED = FlowConfig(
    name="Proof: Inclined channel",
    dx=dx, dt=dt,
    n_dx=n_dx, alpha=30, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=2.5, tau_g=1.0,
    Kf=0.005, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    MULTIPLES=1,
    vf_capillaryForceMultiplier=1.0,      
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 0.0
)

#CAPILLARY FLOW -> vf_theta == 0°
CAPILLARY_PROOF = FlowConfig(
    name="Capillary Proof",
    dx=dx, dt=dt,
    n_dx=n_dx, alpha=0, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    #tau_f=1.5, tau_g=1.0,
    tau_g=0.65, tau_f=1.25, #see eq(25) + Example A , t_g start with 0.6, increase as necessary -> was tau_f=0.65
    #Kf=0.5 * dx**2, Kg=1e-5 * dx**2,
    #Kf=0.5*dx**2 -> 0.08*dx**2
    #Kg=2.5e-4*dx**2 -> 1e-5*dx**2 -> 1e-7*dx**2
    Kf=0.002, Kg=1e-6 * n_dx**2,
    epsilon_cutoff=1e-4,
    #Increase interface smoothness: Set vf_W = 6 or 8 in FlowConfig to widen the diffuse interface, reducing sharp edges.
    vf_W = 4, #was 6
    vf_sigma = 0.01, #0.072
    vf_theta = 120.0, #60
    vf_capillaryForceMultiplier=1,
    MULTIPLES=1,
    ENFORCE_MASS_CONSERVATION = True,
    ADD_SURFACE_TENSION_FORCE = 1,
    ADD_BODY_FORCE = 1
)

# §3.1 Capillary wave
CAPILLARY_WAVE = FlowConfig(
    name="Capillary wave",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.0, tau_g=1.0,
    Kf=0.5 * n_dx**2, Kg=0.0,
    epsilon_cutoff=1e-4,
    vf_capillaryForceMultiplier=1,
    MULTIPLES=1,    
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

# §3.2 Droplet collision
DROPLET_COLLISION = FlowConfig(
    name="Droplet collision",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=6.7, T=3.5e-2,
    tau_f=1.0, tau_g=1.0,
    Kf=0.5 * n_dx**2, Kg=20.0,
    epsilon_cutoff=1e-4,
    vf_capillaryForceMultiplier=1,
    MULTIPLES=1,        
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)

# §3.3 Bubble flow
BUBBLE_FLOW = FlowConfig(
    name="Bubble flow",
    dx=dx, dt=dt,
    n_dx=n_dx, CF=CF,
    a=1.0, b=1.0, T=2.93e-1,
    tau_f=1.0, tau_g=1.0,
    Kf=0.08 * n_dx**2, Kg=1e-7 * n_dx**2,
    epsilon_cutoff=1e-5,
    vf_capillaryForceMultiplier=1,
    MULTIPLES=1,        
    vf_W = 4,
    vf_sigma = 0.001,
    vf_theta = 60.0
)


USE_CASES = {
    "proof_step": {"PHI_DISTRIBUTION": "STEP", "fc": PROOF_STEP6},
    "proof_inclined": {"PHI_DISTRIBUTION": "HORIZONTAL", "fc": PROOF_INCLINED},
    "proof_capillary": {"PHI_DISTRIBUTION": "HORIZONTAL", "fc": CAPILLARY_PROOF}
}

ACTIVE_CASE = "proof_capillary"
PHI_DISTRIBUTION = USE_CASES[ACTIVE_CASE]["PHI_DISTRIBUTION"]
fc = USE_CASES[ACTIVE_CASE]["fc"]  # current FlowConfig

# CLI overrides for param_study.py sweeps (tau_g, tau_f, vf_theta)
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument('--tau_g', type=float, default=None)
_arg_parser.add_argument('--tau_f', type=float, default=None)
_arg_parser.add_argument('--vf_theta', type=float, default=None)
_args, _ = _arg_parser.parse_known_args()
if _args.tau_g is not None:
    fc.tau_g = _args.tau_g
if _args.tau_f is not None:
    fc.tau_f = _args.tau_f
if _args.vf_theta is not None:
    fc.vf_theta = _args.vf_theta


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


def zhang_gradient(__phi, n_dx=1.0, n_dy=1.0):
    Nx, Ny = __phi.shape
    grad_x = np.zeros_like(__phi)
    grad_y = np.zeros_like(__phi)

    for i in range(9):
        cx, cy = c[i]
        phi_shifted = np.roll(np.roll(__phi, -cx, axis=0), -cy, axis=1)
        grad_x += E[i] * cx * phi_shifted
        grad_y += E[i] * cy * phi_shifted

    prefactor = 1.0 / Cs2
    grad_x *= prefactor
    grad_y *= prefactor

    # eq(22) is only meaningful at fluid nodes - zero it at the ghost/solid
    # layer so it can never inject a spurious force there, independent of
    # whatever the stencil read across the wrap.
    grad_x[0, :] = 0.0; grad_x[-1, :] = 0.0
    grad_x[:, 0] = 0.0; grad_x[:, -1] = 0.0
    grad_y[0, :] = 0.0; grad_y[-1, :] = 0.0
    grad_y[:, 0] = 0.0; grad_y[:, -1] = 0.0

    return grad_x, grad_y


def zhang_laplacian(__phi, n_dx=1.0, n_dy=1.0):
    Ny, Nx = __phi.shape
    laplacian = np.zeros((Ny, Nx))

    for i in range(1, 9):  # i=0 term cancels
        cx, cy = c[i]
        phi_shifted = np.roll(np.roll(__phi, cx, axis=0), cy, axis=1)
        laplacian += E[i] * (phi_shifted - __phi)

    laplacian *= (2.0 / Cs2)     # usually 6.0

    # eq(22)'s Laplacian is only meaningful at fluid nodes - zero it at the
    # ghost/solid layer for the same reason as zhang_gradient: nothing
    # downstream (chemical_potential -> Fs) needs a real value there, and this makes
    # the wrap-vs-clamp question moot.
    laplacian[0, :] = 0.0; laplacian[-1, :] = 0.0
    laplacian[:, 0] = 0.0; laplacian[:, -1] = 0.0

    return laplacian


def zhang_interfacial_tension_check(fc, __phi, iteration, check_every=10):
    """
    Zhang eq(48)/(49) self-consistency check on vf_sigma.
    delta(xi) = (3/2W) * [(2*phi(xi)-1)^2 - 1]^2 satisfies integral(delta)=1 (eq 48),
    and sigma(xi) = sigma * delta(xi) (eq 49), so integrating sigma(xi) along a
    1D cut through the interface should recover fc.vf_sigma if the simulated
    profile matches the assumed equilibrium shape (eq 9) at the configured vf_W.
    Auto-picks the column with the largest phi variance so it still finds the
    interface for off-center geometries (e.g. a bubble, not just a centered
    horizontal interface).
    """
    if iteration % check_every != 0:
        return None

    fluid = __phi[1:Xn+1, 1:Yn+1]
    x_slice = 1 + int(np.argmax(np.var(fluid, axis=1)))
    phi_profile = __phi[x_slice, 1:Yn+1]

    delta = (3.0 / (2.0 * fc.vf_W)) * ((2.0 * phi_profile - 1.0) ** 2 - 1.0) ** 2
    sigma_profile = fc.vf_sigma * delta
    sigma_measured = np.sum(sigma_profile) * n_dy  # n_dy = 1 in these lattice units

    print(f"[eq49 check] iter={iteration}  x_slice={x_slice}  vf_sigma={fc.vf_sigma:.6f}  "
          f"sigma_measured={sigma_measured:.6f}  ratio={sigma_measured/fc.vf_sigma:.4f}")

    return sigma_measured


# Zhang text below eq(4): 
# where ... ρ = ϕρL + (1 − ϕ)ρG is the density of the fluid mixture; 
# similarly, η = ϕηL + (1 − ϕ)ηG is the dynamic viscosity of the fluid 
# mixture. In this study, we use the subscripts “L” and “G” to denote 
# liquid and gas, respectively.
# Zhang: ρ = φρ_L + (1-φ)ρ_G   eq.(4)
def density_and_viscosity(fc, _phi):
    _rho = _phi * fc.rho_L + (1.0 - _phi) * fc.rho_G
    _mu  = _phi * fc.mu_L  + (1.0 - _phi) * fc.mu_G
    
    return _rho, _mu


#Zhang eq(27):  the hydrodynamic pressure -> first-order moment
def zp(fc, _z_fi, _rho, u_ckl, iteration, n_dx=1.0, n_dy=1.0):
    """
    Vectorized evolution for p (hydrodynamic pressur) - D2Q9 streaming/collision.
    p =[i=0..8] ∑fi + δt/2 u⋅∇ρCs²
    """
    # channel velocity components
    term1 = np.sum(_z_fi, axis=0)  # Shape (nx, ny)

    # force term
    drho_dx, drho_dy = zhang_gradient(_rho, n_dx, n_dy)
    grad_rho = np.stack([drho_dx, drho_dy], axis=0)   # (2, nx, ny)
    u_dot_grad_rho = np.einsum('cij,cij->ij', u_ckl, grad_rho)
    term2 = (n_dt / 2) * u_dot_grad_rho * Cs2

    # hydrodynamic pressure
    _p = term1 + term2

    return _p


# mobility - anti-diffusion term in Allen–Cahn equation
def mobility(fc):
    m_phi = Cs2 * (fc.tau_g - 0.5) * n_dt

    return m_phi


# fluid viscosity eq(25)
def viscosity(fc, rho):
    nu = rho * Cs2 * (fc.tau_f - 0.5) * n_dt 

    return nu


# lambda - anti-diffusion term in Allen–Cahn equation
def z_lambda(fc, __phi):
    _z_lambda = 4. * __phi * (1 - __phi) / fc.vf_W

    return _z_lambda



def zu_ckl(fc, _z_fi, rho, body_force, _capillary_force):
    _u_ckl = np.einsum('ia,ijk->ajk', c, _z_fi) / (Cs2 * rho)  \
        + 1/(2*rho) * fc.ADD_SURFACE_TENSION_FORCE * _capillary_force \
        + 1/(2*rho) * fc.ADD_BODY_FORCE * body_force

    return _u_ckl


# Zhang eq(13): collision function of pressure distribution function 
def zfi(fc, z_fi, z_fi_c, u_ckl, rho, mu, Fs, G, iteration):
    """
    Zhang eq(13) collision + eq(15) streaming for fi in D2Q9 two-phase LBM.
    """
    # Zhang eq(20)/(21): forcing term Fi, shape (9, nx, ny)
    _Fi = Fi(fc, Fs, G, u_ckl, rho)

    # eq(13): collision
    z_fi_star = z_fi - (1.0 / fc.tau_f) * (z_fi - z_fi_c) + n_dt * _Fi

    # eq(15): streaming
    streamed = [
        np.roll(z_fi_star[i], shift=(c[i, 0], c[i, 1]), axis=(0, 1)) for i in range(9)
    ]
    z_fi[:] = np.stack(streamed, axis=0)

    return z_fi


# Zhang eq(6): Chemical potential
def chemical_potential(fc, __phi):
    """
    Zhang eq(6): Compute checmial potential μϕ
    μϕ = 4βϕ(ϕ - 1)(ϕ - 0.5) - κ∇2ϕ
    """
    term1 = 4. * fc.vf_beta * __phi*(__phi - 1.0) * (__phi - 0.5)
    term2 = zhang_laplacian(__phi)
    _mu_phi =  term1 -  fc.vf_kappa * term2

    return _mu_phi    


# Zhang eq(5): Fs is the surface tension force, expressed in a potential form
def Fs(fc, __phi, n_dx, n_dy):
    drho_dx, drho_dy = zhang_gradient(__phi, n_dx, n_dy)
    nabla_phi = np.stack([drho_dx, drho_dy], axis=0)
    _Fs = chemical_potential(fc, __phi) * nabla_phi

    return _Fs


# Zhang eq(20): Fi discrete forcing term for pressure distribution function in eq(13)
def Fi(fc, Fs, G, u_ckl, rho):
    ei_dot_u = np.einsum('ic,cjk->ijk', c, u_ckl)  
    u_sq = np.einsum('cjk,cjk->jk', u_ckl, u_ckl)  
    s0 = 1 + ei_dot_u / Cs2 + ei_dot_u**2 / (2 * Cs4) - u_sq / (2 * Cs2)
    s = E_exp * s0

    term1 = 1 - 1 / (2 * fc.tau_f)
    ei_u = c_exp - u_ckl[np.newaxis]

    drho_dx, drho_dy = zhang_gradient(rho * Cs2, n_dx, n_dy)
    stack_grad_rho = np.stack([drho_dx, drho_dy], axis=0) 

    forces_s = (Fs * fc.ADD_SURFACE_TENSION_FORCE + G * fc.ADD_BODY_FORCE)[np.newaxis] * s[:, np.newaxis] 
    second_term = (s - E_exp)[:, np.newaxis] * stack_grad_rho[np.newaxis]
    assert forces_s.shape == second_term.shape, f"forces {forces_s.shape} vs second_term {second_term.shape}"
    assert ei_u.shape == forces_s.shape, f"ei_u {ei_u.shape} vs forces {forces_s.shape}"    

    _Fi = term1 * np.einsum('icjk,icjk->ijk', ei_u, forces_s + second_term)
    
    return _Fi


# Zhang eq(18): equilibrium function for pressure disrtibution function
def zfi_c(fc, u, rho, p):
    """
    Zhang eq(18): fi equilibrium distribution for D2Q9.
    fi^eq = wi[p + rho(ei·u + (ei·u)²/(2cs²) - u²/2)]
    """
    c_dot_u = np.einsum('ia,axy->ixy', c, u)                # (9, nx, ny)
    u_dot_u = np.sum(u**2, axis=0)                          # (nx, ny)

    term1 = E_exp * p                                       # wi * p
    term2 = E_exp * rho * c_dot_u                           # wi * rho * (ei·u)
    term3 = E_exp * rho * (3.0/2.0) * c_dot_u**2            # wi * rho * (ei·u)²/(2cs²)
    term4 = E_exp * rho * 0.5 * u_dot_u                     # wi * rho * u²/2

    z_fi_c = term1 + term2 + term3 - term4

    return z_fi_c


# Zhang eq(12): collision function of order parameter distribution function 
def zgi(fc, z_gi, z_gi_c, __phi_old, _u_ckl_old, __phi, _u_ckl):
    """
    Fully vectorized collision and streaming for fi distribution (D2Q9).
    No Python loops anywhere - negative debug uses vectorized argmin/where.
    Assumes globals: RAISE_LESS_THAN_ZERO_ERROR, iteration, c(9,2).
    """
    # Collision: fully vectorized BGK
    omega_g = 1.0 / fc.tau_g
    z_gi_star = z_gi - omega_g * (z_gi - z_gi_c)  +  n_dt * Gi(fc, __phi_old, _u_ckl_old, __phi, _u_ckl)

    # Streaming: batched rolls
    streamed = [
        np.roll(z_gi_star[i], shift=(c[i, 0], c[i, 1]), axis=(0, 1)) for i in range(9)
    ]
    z_gi[:] = np.stack(streamed, axis=0)

    return z_gi 


# Zhang bottom p 32, 2nd column:
def n(fc, __phi):
    # n = ∇ϕ/∣∇ϕ∣ is the unit vector normal to the interface
    dphi_dx, dphi_dy = zhang_gradient(__phi, n_dx, n_dy)
    grad_phi = np.stack([dphi_dx, dphi_dy], axis=0) 
    mag = np.sqrt(dphi_dx**2 + dphi_dy**2)
    mag = np.where(mag < 1e-12, 1.0, mag)   # avoid div-by-zero
    _n = grad_phi / mag[np.newaxis]

    return _n


# Zhang eq(19): Gi discrete forcing term for order parameter in eq(12)
def Gi(fc, __phi_old, _u_ckl_old, __phi, _u_ckl):
    term1 = 1 - 1 / (2 * fc.tau_g)
    # ∂t(ϕu) - Note that in the above equations, the time derivative term ∂t(ϕu) 
    # is explicitly computed by a difference between two consecutive time steps
    dphi_u_dt = __phi[np.newaxis] * _u_ckl - __phi_old[np.newaxis] * _u_ckl_old
    cs2_lambda_n = Cs2 * z_lambda(fc, __phi) * n(fc, __phi)
    vec = dphi_u_dt  + cs2_lambda_n
    ei_dot_vec = np.einsum('ic,cjk->ijk', c, vec)

    _Gi = term1 * E_exp * ei_dot_vec / Cs2

    return _Gi


# Zhang eq(17): equilibrium distribution function for pressure distribution
def zgi_c(fc, u, _phi, iteration):
    """
    Zhang eq(17): gi equilibrium distribution for D2Q9.
    gi^eq = wi * phi * (1 + ei·u/cs²)
    """
    c_dot_u = np.einsum('ia,axy->ixy', c, u)  # (9, nx, ny)
    _zgi_c = E_exp * _phi * (1 + 3.0 * c_dot_u)

    _term1 = - np.max(np.abs(-c_dot_u))
    _term2 = np.max(np.abs(_zgi_c))
    PhiTerms.append((iteration, _term1, _term2))

    return _zgi_c


def print_top_layers(_phi_array, num_nodes=4, num_layers=2):
    """
    Print the first `num_nodes` values at the top `num_layers` of a 2D NumPy array `_phi_array`.
    
    Parameters:
        _phi_array (np.array): 2D array with shape (Xn, Yn)
        num_nodes (int): Number of nodes in x-direction to print
        num_layers (int): Number of top layers to print (from top of y-axis)
    """
    Xn, Yn = _phi_array.shape
    
    for i in range(num_layers):
        layer_idx = -1 - i  # top row first, then downward
        label = "Top Layer:" if i == 0 else f"Layer -{i}:"
        print(f"{label:<15}", end="")
        for x in range(num_nodes):
            print(f"{_phi_array[x, layer_idx]:12.8f}", end="   ")
        print()  # new line


def bounceBackTopBottom_conservation(fc, iteration, __gi, __fi, nx, ny):
    # channel i  = 0,1,2,3,4,5,6,7,8
    # anti-channel i_ = 0,3,4,1,2,7,8,5,6

    if ZERO_BCs:
        __fi[:, :, 0]    = 0.0
        __fi[:, :, ny+1] = 0.0    


    # bottom wall (y=1) - no-slip halfway bounce-back
    # straight vertical reflection
    __gi[2, :, 1] = __gi[4,:,0]                 # 4-> 2, die GhostNodes spielen keine Rolle
    # diagonal reflections with horizontal shift
    __gi[5,1:nx-1,1] = __gi[7,0:nx-2,0]         # 7 -> 5
    __gi[6,1:nx-1,1] = __gi[8,2:nx,0]           # 8 -> 6

    # top wall (y=ny) - no-slip halfway bounce-back
    # straight vertical reflection
    __gi[4,:,ny-2] = __gi[2,:,ny-1]             # 2-> 4, die GhostNodes spielen keine Rolle
    # diagonal reflections with horizontal shift
    __gi[7,1:nx-1,ny-2] = __gi[5,2:nx,ny-1]     # 5 -> 7
    __gi[8,1:nx-1,ny-2] = __gi[6,0:nx-2,ny-1]   # 6 -> 8

    if iteration in iterationsOfInterest:
        __phi = phi(fc, __gi)
        print(f"------ iteration: {iteration} -------------------------------------------------------------------")
        print_top_layers(__phi, num_nodes=4, num_layers=3)

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
    # with streaming relation fi*(xf,t) = fi(xff,t)
    # => fi(xf) = [1/1.5]·[0.5·fi(xff,t) + (0.5)·fi(xff,t) + 0.5·fi*(xs,t)]
    # => fi(xf) = [2/3]·[fi(xff,t) + 0.5·fi*(xs,t)]
    # => fi(xf) = (2/3)·[fi(xff,t) + 0.5·fi*(xs,t)]
    # => fi(xf) = (2/3)·fi(xff) + (2/3)·0.5·fi*(xs)
    # => fi(xf) = (2/3)·fi(xff) + (1/3)·fi*(xs)

    # xff: y-index 2
    # xs: y-index 0
    # xf: y-index 1    

    # bottom wall (i=2, 5, 6 — directions pointing N into fluid):
    __fi[2,:,1] = (2/3)*__fi[2,:,2] + (1/3)*__fi[4,:,0]                             # 4-> 2, die GhostNodes spielen keine Rolle
    __fi[5,1:nx-2,1] = (2/3)*__fi[5,2:nx-1,2] + (1/3)*__fi[7,0:nx-3,0]              # 7 -> 5
    #f[5,1:N-1,1]=f[7,0:N-2,0] # 7 -> 5
    __fi[6,2:nx-1,1] = (2/3)*__fi[6,1:nx-2,2] + (1/3)*__fi[8,3:nx,0]                # 8 -> 6
    #f[6,1:N-1,1]=f[8,2:-1,0]   # 8 -> 6

    # xff=(x+1,2) for NE → roll -1; xs=(x-1,0) → roll +1. Reversed for NW.
    # top wall (i=4, 7, 8 — directions pointing S into fluid):
    __fi[4,:,ny-2] = (2/3)*__fi[4,:,ny-3] + (1/3)*__fi[2,:,ny-1]                    # 2-> 4, die GhostNodes spielen keine Rolle        
    #f[4,:,1]=f[2,:,0] # 2-> 4, die GhostNodes spielen keine Rolle        
    __fi[7,2:nx-1,ny-2] = (2/3)*__fi[7,1:nx-2,ny-3] + (1/3)*__fi[5,3:nx,ny-1]     # 5 -> 7
    #f[7,1:N-1,1]=f[5,2:-1,0] # 5 -> 7        
    __fi[8,1:nx-2,ny-2] = (2/3)*__fi[8,2:nx-1,ny-3] + (1/3)*__fi[6,0:nx-3,ny-1]   # 6 -> 8
    #f[8,1:N-1,1]=f[6,0:N-2,0]   # 6 -> 8            


    # Prevent periodic-roll contamination: ghost nodes carry no distributions.
    # Streaming from interior will repopulate them correctly next step.
    if ZERO_BCs:
        __gi[:, :, 0]    = 0.0
        __gi[:, :, ny+1] = 0.0        

    return __gi, __fi 


def bounceBackLeftRight_conservation(fc, iteration, __gi, __fi, nx, ny):

    if ZERO_BCs:
        __fi[:, 0,    :] = 0.0
        __fi[:, nx+1, :] = 0.0        

    # Left wall (x=1) - no-slip halfway bounce-back
    __gi[1,1,:] = __gi[3,0,:]                   # 3-> 1, die GhostNodes spielen keine Rolle
    #f[1],1,:]=f[3,0,:] # 3-> 1, die GhostNodes spielen keine Rolle    
    __gi[5,1,1:ny-1] = __gi[7,0,0:ny-2]         # 7 -> 5
    #f[5,1,1:N-1]=f[7,0,0:N-2] # 7 -> 5    
    __gi[8,1,1:ny-1] = __gi[6,0,2:ny]           # 6 -> 8
    #f[8,1,1:N-1]=f[6,0,2:-1]  # 6 -> 8    

    # Right wall (x=nx) - no-slip halfway bounce-back
    __gi[3,nx-2,:] = __gi[1,nx-1,:]             # 3-> 1, die GhostNodes spielen keine Rolle
    #f[3,1,:]=f[1,0,:] # 3-> 1, die GhostNodes spielen keine Rolle    
    __gi[7,nx-2,1:ny-1] = __gi[5,nx-1,2:ny]     # 5 -> 7
    #f[7,1,1:N-1]=f[5,0,2:-1] # 5 -> 7    
    __gi[6,nx-2,1:ny-1] = __gi[8,nx-1,0:ny-2]   # 8 -> 6
    #f[6,1,1:N-1]=f[8,0,0:N-2]  # 8 -> 6    


    # Zhang eq(30) for fi: -> see explanation in bounceBackTopBottom_conservation
    # left wall (i=1, 5, 8 — directions pointing E into fluid):
    __fi[1, 1, :] = (2/3)*__fi[1, 2, :] + (1/3)*__fi[3, 0, :]                        # 3 -> 1
    __fi[5, 1, 1:ny-2] = (2/3)*__fi[5, 2, 2:ny-1] + (1/3)*__fi[7, 0, 0:ny-3]         # 7 -> 5
    __fi[8, 1, 2:ny-1] = (2/3)*__fi[8, 2, 1:ny-2] + (1/3)*__fi[6, 0, 3:ny]           # 6 -> 8

    # right wall (i=3, 6, 7 — directions pointing W into fluid):
    __fi[3, nx-2, :] = (2/3)*__fi[3, nx-3, :] + (1/3)*__fi[1, nx-1, :]                        # 1 -> 3
    __fi[6, nx-2, 1:ny-2] = (2/3)*__fi[6, nx-3, 2:ny-1] + (1/3)*__fi[8, nx-1, 0:ny-3]         # 8 -> 6
    __fi[7, nx-2, 2:ny-1] = (2/3)*__fi[7, nx-3, 1:ny-2] + (1/3)*__fi[5, nx-1, 3:ny]           # 5 -> 7

    # Zero left/right ghost columns
    if ZERO_BCs:
        __gi[:, 0,    :] = 0.0
        __gi[:, nx+1, :] = 0.0

    return __gi, __fi


def init_step_phi(xn, yn, phi_star_g, phi_star_l, xi=5.0, smooth_sigma=None):
    """
    Initialize a 2D step function field (phi) in the x-y plane with smooth transitions.

    Parameters
    ----------
    xn, yn : int
        Grid sizes in x and y directions.
    phi_star_g, phi_star_l : float
        Values for gas and liquid phases.
    xi : float, optional
        Interface thickness parameter.
    smooth_sigma : float, optional
        Gaussian smoothing sigma; if None, defaults to xi/1.5.

    Returns
    -------
    phi : ndarray (shape = (xn, yn))
        Initialized scalar field.
    """
    # Create grid
    x, y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')

    # Key geometric reference points
    x_mid, x_34 = 0.5*(xn-1), 0.75*(xn-1)
    y_mid, y_23 = 0.5*(yn-1), (2/3)*(yn-1)

    # Initialize step function regions
    phi = np.where(
        ((x <= x_mid) & (y < y_mid)) |
        ((x_mid < x) & (x <= x_34) & (y < y_23)) |
        ((x > x_34) & (y < y_mid)),
        phi_star_l, phi_star_g
    )

    # Smooth interface transitions
    phi_mid = 0.5 * (phi_star_l + phi_star_g)
    phi_diff = 0.5 * (phi_star_l - phi_star_g)
    sigma = np.sqrt(2) * xi
    w = 3.0 * xi

    # Interface near y = y_mid
    mask = ((x <= x_mid) | (x > x_34)) & (np.abs(y - y_mid) <= w)
    phi[mask] = phi_mid + phi_diff * erf((y_mid - y)[mask] / sigma)

    # Interface near y = y_23
    mask = (x_mid < x) & (x <= x_34) & (np.abs(y - y_23) <= w)
    phi[mask] = phi_mid + phi_diff * erf((y_23 - y)[mask] / sigma)

    # Optional Gaussian smoothing
    phi = gaussian_filter(phi, sigma=(xi/1.5 if smooth_sigma is None else smooth_sigma), mode='nearest')

    return phi


def init_horizontal_phi(xn, yn, phi_star_g, phi_star_l, height=None, W=4.0):
    """
    Initialize the order parameter _phi as the Zhang eq(9) equilibrium tanh profile.

    Args:
        xn (int): Number of grid points in x-direction (Xn+2).
        yn (int): Number of grid points in y-direction (Yn+2).
        phi_star_g (float): Order parameter value for gas phase.
        phi_star_l (float): Order parameter value for liquid phase.
        height (float, optional): Interface centre y. Defaults to (yn-1)/2.
        W (float): Interface width parameter (vf_W). Controls tanh steepness.

    Returns:
        np.ndarray: 2D array of shape (xn, yn) containing the initialized _phi values.
    """
    y0 = (yn - 1) / 2 if height is None else height
    x, y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')
    _phi = (phi_star_l + phi_star_g) / 2 + (phi_star_l - phi_star_g) / 2 * np.tanh(2 * (y0 - y) / W)

    return _phi


def apply_periodic_boundary_conditions(_fi, _gi):
    _gi[:, 0, :] = _gi[:,Xn,:]
    _gi[:, Xn+1, :] = _gi[:,1,:]      
    _fi[:, 0, :] = _fi[:,Xn,:]
    _fi[:, Xn+1, :] = _fi[:,1,:]    


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
    return all_iters[:no_slices]


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


# ────────────────────────────────────────────────
# Tangent vector m (vector output)
# ────────────────────────────────────────────────
def m_vector(fc, wall):
    """
    Returns the capillary vector m at wall, with inversion for left and right walls.
    Uses local variables before returning for easier debugging.
    """
    # Get original vector from the existing m_vector function
    theta = fc.vf_theta_rad
    phi = np.pi/2 - theta  # angle relative to wall normal

    if wall == 'bottom':
        m_bottom = np.array([0, 1])
        return m_bottom
    elif wall == 'top':
        m_top = np.array([0, -1])
        return m_top
    elif wall == 'left':
        n = np.array([1, 0])
        t = np.array([0, 1])
        m = n * np.cos(phi) - t * np.sin(phi)
        # invert the vector
        m_left = -m / np.linalg.norm(m)
        return m_left
    elif wall == 'right':
        n = np.array([-1, 0])
        t = np.array([0, 1])
        m = n * np.cos(phi) - t * np.sin(phi)
        # invert the vector
        m_right = -m / np.linalg.norm(m)
        return m_right
    else:
        raise ValueError("Wall must be 'bottom','top','left','right'")


# Quick aliases
m_left   = lambda fc: m_vector(fc, "left")
m_right  = lambda fc: m_vector(fc, "right")
m_bottom = lambda fc: m_vector(fc, "bottom")
m_top    = lambda fc: m_vector(fc, "top")


def zhang_weight_function(fc, __phi, eps=1e-12):
    __phi = np.asarray(__phi)

    w = np.zeros_like(__phi, dtype=float)
    mask = (__phi > eps) & (__phi < 1.0 - eps)

    w[mask] = (
        (3.0 * fc.vf_sigma / 8.0)
        * ((2.0 * __phi[mask] - 1.0) ** 2 - 1.0) ** 2
        / (__phi[mask] * (1.0 - __phi[mask]))
    )

    return w


def zhang_fc(fc, __phi):
    """
    Compute Zhang surface tension force at walls.
    
    Parameters
    ----------
    fc : object
        Contains fluid properties, e.g., vf_sigma.
    _phi : np.ndarray
        Phase field, shape (Xn+2, Yn+2)

    Returns
    -------
    _fc : np.ndarray
        Force array, shape (2, Xn+2, Yn+2)
    """
    #dphidx, dphidy = c_first_derivative0(_phi)
    dphidx, dphidy = zhang_gradient(__phi)

    _fc = np.zeros((__phi.shape[0], __phi.shape[1], 2), dtype=np.float64)

    # --- LEFT WALL ---
    i = 1
    j = np.arange(1, Yn+1)
    m = m_left(fc)
    delta_x, delta_y = 0.0, n_dy
    abs_phi = np.abs(dphidx[i, j]) * delta_x + np.abs(dphidy[i, j]) * delta_y
    w = zhang_weight_function(fc, __phi[i, j])
    val_left = (w * abs_phi)[:, None] * m[None, :]
    _fc[i, j, :] = val_left

    # --- RIGHT WALL ---
    i = Xn
    j = np.arange(1, Yn+1)
    m = m_right(fc)
    delta_x, delta_y = 0.0, n_dy
    abs_phi = np.abs(dphidx[i, j]) * delta_x + np.abs(dphidy[i, j]) * delta_y
    w = zhang_weight_function(fc, __phi[i, j])
    val_right = (w * abs_phi)[:, None] * m[None, :]
    _fc[i, j, :] = val_right

    # --- BOTTOM WALL ---
    j = 1
    i = np.arange(1, Xn+1)
    m = m_bottom(fc)
    delta_x, delta_y = 1.0, 0.0
    abs_phi = np.abs(dphidx[i, j]) * delta_x + np.abs(dphidy[i, j]) * delta_y
    w = zhang_weight_function(fc, __phi[i, j])
    val_bottom = (w * abs_phi)[:, None] * m[None, :]
    _fc[i, j, :] = val_bottom

    # --- TOP WALL ---
    j = Yn
    i = np.arange(1, Xn+1)
    m = m_top(fc)
    delta_x, delta_y = 1.0, 0.0
    abs_phi = np.abs(dphidx[i, j]) * delta_x + np.abs(dphidy[i, j]) * delta_y
    w = zhang_weight_function(fc, __phi[i, j])
    val_top = (w * abs_phi)[:, None] * m[None, :]
    _fc[i, j, :] = val_top

    return np.transpose(_fc, (2, 0, 1))  # <-- change here


def set_solid_nodes(iteration, fc, _phi):
    #2. viscous force - adaptation of Zhang et al. eq(5), with μ_phi exchanged for μ_c
    #fc.vf_kappa = 3 * fc.vf_sigma * fc.vf_W / 2
    #fc.vf_beta = 12 * fc.vf_sigma / fc.vf_W

    # ------- Capillary effect (wetting on bottom, left, and right walls only) -------
    # n = ∇ϕ|∇ϕ∣

    __phi = _phi.copy()

    # The outer ghost ring can hold stale values (periodic np.roll streaming
    # + phi(fc,__gi) reconstruction in bounceBackTopBottom/LeftRight_conservation
    # leaves old data in corners that were never a real phi_p). Mirror each
    # ghost cell from its nearest interior neighbour (zero-gradient/Neumann)
    # so the corner entries phi_s() reads below are physically consistent
    # instead of leftover noise.
    __phi[:, 0]    = __phi[:, 1]
    __phi[:, Yn+1] = __phi[:, Yn]
    __phi[0, :]    = __phi[1, :]
    __phi[Xn+1, :] = __phi[Xn, :]

    # Contact angle (same for all solid walls — modify later if needed)
    thetaCap = compContactAngle(fc)

    # === Top wall (y=1) ===
    _phi_p_n_top = __phi[:, Yn]                    # 1. fluid node inside
    _phi_s_top = phi_s(_phi_p_n_top, fc.vf_theta, thetaCap, n_dx)   

    # === Bottom wall (y=1) ===
    _phi_p_n_bottom = __phi[:, 1]                    # 1. fluid node inside
    _phi_s_bottom = phi_s(_phi_p_n_bottom, fc.vf_theta, thetaCap, n_dx)           

    # === Left wall (x=1) ===
    _phi_p_n_left = __phi[1, :]                    # 1. fluid node inside
    _phi_s_left = phi_s(_phi_p_n_left, fc.vf_theta, thetaCap, n_dx)                   

    # === Right wall (x=Xn) ===
    _phi_p_n_right = __phi[Xn, :]                    # 1. fluid node inside
    _phi_s_right = phi_s(_phi_p_n_right, fc.vf_theta, thetaCap, n_dx)                           


    if iteration in iterationsOfInterest:
        _base = (Yn + 2) // 2

        node_data = []

        for _offset in [4,3,2,1,0,-1,-2,-3,-4]:
            result = calc_node_s_diff(iteration, _base, _offset, _phi_s_left, _phi)
            node_data.append(result)

        plotter.plot_left_wall_all_nodes(iteration, node_data) 


    # Step 3: Assign to solid wall nodes only
    # Top wall
    __phi[:, Yn+1] = _phi_s_top  # current x=Xn+1  

    # Bottom wall
    __phi[:, 0]  = _phi_s_bottom   # bottom

    # Left wall
    __phi[0, :] = _phi_s_left   # current x=0

    # Right wall
    __phi[Xn+1, :] = _phi_s_right  # current x=Xn+1  

    return __phi


def calc_node_s_diff(iteration, _base, offset, _phi_s, _phi):
    _pos = _base + offset

    _pos_0 = _phi_s[_pos]
    _pos_1 = _phi[1, _pos]
    _pos_2 = _phi[2, _pos]
    _pos_3 = _phi[3, _pos]
    _pos_4 = _phi[4, _pos]
    _pos_5 = _phi[5, _pos]
    _pos_6 = _phi[6, _pos]

    _diff_0 = _pos_0 - _phi[0, _pos]

    lstNodes = [_pos_0, _pos_1, _pos_2, _pos_3, _pos_4, _pos_5, _pos_6]

    print(f"Node: {_pos}; {_pos_0}; [diff: {_diff_0}]; "
          f"{_pos_1}; {_pos_2}; {_pos_3}; {_pos_4}; {_pos_5}; {_pos_6}")

    return offset, lstNodes, _diff_0


def bodyForce(fc, rho):
    _force = fc.F_body[:, None, None] * rho 

    return _force


def sufficient_stability_condition(u):
    _sufficient_stability_condition = np.abs(np.max(u)) < (np.sqrt(1/3)*n_dx/n_dt)
    return _sufficient_stability_condition


def optimal_stability_condition(u):
    _optimal_stability_condition1 = tau_nd/n_dx >= 1
    _optimal_stability_condition2 = np.abs(np.max(u)) < (np.sqrt(2/3)*n_dx/n_dt)
    
    return (_optimal_stability_condition1 and _optimal_stability_condition2)


def apply_mass_conservation(phi_old_total, __phi):
    # CRITICAL: Restore exact mass conservation after overwrite
    # Compute total ϕ in fluid domain AFTER wetting
    phi_total_after = np.sum(__phi[1:Xn+1, 1:Yn+1])

    # Scale ONLY the fluid domain back to pre-wetting total (very small correction)
    if fc.ENFORCE_MASS_CONSERVATION and abs(phi_total_after - phi_old_total) > 1e-8:  # avoid div-by-zero or noise
        scale_factor = phi_old_total / phi_total_after
        __phi[1:Xn+1, 1:Yn+1] *= scale_factor

        # Optional print to monitor (remove later)
        if iteration % 500 == 0:
            print(f"          | Wetting overwrote → applied scale {scale_factor:.10f} "
                f"(delta before scale: {phi_total_after - phi_old_total:+.8f})")
    # ──────────────────────────────────────────────────────────────

    phi_after_stream_collide = np.sum(__phi[1:Xn+1, 1:Yn+1])
    if iteration % 500 == 0 or iteration < 50:  # print every 50 steps + first 50
        print(f"Iter {iteration:5d} | ϕ after sum fi = {phi_after_stream_collide:.4f}")

    return __phi


def save_phi_results(_phi_n_ext, _phi_min_ext, _phi_max_ext, filename="phi_results.txt"):
    """
    Save normalized phi results to a text file.

    Parameters
    ----------
    _phi_n_ext : 2D np.array
        Array of size (Xn, Yn), x horizontal, y vertical.
    _phi_min_ext : float
        Minimum phi value (first value in file)
    _phi_max_ext : float
        Maximum phi value (second value in file)
    filename : str
        Name of the file to save, defaults to 'phi_results.txt' in script directory.
    """
    # File path in the same directory as the calling script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Write header and flipped 2D array
    with open(file_path, "w") as f:
        f.write(f"{_phi_min_ext} {_phi_max_ext}\n")
        np.savetxt(f, _phi_n_ext[:, ::-1].T, fmt="%.6f")


# ──────────────────────────────────────────────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────────────────────────────────────────────
#lattice for phase space; Nx+3 is due to periodic boundary conditions
#Nx is the number of divisions in the x-direction, thus there are Nx+3 points when including the extra nodes 0 and N+1 in x-direction
#lattice columns start with 0 and end with Nx+2, X(0) = X(0) and X(N+1) = X(Nx+2)

#average velocity, cartesion x,y-directions, k is y-position, l is x-position
u_ckl = np.zeros((2, Xn+2, Yn+2), dtype=np.float64)
INIT_RHO = 1 #0.001
rho = np.full((Xn+2, Yn+2), INIT_RHO, dtype=np.float64)

# Simulation parameters
R = D / 2  # Radius of the pipe

start = time.perf_counter()
y0 = (Yn-1)/2
x,y = np.meshgrid(np.arange(Xn+2),np.arange(Yn+2),indexing='ij')

PARAMETER_STUB = "__" + ACTIVE_CASE + "__nodes_" + str(DEFAULT_D_ND) + "__tau_f_" + str(fc.tau_f) + "__tau_g_" + str(fc.tau_g) + "__Kf_" + str(fc.Kf) + "__Theta_" + str(fc.vf_theta)
images_dir = os.path.join(images_subdir, SCRIPT_FILENAME + PARAMETER_STUB)
os.makedirs(images_dir, exist_ok=True)

#phi initialisation
# Corrected order parameter: phi decreases from phi_star_L to phi_star_G as y > y0
_phi = (fc.phi_star_L + fc.phi_star_G) / 2 + (fc.phi_star_L - fc.phi_star_G) / 2 * erf((y0 - y) / (np.sqrt(2) * fc.vf_W))

# Example usage during initialization
if PHI_DISTRIBUTION == "STEP":    
    # Replace the original _phi initialization with a call to the method
    _phi = init_step_phi(Xn+2, Yn+2, fc.phi_star_G, fc.phi_star_L, xi=5.0) 
    density_profile_x_position = int(2/3*Xn)
    density_profile_y_position = int(2/3*Yn)   
if PHI_DISTRIBUTION == "HORIZONTAL":
    # Replace the original _phi initialization with a call to the method
    _phi = init_horizontal_phi(Xn+2, Yn+2, fc.phi_star_G, fc.phi_star_L, height=(Yn+1)/2, W=fc.vf_W)
    density_profile_x_position = Xn//2
    density_profile_y_position = Yn//2


PhiTerms = []
rho_bounds = []
Invariants = []
MomentumBounds = []
StabilityConditions = []
GrowthMetric_uckl_x = []
GrowthMetric_uckl_y = []
GrowthMetric_uckl_star_y = []
GrowthMetric_div_u_raw = []
GrowthMetric_u_ckl_du_dy = []
DivU_max = []
fcBounds_left = []
fcBounds_right = []
SpuriousFields = []
PhiCollector = []
PhiOnPlaneCollector_0 = []
PhiOnPlaneCollector_1 = []
BondNumber = []

iteration = 0

list_avg_velocities_x = {}
list_avg_velocities_y = {}

list_phi = {}
list_dphi_0 = {}
list_dphi_1 = {}

list_BodyForce_0 = {}
list_BodyForce_1 = {}
list_NetForce = {}

yc = (Yn+2)//2 -4

u_ckl = np.zeros((2, Xn+2, Yn+2),dtype=np.float64)
p = np.zeros((Xn+2, Yn+2),dtype=np.float64)
z_gi_c = np.zeros((9,Xn+2, Yn+2),dtype=np.float64)
z_gi = np.zeros((9,Xn+2, Yn+2),dtype=np.float64)
# Before the main loop
z_gi = zgi_c(fc, np.zeros_like(u_ckl), _phi, iteration)
z_fi_c = np.zeros((9, Xn+2, Yn+2),dtype=np.float64)
z_fi = np.zeros((9, Xn+2, Yn+2),dtype=np.float64)

__phi_old = np.copy(_phi) 
_u_ckl_old = np.copy(u_ckl)

rho, mu = density_and_viscosity(fc, _phi)
zhang_surface_tension_force = np.zeros((2, Xn+2, Yn+2))

# --- Compact Iteration Snapshots (Directly Using TOTAL_ITERATIONS) ---
'''
no_slices = 11
fixed = [0, 500, 1000, 2000, 3000, 4000]
n_rem = no_slices - len(fixed)
exp = 3.0

t = np.linspace(0, 1, n_rem + 1)[1:]
post = np.floor((1 - np.exp(-exp * t)) * (TOTAL_ITERATIONS - 1 - fixed[-1])).astype(int) + fixed[-1]
iterationsOfInterest = sorted(set(fixed + post.tolist()))[:no_slices]'''
iterationsOfInterest = get_iterations_of_interest(fc.TOTAL_ITERATIONS, no_slices=fc.NO_DATA_DUMP_SLICES, exp_factor=4.0)
density_slices = []

plotter = Plotter2D(
    script_dir=script_dir,
    script_filename=SCRIPT_FILENAME + PARAMETER_STUB,
    #images_subdir=IMAGES_SUBDIR,
    images_subdir=images_dir,
    total_iterations=fc.TOTAL_ITERATIONS,
    filename_padding_width=fc.FILENAME_PADDING_WIDTH,
    debug_log=debug_log,
    PLOTREALTIME=True
)

rho_min = np.min(rho)
rho_max = np.max(rho)
title = "Density map"
plotter.density_map_standalone(rho, rho_min, rho_max, title, iteration)
plotter.save_phi_snapshot(_phi, iteration, fc.phi_star_G, fc.phi_star_L)

u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2)]
epsilon_u_ckl = 0
epsilon_u_ckl_list = []

# ──────────────────────────────────────────────────────────────────────────────────────────
# simulation
# ──────────────────────────────────────────────────────────────────────────────────────────

# Realtime plotting
if PLOTREALTIME:
    #fig_rt, ax = plt.subplots(figsize=(6,6), dpi=80)  # create one figure
    #fig_rt, axes = plt.subplots(3, 1, figsize=(8, 12))  # 3 stacked plots
    #ax_phi, ax_rho, ax_vort = axes
    plt.ion()  # turn on interactive mode


while iteration < fc.TOTAL_ITERATIONS:
    if iteration % 100 == 0:
        debug_log('ITER', 'Iter %d: phi min=%.3e, max=%.3e', iteration, np.min(_phi), np.max(_phi))

    # ──────────────────────────────────────────────────────────────
    #          Forces: Body and Surface Tension
    # ──────────────────────────────────────────────────────────────
    #Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    #Kürger et al, p. 241 eq. (6.29) & Table 6.1
    #1. Shan-Chen - A=tau*n_dt
    A = n_dt * n_dt

    # 1. Body mass force
    # variable G in eq(20)
    body_force = A * bodyForce(fc, rho) * n_dt
    assert body_force.shape == (2, Xn+2, Yn+2), f"body_force shape: {body_force.shape}"

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang functions: 2.Equilibrium , 3.Collision/Relaxation, 4.Streaming
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 1. Compute equilibrium distributions (fi_c, gi_c) ===
    # Zhang eq(17):  equilibrium distribution function for order parameter
    z_gi_c = zgi_c(fc, u_ckl, _phi, iteration)
    # Zhang eq(18):  equilibrium distribution function for pressure distribution function
    z_fi_c = zfi_c(fc, u_ckl, rho, p)      

    validate_field(z_gi_c, 'z_gi_c', iter=iteration, allow_neg=True)
    validate_field(z_fi_c, 'z_fi_c', iter=iteration, allow_neg=True)  

    # isolate Gi() specifically, since dphi_u_dt just went live for the first time
    _Gi_check = Gi(fc, __phi_old, _u_ckl_old, _phi, u_ckl)
    validate_field(_Gi_check, 'Gi output', iter=iteration, allow_neg=True)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang functions: 3.Collision/Relaxation, 4.Streaming (advection)
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 3. COLLISION + STREAMING (inside fi and gi) ===    
    # Zhang eq(2): calculation of the order parameter which distiguishes the two phases
    z_gi  = zgi(fc, z_gi, z_gi_c, __phi_old, _u_ckl_old, _phi, u_ckl)
    validate_field(z_gi, 'z_gi (post collision+stream)', iter=iteration, allow_neg=True)

    _Fs = Fs(fc, _phi, n_dx, n_dy)
    validate_field(_Fs, 'Fs', iter=iteration, allow_neg=True)
    assert _Fs.shape == (2, Xn+2, Yn+2), f"_Fs shape: {_Fs.shape}"
    assert _Fs.shape == body_force.shape, f"_Fs {_Fs.shape} != body_force {body_force.shape}"
    # Zhang eq(2): calculation of the pressure distribution function   
    z_fi = zfi(fc, z_fi, z_fi_c, u_ckl, rho, mu, _Fs, body_force , iteration)
    validate_field(z_fi, 'z_fi (post collision+stream)', iter=iteration, allow_neg=True)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # 5. Boundary conditions
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # === 2. UPDATE GHOST NODES FROM CURRENT POST-COLLISION STATE ===
    # Use _fi_c and _gi_c (post-collision, pre-streaming)

    # 4. top/bottom conservative bounceback
    z_gi, z_fi = bounceBackTopBottom_conservation(fc, iteration, z_gi, z_fi, Xn+2, Yn+2)
    

    #4.1b. top/bottom conservative bounceback
    #apply_periodic_boundary_conditions(_fi, _gi)    
    z_gi, z_fi = bounceBackLeftRight_conservation(fc, iteration, z_gi, z_fi, Xn+2, Yn+2)


    # ──────────────────────────────
    # NEW: Measure ϕ mass from previous step
    phi_old_total = np.sum(_phi[1:Xn+1, 1:Yn+1])
    __phi_old = np.copy(_phi) 
    _u_ckl_old = np.copy(u_ckl) 
    # ──────────────────────────────     

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # 1. Moment update
    # ──────────────────────────────────────────────────────────────────────────────────────────
    # Zhang eq(26): zeroth-order moment, calculation of order parameter to distiguish the 2 phases
    _phi = phi(fc, z_gi)
    # === LIGHT INTERFACE SMOOTHING TO REDUCE GRID PINNING ===
    # Apply very light Gaussian filter — smooth meniscus and reduce sharp corners
    #_phi = gaussian_filter(_phi, sigma=0.15)   # sigma=0.5–1.0 is usually enough

    # ──────────────────────────────
    # Mass conservation diagnostoics
    # ──────────────────────────────   
    _phi = apply_mass_conservation(phi_old_total, _phi)
    # ──────────────────────────────  

    if ADD_METRICS: 
        debug_log('ITER', 'Iter %d: fi_c min=%.3e, max=%.3e | fi min=%.3e, max=%.3e', 
          iteration, np.min(z_gi_c), np.max(z_gi_c), np.min(z_gi), np.max(z_gi))          
        debug_log('ITER', ' advisory Iter %d: fi min=%.3e, max=%.3e', 
          iteration, np.min(z_gi), np.max(z_gi))    
        debug_log('ITER', 'Iter %d: phi at y=0: %.3e, y=50: %.3e, y=51: %.3e', iteration, np.mean(_phi[:,1]), np.mean(_phi[:,50]), np.mean(_phi[:,51])) 
        debug_log('ITER', 'Iter %d: rho min=%.3e, max=%.3e', iteration, np.min(rho), np.max(rho))        

       
    if iteration in iterationsOfInterest:
        #phi mapping
        plotter.save_phi_snapshot(_phi, iteration, fc.phi_star_G, fc.phi_star_L)    

        # Store 2D data (existing)
        list_avg_velocities_x[iteration] = u_ckl[0, 1:-1, :].copy()
        list_avg_velocities_y[iteration] = u_ckl[1, 1:-1, :].copy()
        list_phi[iteration] = _phi[:,yc].copy()
        list_dphi_0[iteration] = zhang_gradient(_phi)[0][:,yc].copy()
        list_dphi_1[iteration] = zhang_gradient(_phi)[1][:,yc].copy()
        
        # density mapping
        rho_min = np.min(rho)
        rho_max = np.max(rho)
        title = "Density map"
        plotter.density_map_standalone(rho, rho_min, rho_max, title, iteration)
        rho_slice = rho[density_profile_x_position, :].copy()
        density_slices.append((iteration, rho_slice))

    # ──────────────────────────────────────────────────────────────
    #          Forces: Surface Tension
    # ──────────────────────────────────────────────────────────────
    # Calculation of a predicted velocity of the 2 phase fluid without pressure gradient
    # Zhang  eq(20): Compute u(x,t+n_dt)
    # Kürger et al, p. 241 eq. (6.29) & Table 6.1

    # Bond Nummber
    Bnon = rho_0 * fc.g * dx**2 / fc.vf_sigma
    Blat = rho_0 * fc.g * n_dx**2 / fc.vf_sigma
    #if iteration == 0 or iteration==(fc.TOTAL_ITERATIONS - 1) or iteration % 500 == 0:
    if ADD_METRICS and iteration in iterationsOfInterest:
        BondNumber.append((iteration, Bnon, Blat))
        debug_log('ITER', 'iteration: %d; Bond no. (non-dimensional): %.1f %%; Bond no. (lattice) %.1f %%', 
            iteration, Bnon, Blat)
        

    # 2. Surface tensions forces
    # Only write on the final iteration -- np.savetxt formats every cell as
    # text, ~77ms/call, and the file is fully overwritten each time anyway
    # (nothing reads it until the process exits), so writing it every
    # iteration was ~15 min of pure waste per 12001-iteration run.
    if iteration == fc.TOTAL_ITERATIONS - 1:
        save_phi_results(_phi, fc.phi_star_G, fc.phi_star_L)


    _phi  = set_solid_nodes(iteration, fc, _phi)
    if iteration in iterationsOfInterest and fc.vf_theta > 90:
        print("phi at left wall (ghost + first fluid nodes):", _phi[0:3, yc])

    #Calculation of rho, mu
    rho, mu = density_and_viscosity(fc, _phi)

    #if fc.ADD_SURFACE_TENSION_FORCE == 1:
    # Zhang eq(5): Fs is the surface tension force, expressed in a potential form 
    #zhang_surface_tension_force = zhang_fc(fc, _phi)
    _Fs = Fs(fc, _phi, n_dx, n_dy)
    if ZERO_BCs:
        _Fs[:, :, 0]  = 0.0   # bottom ghost row
        _Fs[:, :, -1] = 0.0   # top ghost row
        _Fs[:, 0, :]  = 0.0   # left ghost column
        _Fs[:, -1, :] = 0.0   # right ghost column
    zhang_surface_tension_force = _Fs
    assert zhang_surface_tension_force.shape == (2, Xn+2, Yn+2), f"zhang_surface_tension_force shape: {zhang_surface_tension_force.shape}"
    _capillary_force = fc.ADD_SURFACE_TENSION_FORCE * zhang_surface_tension_force * fc.vf_capillaryForceMultiplier

        # ── CONTACT-LINE FORCE DIAGNOSTIC ─────────────────────────────────────────
    if iteration in iterationsOfInterest:
        # find y-index of interface at left wall (phi closest to 0.5)
        phi_left = _phi[1, 1:Yn+1]          # fluid column at x=1
        y_cl = int(np.argmin(np.abs(phi_left - 0.5))) + 1   # index in full array

        _fi_term  = np.einsum('ia,ijk->ajk', c, z_fi) / (Cs2 * rho)
        _bf_term  = (1.0 / (2.0 * rho)) * fc.ADD_BODY_FORCE  * body_force
        _cap_term = (1.0 / (2.0 * rho)) * fc.ADD_SURFACE_TENSION_FORCE * _capillary_force
        _Fs_bulk  = Fs(fc, _phi, n_dx, n_dy)
        _Fs_term  = (1.0 / (2.0 * rho)) * fc.ADD_SURFACE_TENSION_FORCE * _Fs_bulk

        print(f"\n── CL DIAG iter={iteration}  left-wall contact line at y={y_cl} ──")
        print(f"  phi[1,y_cl-1..y_cl+1] = {_phi[1, y_cl-1]:.4f}  {_phi[1, y_cl]:.4f}  {_phi[1, y_cl+1]:.4f}")
        print(f"  u_y total   = {(_fi_term+_bf_term+_cap_term)[1, 1, y_cl]:.4e}")
        print(f"    fi_term   = {_fi_term [1, 1, y_cl]:.4e}")
        print(f"    bf_term   = {_bf_term [1, 1, y_cl]:.4e}")
        print(f"    cap_term  = {_cap_term[1, 1, y_cl]:.4e}   <-- zhang_fc contribution")
        print(f"    Fs_term   = {_Fs_term [1, 1, y_cl]:.4e}   <-- bulk Fs contribution")
        print(f"  zhang_fc[x,y] at (1,y_cl)  = {zhang_surface_tension_force[:, 1, y_cl]}")
        print(f"  Fs_bulk  at (1,y_cl)        = {_Fs_bulk[:, 1, y_cl]}")
        print(f"  body_force at (1,y_cl)      = {body_force[:, 1, y_cl]}")
        print(f"  rho at (1,y_cl)             = {rho[1, y_cl]:.4e}")
        # same for right wall
        phi_right = _phi[Xn, 1:Yn+1]
        y_cl_r = int(np.argmin(np.abs(phi_right - 0.5))) + 1
        print(f"  u_y total right wall (x=Xn, y={y_cl_r}) = {(_fi_term+_bf_term+_cap_term)[1, Xn, y_cl_r]:.4e}")
        print(f"    fi_term   = {_fi_term [1, Xn, y_cl_r]:.4e}")
        print(f"    bf_term   = {_bf_term [1, Xn, y_cl_r]:.4e}")
        print(f"    cap_term  = {_cap_term[1, Xn, y_cl_r]:.4e}")
    # ── END DIAGNOSTIC ────────────────────────────────────────────────────────

    # --> Zhang eq(28) - fluid velocity
    if iteration == 0:
        _fi_term  = np.einsum('ia,ijk->ajk', c, z_fi) / (Cs2 * rho)
        _bf_term  = (1.0 / (2.0 * rho)) * fc.ADD_BODY_FORCE * body_force
        _cap_term = (1.0 / (2.0 * rho)) * fc.ADD_SURFACE_TENSION_FORCE * _capillary_force
        _u_test   = _fi_term + _bf_term + _cap_term
        _loc = np.unravel_index(np.argmax(np.abs(_u_test[1])), _u_test[1].shape)
        _x, _y = _loc
        print(f"[DIAG iter=0] max |u_y| loc=({_x},{_y})  u_y={_u_test[1,_x,_y]:.6e}")
        print(f"  fi_term  u_y = {_fi_term [1,_x,_y]:.6e}")
        print(f"  bf_term  u_y = {_bf_term [1,_x,_y]:.6e}")
        print(f"  cap_term u_y = {_cap_term[1,_x,_y]:.6e}")
        print(f"  rho[{_x},{_y}]          = {rho[_x,_y]:.6e}")
        print(f"  z_fi[:,{_x},{_y}]       = {z_fi[:,_x,_y]}")
        print(f"  cap_force[:,{_x},{_y}]  = {_capillary_force[:,_x,_y]}")
        print(f"  body_force[:,{_x},{_y}] = {body_force[:,_x,_y]}")
        # also report max over interior only
        _u_int = _u_test[1, 1:Xn+1, 1:Yn+1]
        _loc_int = np.unravel_index(np.argmax(np.abs(_u_int)), _u_int.shape)
        _xi, _yi = _loc_int[0]+1, _loc_int[1]+1
        print(f"  interior max |u_y| loc=({_xi},{_yi})  u_y={_u_test[1,_xi,_yi]:.6e}")
        print(f"    fi_term={_fi_term[1,_xi,_yi]:.4e}  bf={_bf_term[1,_xi,_yi]:.4e}  cap={_cap_term[1,_xi,_yi]:.4e}  rho={rho[_xi,_yi]:.4e}")

    u_ckl = zu_ckl(fc, z_fi, rho, body_force, _capillary_force)
    _u_ckl_abs = np.abs(u_ckl)
    print(f"iter {iteration}: u_ckl max |value| = {np.max(_u_ckl_abs):.6e}")
    _loc = np.unravel_index(np.argmax(_u_ckl_abs), u_ckl.shape)
    print(f"iter {iteration}: u_ckl max loc={_loc}  phi there={_phi[_loc[1],_loc[2]]:.4f}")

    if iteration in iterationsOfInterest:
        print(f"Iter {iteration:5d} | u_max = {np.max(np.abs(u_ckl)):.6e} | u_loc = {np.unravel_index(np.argmax(np.abs(u_ckl[1])), u_ckl[1].shape)}")
    assert u_ckl.shape == (2, Xn+2, Yn+2), f"u_ckl shape: {u_ckl.shape}"

       
    if iteration in iterationsOfInterest:
        # Store 2D data (existing)
        list_BodyForce_0[iteration] = body_force[0][:,yc].copy()
        list_BodyForce_1[iteration] = body_force[1][:,yc].copy()
        netForce = fc.ADD_BODY_FORCE * body_force + fc.ADD_SURFACE_TENSION_FORCE * zhang_surface_tension_force
        list_NetForce[iteration] = netForce[1][:,yc].copy()

        if fc.ADD_SURFACE_TENSION_FORCE:
            _chemical_potential_Zhang = chemical_potential(fc, _phi)

            #zhangChemicalPotential_center = _chemical_potential_Zhang[:,yc].copy()

            label=r'$\mu_\phi$'
            label_Zhang=r'$Zhang  \mu_\phi$'
            title_Zhang = "chem_pot_zhang"
            plotter.chemical_potential_map(None, _chemical_potential_Zhang, iteration, title_Zhang, label_Zhang)

            plotter.plot_capillary_forces(
                zhang_surface_tension_force,   # shape (Ny, Nx, 2)
                yc=None,
                iteration=iteration,
                title="zhang_surface_tension_force"
            )

    if ADD_METRICS and iteration in iterationsOfInterest:
        fcBounds_left.append((iteration,
                        zhang_surface_tension_force[0,1,:].copy(), # left-x  
                        zhang_surface_tension_force[1,1,:].copy())) # left-y 
        fcBounds_right.append((iteration,
                        zhang_surface_tension_force[0,Xn,:].copy(), # right-x  
                        zhang_surface_tension_force[1,Xn,:].copy())) # right-y   

        _, left_x, left_y = fcBounds_left[-1]
        plotter.phi_boundary_forces_vertical(
                    left_x, left_y,
                    label_left_x="Left wall Fx",
                    label_left_y="Left wall Fy",
                    yc=yc,
                    iteration=iteration,
                    title="Capillary forces on vertical boundary left",
                    figsize=(7, 5)
                )

        _, right_x, right_y = fcBounds_right[-1]
        plotter.phi_boundary_forces_vertical(
                    right_x, right_y,
                    label_left_x="Right wall Fx",
                    label_left_y="Right wall Fy",
                    yc=yc,
                    iteration=iteration,
                    title="Capillary forces on vertical boundary right",
                    figsize=(7, 5)
                )

    #Step2a. calculate h, p
    # Zhang eq(27) - hydrostatic pressure
    ############################ 2. Add Successive Over-Relaxation (SOR) with Residual Monitoring (Accelerate/Damp) #################
    p = zp(fc, z_fi, rho, u_ckl, iteration)  # Use damped div_u if added above
    #################################################################################################################################


    
    # In main loop, after u_ckl update:total_mom_x = np.sum(rho * u_ckl[0])  # Add this
    #if iteration == 0 or iteration==(fc.TOTAL_ITERATIONS - 1) or iteration % 100 == 0:
    #    if ADD_METRICS: 
    if ADD_METRICS and iteration in iterationsOfInterest:
        u_ckl_x_min = np.min(u_ckl[0])
        u_ckl_x_max = np.max(np.abs(u_ckl[0]))
        u_ckl_y_min = np.min(u_ckl[1])
        u_ckl_y_max = np.max(np.abs(u_ckl[1]))

        ssc = sufficient_stability_condition(u_ckl)
        osc = optimal_stability_condition(u_ckl)
        StabilityConditions.append((iteration, ssc, osc))

        invariant = np.sum(rho*u_ckl[0])
        MomentumBounds.append((iteration, u_ckl_x_min, u_ckl_x_max, invariant))
        rho_min = np.min(rho)
        rho_max = np.max(rho)    
        rho_bounds.append((iteration, rho_min, rho_max))
        Invariants.append((iteration, invariant))
        debug_log('FIELD', 'Iteration=%d; max|u_x|=%.2e; invariant=%.2e', iteration, u_ckl_x_max, invariant)
        GrowthMetric_uckl_x.append((iteration, u_ckl_x_max))
        GrowthMetric_uckl_y.append((iteration, u_ckl_y_max))

        uckl_star_y = np.max(np.abs(u_ckl[1]))
        GrowthMetric_uckl_star_y.append((iteration, uckl_star_y))
        du_dx, du_dy = zhang_gradient(u_ckl[0])
        dv_dx, dv_dy = zhang_gradient(u_ckl[1])
        div_u_raw = du_dx + dv_dy

        GrowthMetric_div_u_raw.append((iteration, np.max(np.abs(du_dx)), np.max(np.abs(du_dy)), np.max(np.abs(div_u_raw))))
        GrowthMetric_u_ckl_du_dy.append((iteration, du_dy))            

        spuriousField1 = np.max(np.abs(zhang_gradient(p)))
        laplacian_phi = zhang_laplacian(_phi)
        spuriousField2 = np.max(np.abs(laplacian_phi))
        SpuriousFields.append((iteration, spuriousField1, spuriousField2))

        ################################# 6. Additional Diagnostics and Rollbacks ######################################
        DivU_max.append((iteration, np.max(np.abs(div_u_raw))))
        ################################################################################################################
        #  NEW – collect vertical integrals of φ
        phi_total = np.sum(_phi[1:Xn+1, 1:Yn+1])
        PhiCollector.append((iteration, phi_total))
        phi_on_plane = _phi[1, 1:Yn+1]
        PhiOnPlaneCollector_0.append((iteration, phi_on_plane))
        phi_on_plane = _phi[(Xn+1)//2, 0:Yn]
        PhiOnPlaneCollector_1.append((iteration, phi_on_plane))
        phi_on_plane = _phi[1, 1:Yn+1]

    #streaming has commenced
   
    # Update plots and parameters
    _rho_full_range = rho
    if iteration in iterationsOfInterest:
        list_avg_velocities_x[iteration] = u_ckl[0, 1:-1, :]
        list_avg_velocities_y[iteration] = u_ckl[1, 1:-1, :]

        #plot force per iteration of interest
  

    if iteration == 0 or iteration==(fc.TOTAL_ITERATIONS - 1) or iteration % 100 == 0:
        epsilon_u_ckl = np.abs(u_ckl[0,int(Xn/2),int(Yn/2)] - u_ckl_midpoint0)
        epsilon_u_ckl_list.append((iteration, epsilon_u_ckl))    
        u_ckl_midpoint0 = u_ckl[0,int(Xn/2),int(Yn/2)]


    # plot in real time - color 1/2 particles blue, other half red
    if (PLOTREALTIME and (iteration % 10) == 0):
        plotter.update(iteration, _phi, rho, u_ckl)

    if (PLOTREALTIME and (iteration == fc.TOTAL_ITERATIONS - 1)):
        plotter.update(iteration, _phi, rho, u_ckl)        

    #Step 4: re-iterate
    iteration += 1
    if iteration % 100 == 0:
        progress = (iteration / fc.TOTAL_ITERATIONS) * 100.0
        debug_log('ITER', 'Simulation Execution -> TOTAL_ITERATIONS: %d; iteration: %d; %.1f %%', 
          fc.TOTAL_ITERATIONS, iteration, progress)
        

    if ADD_METRICS and iteration in iterationsOfInterest:
        print(f"iter {iteration}: u_max={np.max(np.abs(u_ckl)):.4e}, phi_min={np.min(_phi):.4e}, phi_max={np.max(_phi):.4e}")        
        if len(list_phi.keys()) > 0:
            phi_center    = list_phi.popitem()[1]
            dPhi_center0  = list_dphi_0.popitem()[1]
            dPhi_center1  = list_dphi_1.popitem()[1]

            label1 = r'$\phi$'
            label2 = r'$\partial \phi_x$'
            label3 = r'$\partial \phi_y$'
            plotter.phi_x_axis_plot_3(
                None,
                phi_center, dPhi_center0, dPhi_center1,
                label1, label2, label3,
                axis1=0, axis2=1, axis3=1,  # numeric axes
                yc=yc,
                iteration=iteration,
                title=f"_phi + _phid distribution y={yc}"
            )

    if iteration in iterationsOfInterest:
        lam_n = Cs2 * z_lambda(fc, _phi) * n(fc, _phi)
        dphi_u = _phi[np.newaxis] * u_ckl - __phi_old[np.newaxis] * _u_ckl_old
        print("cs2_lambda_n max:", np.max(np.abs(lam_n)))
        print("dphi_u_dt    max:", np.max(np.abs(dphi_u)))
        print("phi at x=150 (the midcolumn):", _phi[150, :])      

        zhang_interfacial_tension_check(fc, _phi, iteration, check_every=10)    



end = time.perf_counter()
#iterationsOfInterest = [0, 10, 50, 100, 200, 500, 1000, 5000, 10000, 12000]
diff = end - start

plotter.save_phi_snapshot(_phi, iteration - 1, fc.phi_star_G, fc.phi_star_L)

rho_in, rho_out = _rho_full_range[1, Yn // 2], _rho_full_range[Xn, Yn // 2]
rho_min = np.min(_rho_full_range)
rho_max = np.max(_rho_full_range)
debug_log('FIELD', '_rho_full_range min = %(min).6f, max = %(max).6f', extra=dict(min=rho_min, max=rho_max))

# height ratios based on lattice dimensions
aspect_ratio = Xn / Yn
top_row_height = 1 #1.5
bottom_row_height = 1
height_ratios0 = [
    top_row_height,
    bottom_row_height,
    bottom_row_height,
    bottom_row_height
]

height_ratios1 = [top_row_height, bottom_row_height, bottom_row_height] if 'top_row_height' in globals() else [1, 1, 1]

# 4x2 multi-plot grid
paneLabel = f"Dashboard D2Q9 LB method for incompressible two-phase flows Zhang et al 2020 Lattice [{Xn} {Yn}] Single processor"
fig1, ax1 = plt.subplots(
    4, 2,
    figsize=(15, 10),
    gridspec_kw={
        'width_ratios': [2, 4], 
        'height_ratios': height_ratios0,
        'left': 0.15, 'right': 0.85, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

# In the fig1, ax1 section
sectionPosition = int(Xn/2)

# avg_velocities_x, avg_velocities_y
filtered_u_ckl_dict_x = plotter.filter_u_ckl_fullrange(list_avg_velocities_x, iterationsOfInterest)
filtered_u_ckl_list_x = list(filtered_u_ckl_dict_x.values())

filtered_u_ckl_dict_y = plotter.filter_u_ckl_fullrange(list_avg_velocities_y, iterationsOfInterest)
filtered_u_ckl_list_y = list(filtered_u_ckl_dict_y.values())

U_max_x = np.max(filtered_u_ckl_list_x[-1][sectionPosition, 1:Yn+1])
plotter.amplitude_plot(ax1[0, 0], filtered_u_ckl_dict_x, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_x$", f"Amplitude u$_x$ at x={Xn}", sectionPosition, Yn)
plotter.amplitude_plot(ax1[1, 0], filtered_u_ckl_dict_y, iterationsOfInterest, np.arange(1, Yn + 1), "y-axis", "Amplitude u$_y$", f"Amplitude u$_y$ at x={Xn}", sectionPosition, Yn)

# phi plots at centerline
iteration, phi_on_plane = PhiOnPlaneCollector_1[-1]
plotter.phi_profile(phi_on_plane, f"phi_profile_", iteration=iteration)

_iteration = fc.TOTAL_ITERATIONS
plotter.velocity_map(ax1[0, 1], filtered_u_ckl_list_x[-1][1:-1, 1:Yn+1], _iteration, "Velocity [u$_x$] map")
plotter.velocity_map(ax1[1, 1], filtered_u_ckl_list_y[-1][1:-1, 1:Yn+1], _iteration, "Velocity [u$_y$] map")

plotter.density_profiles(ax1[2, 0], density_slices, density_profile_x_position, Xn, Yn, iteration)

if PRESSURE_IN_DENSITY_MAP:
    min_value = 0 
    _pressure_full_range = (_rho_full_range - min_value) * Cs2 
    _pressure_out = (rho_min - min_value) * Cs2
    _pressure_in = (rho_max - min_value) * Cs2
    title = "Pressure map"
    plotter.density_mapExt(ax1[2, 1], _pressure_full_range, _pressure_out, _pressure_in, title, iteration)
else:
    title = "Density map"
    plotter.density_mapExt(ax1[2, 1], _rho_full_range, rho_min, rho_max, title, iteration)

BodyForce_center_0 = list_BodyForce_0.popitem()[1]
BodyForce_center_1 = list_BodyForce_1.popitem()[1]
NetForce_center = list_NetForce.popitem()[1]

#phi: Phi, dPhix, dPhiy
label1 = r'$\phi$'
label2 = r'$\partial \phi_x$'
label3 = r'$\partial \phi_y$'
phi_center    = list_phi.popitem()[1]
dPhi_center0  = list_dphi_0.popitem()[1]
dPhi_center1  = list_dphi_1.popitem()[1]
plotter.phi_x_axis_plot_3(
    ax1[3,0],
    phi_center, dPhi_center0, dPhi_center1,
    label1, label2, label3,
    axis1=0, axis2=1, axis3=1,  # numeric axes
    yc=yc,
    iteration=iteration,
    title=f"_phi + _phid distribution y={yc}"
)

#chemical potential: Zhang
if fc.ADD_SURFACE_TENSION_FORCE:
    zhangChemicalPotential_center = _chemical_potential_Zhang[:,yc].copy()
    label1=r'$Zhang  \mu_\phi$'
    plotter.phi_x_axis_plot_1(ax1[3, 1], zhangChemicalPotential_center, yc, iteration, f"ChemicalPotential distribution y={yc}", label1)

text = f"Run-time: {diff:.1f} s"
fig1.text(0.5, 0.98, text, ha='center', va='top', fontsize=12)
fig1.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_{ACTIVE_CASE}_channel_parameters.png")
fig1.savefig(save_path, dpi=300, bbox_inches='tight')
debug_log('INIT', 'Saved 3x2 grid: %s', save_path)
plt.close(fig1)


# 3 rows, 4 columns
paneLabel = f"Metrics - D2Q9 LB method for incompressible two-phase ﬂows Zhang et al 2020 Lattice [{Xn} {Yn}] Single processor"
fig2, ax2 = plt.subplots(
    3, 4,  
    figsize=(18, 10), 
    gridspec_kw={
        'width_ratios': [1, 1, 1, 1],
        'height_ratios': height_ratios1,
        'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1,
        'wspace': 0.3, 'hspace': 0.4
    },
    sharey=False,
    num=paneLabel
)

if ADD_METRICS: 
    plotter.plot_bounds_ext(GrowthMetric_uckl_x, "GrowthMetric_uckl_x", ax2[0, 0])
    plotter.plot_bounds_ext(GrowthMetric_uckl_y, "GrowthMetric_uckl_y", ax2[0, 1])
    plotter.plot_bounds_ext(GrowthMetric_uckl_star_y, "GrowthMetric_uckl_star_y", ax2[0, 2])
    
    plotter.plot_bounds_ext(epsilon_u_ckl_list, "epsilon_u_ckl growth", ax2[1, 0])
    plotter.plot_bounds_ext(Invariants, "Invariants", ax2[1, 1])
    series_labels = ["ei ⋅ u/cs2", "wiϕ(1 + ei ⋅ u/cs2)"]
    plotter.plot_bounds_ext(PhiTerms, "PhiTerms", ax2[1, 2], series_labels)

    series_labels = ["sufﬁcient stability condition","optimal stability condition"]
    plotter.plot_bounds_ext(StabilityConditions, "stability conditions", ax2[2, 0], series_labels)

    series_labels = ["c_first_derivative0(p)","laplacian_phi"]
    plotter.plot_bounds_ext(SpuriousFields, "SpuriousFields", ax2[2, 1], series_labels)


    if fc.ADD_SURFACE_TENSION_FORCE == 1:
        _, left_x, left_y = fcBounds_left[-1]
        plotter.phi_boundary_forces_vertical(
                    left_x, left_y,
                    label_left_x="Left wall Fx",
                    label_left_y="Left wall Fy",
                    yc=yc,
                    iteration=iteration,
                    title="Capillary forces on vertical boundary left",
                    figsize=(7, 5)
                )

        _, right_x, right_y = fcBounds_right[-1]
        plotter.phi_boundary_forces_vertical(
                    right_x, right_y,
                    label_left_x="Right wall Fx",
                    label_left_y="Right wall Fy",
                    yc=yc,
                    iteration=iteration,
                    title="Capillary forces on vertical boundary right",
                    figsize=(7, 5)
                )

    plotter.plot_bounds_ext(PhiCollector, "Mass Conservation (Intg. _phi)", ax2[2, 3])

fig2.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
save_path = os.path.join(images_dir, f"{SCRIPT_FILENAME}_{ACTIVE_CASE}_Metrics_{fc.TOTAL_ITERATIONS:0{fc.FILENAME_PADDING_WIDTH}d}.png")
fig2.savefig(save_path, dpi=300, bbox_inches='tight')
debug_log('INIT', 'Saved 3x4 grid: %s', save_path)
plt.close(fig2)


########### upload this file and results to GitHub repo
uploader = GitHubUploader(
    debug_log=debug_log,
    timeout=60,
    script_filename=SCRIPT_FILENAME,
    script_full_path=SCRIPT_FULL_PATH,
    scripts_path=SCRIPTS_PATH,
    plots_path=PLOTS_PATH,
    #images_subdir=IMAGES_SUBDIR,
    images_subdir=images_dir,
    log_file=LOG_FILE,
    token_file='github-repo-token.txt'
)
try:
    if fc.WRITE_TO_GITHUB:
        uploader.upload_results(upload_log=True)
        debug_log('INIT', f'Upload complete: Script at root, results in https://github.com/faircm2/Lb-Python/tree/main/{uploader.results_folder}')
except Exception as e:
    debug_log('ERROR', f'Upload failed: {e}')

# Example usage at the end of your script:
# Assuming SCRIPT_FILENAME = 'your_script.py'
# uploader = GitHubUploader(SCRIPT_FILENAME, repo_name='yourusername/your-repo-name')
# uploader.upload_results()