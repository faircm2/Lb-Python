# Formula for generating herringbone pattern on smooth microchannel floor
# [10] 	G. C. Saliba, J. G. Korvink und J. J. Brandner, „Magnetic resonance velocimetry reveals secondary flow in falling films at the microscale,“ Physics of Fluids, Bd. 36, Nr. 7, p. 071705, July 2024. 

import os
import numpy as np
from dataclasses import dataclass


@dataclass
class Herringbone:
    width: float = 1.2             # channel width (mm), only used for the side-wall term of the distance
    depth: float = 0.2             # groove depth
    groove_width: float = 0.2      # perpendicular to the groove
    groove_pitch: float = 0.4      # perpendicular, groove to groove
    groove_angle_deg: float = 45.0        # between groove and across-channel axis
    apex_offset: float = 0.2       # |t| of the apex
    apex_flip: bool = True         # apex alternates between -offset and +offset from train to train
    n_per_train: int = 6
    train_period: float = float(os.environ.get("TP", 3.83))     # s distance between train starts (env TP overrides; train is 3.9 long, so TP=4.9 leaves a 1.0 gap)
    x_start: float = 1.0           # s of the first groove front edge at its apex
    n_trains: int = 2
    z_floor: int = 15
    wrap: bool = False             # True: train_period = box length / n_trains, pattern wraps around periodic x


def groove_r(s, t, s0, t_apex, groove_angle_deg):
    groove_angle_rad= np.radians( groove_angle_deg)
    s_edge = np.where(t < t_apex, s0 + (t_apex - t)*np.tan(groove_angle_rad),s0 + (t - t_apex)*np.tan(groove_angle_rad))

    r = (s-s_edge) * np.cos(groove_angle_rad)
    return r


def in_groove_r(s, t, s0, t_apex, groove_angle_deg, Breite):
    r = groove_r(s, t, s0, t_apex, groove_angle_deg)
    r_test = (-1e-9 <= r) & (r <= Breite)
    return r_test 


def in_train_r(s, t, s0, t_apex, groove_angle_deg, Breite, Pitch, n_per_train):
    r_train = np.zeros(np.shape(s), dtype=bool)
    angle_from_flow_rad= np.radians(90-groove_angle_deg)
    for k in range(n_per_train):
        s0_k = s0 + k * Pitch / np.sin(angle_from_flow_rad)
        r_train = r_train | in_groove_r(s, t, s0_k , t_apex, groove_angle_deg, Breite)

    return r_train


#Next, step 2: in_pattern. It loops over the 7 trains j, with s0_train = z_start + j * train_period and t_apex = t_apex_list[j % 2], calls in_train_r and ORs the results together.
def in_pattern(x_start, n_trains, train_period, s, t, groove_angle_deg, Breite, Pitch, n_per_train):
    result = np.zeros(np.shape(s), dtype=bool)
    # the two tip positions
    t_apex_list = [-0.2, +0.2]
    for j in range(n_trains):
        t_apex = t_apex_list[j % 2]
        s0_train = x_start + j * train_period
        result = result | in_train_r(s, t, s0_train, t_apex, groove_angle_deg, Breite, Pitch, n_per_train)

    return result


def is_solid(s, t, h, depth, x_start, n_trains, train_period, groove_angle_deg, Breite, Pitch, n_per_train):
    solid = (h < -depth) | ((h < 0) & ~in_pattern(x_start, n_trains, train_period, s, t, groove_angle_deg, Breite, Pitch, n_per_train))
    return solid


def herringbone_mask(Xn, Yn, Zn, hb):
    i = np.arange(0, Xn + 2)          # cell numbers incl. the ghost layers 0 and N+1
    j = np.arange(0, Yn + 2)
    k = np.arange(0, Zn + 2)
    dx = hb.width / Yn
    assert hb.depth <= hb.z_floor * dx, "floor thinner than the groove depth"
    s = ((i - 0.5) * dx)[:, None, None]
    t = ((j - 0.5) * dx - Yn * dx / 2)[None, :, None]
    h = ((k - 0.5 - hb.z_floor) * dx)[None, None, :]    

    #s = np.array([1.4, 1.05, 1.4, 1.05])
    #t = np.array([-0.2, -0.2, -0.2, -0.2])
    #h = np.array([0.1, -0.1, -0.1, -0.3])

    if hb.wrap and hb.n_trains > 0:
        # periodic box: the n_trains trains exactly fill Xn cells, the tail of the last train wraps to x=0
        L = Xn * dx
        train_period = L / hb.n_trains
        open_ = np.zeros(np.broadcast(s, t, h).shape, dtype=bool)
        for shift in (-L, 0.0, L):
            open_ = open_ | in_pattern(hb.x_start, hb.n_trains, train_period, s - shift, t, hb.groove_angle_deg, hb.groove_width, hb.groove_pitch, hb.n_per_train)
        return (h < -hb.depth) | ((h < 0) & ~open_)

    isSolid = is_solid(s, t, h, hb.depth, hb.x_start, hb.n_trains, hb.train_period, hb.groove_angle_deg, hb.groove_width, hb.groove_pitch, hb.n_per_train)

    return isSolid


def herringbone_test():
    s = np.array([1.4, 1.05, 1.4, 1.05])
    t = np.array([-0.2, -0.2, -0.2, -0.2])
    h = np.array([0.1, -0.1, -0.1, -0.3])    
    depth = 0.2
    x_start = 1.0
    n_trains = 2
    train_period = 3.83
    groove_angle_deg = 45
    groove_width = 0.2
    groove_pitch = 0.4
    n_per_train = 6

    hb = Herringbone()
    mask = herringbone_mask(512, 64, 47, hb)
    print(mask.shape, mask.mean())  

    ret = is_solid(s, t, h, depth, x_start, n_trains, train_period, groove_angle_deg, groove_width, groove_pitch, n_per_train)
    print(ret)           


if __name__ == "__main__":
    herringbone_test()
