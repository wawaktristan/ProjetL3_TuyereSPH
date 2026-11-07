'''
Auteur: TGristan Wawak

Geometrie de la tuyere soit forme linaire soit cosinus

Reflection des particules sur les bords

'''

import numpy as np

def nozzle_radius_and_slope(x, x_inlet, x_throat, x_exit,
                             R_inlet, R_throat, R_exit, profile="cosine"):
    R    = np.empty_like(x, dtype=float)
    dRdx = np.zeros_like(x, dtype=float)

    conv = (x >= x_inlet) & (x <= x_throat)
    div  = (x >  x_throat) & (x <= x_exit)
    before = x < x_inlet
    after  = x > x_exit

    if profile == "cosine":
        # Convergent
        L = x_throat - x_inlet
        t = (x[conv] - x_inlet) / L
        R[conv]    = R_inlet + (R_throat - R_inlet) * (1 - np.cos(np.pi * t)) / 2
        dRdx[conv] = (R_throat - R_inlet) * np.pi * np.sin(np.pi * t) / (2 * L)

        # Divergent
        L = x_exit - x_throat
        t = (x[div] - x_throat) / L
        R[div]    = R_throat + (R_exit - R_throat) * (1 - np.cos(np.pi * t)) / 2
        dRdx[div] = (R_exit - R_throat) * np.pi * np.sin(np.pi * t) / (2 * L)

    else:  # linear
        L = x_throat - x_inlet
        R[conv]    = R_inlet + (R_throat - R_inlet) * (x[conv] - x_inlet) / L
        dRdx[conv] = (R_throat - R_inlet) / L

        L = x_exit - x_throat
        R[div]    = R_throat + (R_exit - R_throat) * (x[div] - x_throat) / L
        dRdx[div] = (R_exit - R_throat) / L

    R[before] = R_inlet
    R[after]  = R_exit
    # dRdx stays 0 outside the nozzle (flat caps)

    return R, dRdx


def reflectParticlesOnWall(positions: np.ndarray,
                           velocities: np.ndarray,
                           x_inlet:  float = 0.0,
                           x_throat: float = 0.5,
                           x_exit:   float = 1.0,
                           R_inlet:  float = 0.30,
                           R_throat: float = 0.10,
                           R_exit:   float = 0.40,
                           profile:  str   = "cosine",
                           restitution: float = 1.0):

    dim = positions.shape[0]
    x = positions[0, :]

    if dim == 2:
        y = positions[1, :]
        r = np.abs(y)
        rhat_y = np.sign(y)
        rhat_y[rhat_y == 0] = 1.0
        rhat_z = None
    elif dim == 3:
        y = positions[1, :]
        z = positions[2, :]
        r = np.sqrt(y*y + z*z)
        r_safe = np.where(r > 1e-12, r, 1.0)
        rhat_y = y / r_safe
        rhat_z = z / r_safe
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    R_max, dRdx = nozzle_radius_and_slope(
        x, x_inlet, x_throat, x_exit, R_inlet, R_throat, R_exit, profile
    )
    #reflection
    outside = r > R_max
    if np.any(outside):
        idx = np.where(outside)[0]
        slope = dRdx[idx]

        norm_denom = np.sqrt(1.0 + slope**2)
        n_x = -slope / norm_denom
        n_r =  1.0   / norm_denom  # radial component magnitude

        if dim == 2:
            n_y = n_r * rhat_y[idx]
            vx = velocities[0, idx]
            vy = velocities[1, idx]
            v_dot_n = vx * n_x + vy * n_y
            # Only reflect particles actually moving INTO the wall
            hitting = v_dot_n > 0
            if np.any(hitting):
                k = (1.0 + restitution) * v_dot_n[hitting]
                sel = idx[hitting]
                velocities[0, sel] -= k * n_x[hitting]
                velocities[1, sel] -= k * n_y[hitting]

            # Project position back onto the wall along radial direction
            positions[1, idx] = R_max[idx] * rhat_y[idx]

        else:  # 3D
            n_y = n_r * rhat_y[idx]
            n_z = n_r * rhat_z[idx]
            vx = velocities[0, idx]
            vy = velocities[1, idx]
            vz = velocities[2, idx]
            v_dot_n = vx * n_x + vy * n_y + vz * n_z
            hitting = v_dot_n > 0
            if np.any(hitting):
                k = (1.0 + restitution) * v_dot_n[hitting]
                sel = idx[hitting]
                velocities[0, sel] -= k * n_x[hitting]
                velocities[1, sel] -= k * n_y[hitting]
                velocities[2, sel] -= k * n_z[hitting]

            positions[1, idx] = R_max[idx] * rhat_y[idx]
            positions[2, idx] = R_max[idx] * rhat_z[idx]

   #
    before_inlet = x < x_inlet
    if np.any(before_inlet):
        moving_out = velocities[0, before_inlet] < 0
        sel = np.where(before_inlet)[0][moving_out]
        velocities[0, sel] *= -restitution
        positions[0, before_inlet] = x_inlet

    after_exit = x > x_exit
    if np.any(after_exit):
        moving_out = velocities[0, after_exit] > 0
        sel = np.where(after_exit)[0][moving_out]
        velocities[0, sel] *= -restitution
        positions[0, after_exit] = x_exit

    return positions, velocities

def nozzle_radius_profile(x, inletX, throatX, exitX, rIn, rTh, rEx, profile="cosine"):

    r = np.zeros_like(x)
    # Partie convergente
    mask_conv = (x >= inletX) & (x <= throatX)
    if profile == "cosine":
        xi = (x[mask_conv] - inletX) / (throatX - inletX)
        r[mask_conv] = rTh + 0.5 * (rIn - rTh) * (1 + np.cos(np.pi * xi))
    else:
        r[mask_conv] = rIn + (rTh - rIn) * (x[mask_conv] - inletX) / (throatX - inletX)
    # Partie divergente 
    mask_div = (x > throatX) & (x <= exitX)
    if profile == "cosine":
        xi = (x[mask_div] - throatX) / (exitX - throatX)
        r[mask_div] = rTh + 0.5 * (rEx - rTh) * (1 - np.cos(np.pi * xi))
    else:
        r[mask_div] = rTh + (rEx - rTh) * (x[mask_div] - throatX) / (exitX - throatX)
    return r

