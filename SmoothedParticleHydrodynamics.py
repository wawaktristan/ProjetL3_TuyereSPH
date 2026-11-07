"""
Implémentation SPH 2D basée sur la liste de voisins matscipy.

Conventions
positions : array (D, N), D=2
velocities : array (D, N)
masses, densities, pressures : array (N,)
Pour les paires de voisins, on travaille avec des tableaux 1D de longueur  : i_p, j_p, d_p (distance), r_p (D, P) = r_i - r_j
Convention SPH : r_ij = r_i - r_j, donc nij = (r_i - r_j) / |r_i - r_j|
Noyau : cubic spline 2D, normalisation 40/(7 pi h^2)
"""

import numpy as np
from matscipy.neighbours import neighbour_list

#############
# liste voisin
##############
def get_neighbours(positions, cutoff, domain=None, periodicity=None):

    D, N = positions.shape

    if domain is None:
        domain = np.max(positions, axis=1) - np.min(positions, axis=1)
        domain[domain == 0] = np.max(domain) + 1e-12
    domain = np.asanyarray(domain)
    if domain.ndim == 1:
        domain = np.diag(domain)
    if domain.shape == (2, 2):
        ext = np.eye(3)
        ext[:2, :2] = domain
        domain = ext

    if periodicity is None:
        periodicity = np.array([False, False, False])
    periodicity = np.asanyarray(periodicity)
    if periodicity.shape[0] == 2:
        periodicity = np.concatenate((periodicity, [False]))

    if D == 2:
        full_pos = np.vstack((positions, np.zeros(N)))
    else:
        full_pos = positions

    i, j, d, r_ase = neighbour_list(
        'ijdD',
        positions=full_pos.T,
        cell=domain,
        pbc=periodicity,
        cutoff=float(cutoff),
    )
    r = -r_ase.T  
    r = r[:D]    

    return i, j, d, r

def kernel_W(d, h):

    alpha = 40.0 / (7.0 * np.pi * h * h)
    q = d / h
    W = np.zeros_like(d)
    m1 = q <= 0.5
    m2 = (q > 0.5) & (q < 1.0)
    W[m1] = 6.0 * q[m1]**3 - 6.0 * q[m1]**2 + 1.0
    W[m2] = 2.0 * (1.0 - q[m2])**3
    return alpha * W


def kernel_dWdr(d, h):

    alpha = 40.0 / (7.0 * np.pi * h * h)
    q = d / h
    dW = np.zeros_like(d)
    m1 = q <= 0.5
    m2 = (q > 0.5) & (q < 1.0)

    dW[m1] = (18.0 * q[m1]**2 - 12.0 * q[m1]) / h

    dW[m2] = -6.0 * (1.0 - q[m2])**2 / h
    return alpha * dW

def compute_density(N, masses, i, j, d, h):

    W = kernel_W(d, h)
    rho = np.zeros(N)
    np.add.at(rho, i, masses[j] * W)
    # auto-contribution : W(0) = alpha
    W0 = 40.0 / (7.0 * np.pi * h * h)
    rho += masses * W0
    return rho


def compute_pressure_forces(N, masses, densities, pressures, i, j, d, r, h):

    D = r.shape[0]
    dW = kernel_dWdr(d, h)            # (P,)
    n_ij = r / d[np.newaxis, :]       # (D, P), normalisé

    coeff = (pressures[i] / densities[i]**2
             + pressures[j] / densities[j]**2)        # (P,)
    factor = -masses[j] * coeff * dW                  # (P,)

    contrib = factor[np.newaxis, :] * n_ij            # (D, P)

    acc = np.zeros((D, N))
    for k in range(D):
        np.add.at(acc[k], i, contrib[k])
    return acc

def compute_viscosity_forces(N, masses, densities, velocities,
                             i, j, d, r, h, eta=0.01):

    D = r.shape[0]
    v_ij = velocities[:, i] - velocities[:, j]            # (D, P)
    vdotr = np.sum(v_ij * r, axis=0)                      # (P,) <-- vrai produit scalaire

    dW = kernel_dWdr(d, h)
    n_ij = r / d[np.newaxis, :]

    denom = densities[i] * densities[j] * (d**2 + eta * h**2)
    factor = 2.0 * masses[j] * vdotr / denom * dW         # (P,)

    contrib = factor[np.newaxis, :] * n_ij                # (D, P)

    acc = np.zeros((D, N))
    for k in range(D):
        np.add.at(acc[k], i, contrib[k])
    return acc


def temperature_from_pressure(densities, pressures, R=287.05):

    return pressures / (densities * R)


def pressure_from_temperature(densities, temperatures, R=287.05):

    return densities * R * temperatures

def calcul_mach(velocities, temperatures, gamma=1.4, R=287.05):

    v = np.sqrt(velocities[0,:]**2 + velocities[1,:]**2)

    c = np.sqrt(gamma * R * temperatures)

    return v / c

def calcul_vx_depuis_mach(mach, vy, temperatures, gamma=1.4, R=287.05, signe=1):

    c = np.sqrt(gamma * R * temperatures)
    v = mach * c

    vx_carre = v**2 - vy**2

    if np.any(vx_carre < 0):
        raise ValueError(
            "impossible de retrouver vx (racine négative)."
        )

    return signe * np.sqrt(vx_carre)

def compute_dij_nij(positions: np.ndarray[float]):

    N = positions.shape[1]  # le nombre de particules
    D = positions.shape[0]  # la dimension physique du problème (ici 2D)
    rij = np.zeros([D, N, N]) # on définit la matrice des vecteurs entre particules
    dij = np.zeros([N,N])

    rij = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]

    dij = np.linalg.norm(rij, axis=0) 

    nij = rij/dij

    np.nan_to_num(nij, copy=False, nan=0.)

    return dij, nij

def isentropique(T0, densities, density, gamma  = 1.4):
    
    return T0*(densities/density)**(gamma-1)

def pression(rho, k, rho_0):

    return k * (rho / rho_0 - 1.0)