import numpy as np 
import matplotlib.pyplot as plt
import math
import random as rdm
import numpy.typing as npt
import os

mStepSize = 0.1

time = 1

N = 100   # Nombre de particules
D = 2     # Dimension du problème

positions = np.zeros([D, N])           # Crée un tableau 2D avec des zéros
velocities = np.zeros_like(positions) 

masses = np.ones([N])  # Crée un tableau 1D avec des 1
forces = np.zeros_like(positions)

def verlet_predictor(positions: np.ndarray[float],
                     velocities: np.ndarray[float],
                     masses: np.ndarray[float],
                     forces: np.ndarray[float],
                     dt: float):
    

    velocities += 0.5*dt*forces/(masses)

    positions += dt*velocities

def verlet_corrector(positions: np.ndarray[float],
                     velocities: np.ndarray[float],
                     masses: np.ndarray[float],
                     forces: np.ndarray[float],
                     dt: float):
    velocities += 0.5*dt*forces/(masses)


def analytical_position(initial_velocity: np.ndarray[float],
                        gravity: np.ndarray[float],
                        t: float):
    final_position = np.zeros([D, N])

    final_position[0,:] = initial_velocity[0,:]*t
    final_position[1,:] = initial_velocity[1,:]*t + gravity[1,:]*0.5*(t**2)
    return final_position

def write_xyz(fname: str, positions: np.ndarray[float], symbol: str = None):
    """Writes an XYZ file from atom positions (2d or 3d)"""
    natoms = positions.shape[1]

    dtype = [('symbols', np.str_, 2),
             ('positions', float, 3)]
    dtype = np.dtype(dtype)
    
    extended_pos = np.zeros(natoms, dtype=dtype)
    extended_pos['positions'][:, :positions.shape[0]] = positions.T
    extended_pos['symbols'][:] = 'H' if symbol is None else symbol
    
    with open(fname, 'w') as fh:
        fh.write(str(natoms) + '\n\n')
        for row in extended_pos:
            s, x = row['symbols'], row['positions']
            fh.write(f"{s} {x[0]:.18e} {x[1]:.18e} {x[2]:.18e}\n")

rng = np.random.default_rng()  # crée un générateur de nombres aleatoires
initial_velocity = rng.uniform(-1, 1, velocities.shape)  # rempli initial_velocity de nombres distribues entre -1 et 1

velocities[:] = initial_velocity[:]

# Étape 1 - donner la bonne valeur (en fonction de g) aux forces (la gravité s'applique selon y uniquement)


for i in range(len(forces[1,:])):

    forces[1,i] = -9.81
# Étape 2 - la boucle d'intégration
for i in range(10):

    verlet_predictor(positions,velocities,masses,forces,mStepSize)
    
    verlet_corrector(positions,velocities,masses,forces,mStepSize)

    write_xyz(f"traj_{i:04d}.xyz",positions)

positions_analytical = analytical_position(initial_velocity,forces,time)

# Pour tester la position des particules, on peut se servir de np.testing.assert_allclose
# Cette fonction donne une erreur si les valeurs ne sont pas identiques
tolerance = 1e-14
np.testing.assert_allclose(positions, positions_analytical, atol=tolerance)

