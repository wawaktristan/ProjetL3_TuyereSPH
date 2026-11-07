'''
Auteur : Tristan WAWAK

Genere particule en amont de la tuyere avec grandeur initiales

liste fonction :

intializeParticlesFromCombustionChamber1 NE PAS UTILISER!!!!!!!!!!!!

intializeParticlesFromCombustionChamber

Genere une ligne de particule separé d'une distance combustionChamberRadius/ny 

'''

import numpy as np

def intializeParticlesFromCombustionChamber1(combustionChamberRadius, particleMass, particlePressure, particleTemperature, particleDensity,particleVelocityX, ny=10, ):

    latticeSpacing = combustionChamberRadius/ny

    nx=1

    angle=np.pi / 2

    ei = np.array([1.0, 0.0, 0.0])
    ej = np.array([np.cos(angle), np.sin(angle), 0.0])
    i, j = np.mgrid[0:nx, 0:ny]
    lattice = (
        ei[np.newaxis, np.newaxis] * i[..., np.newaxis]
        + ej[np.newaxis, np.newaxis] * j[..., np.newaxis]
    )
    lattice = lattice.reshape(nx * ny, -1).T
    shifts = np.floor(lattice[0] / nx)
    lattice[0] -= shifts * nx
    lattice[1] -= (combustionChamberRadius)/2-0.5

    

    positions = lattice * latticeSpacing

    masses = np.full(ny, particleMass)

    densities = np.full(ny, particleMass*ny/(combustionChamberRadius))


    velocities = np.zeros_like(positions)

    velocities[0,:] = particleVelocityX

    pressures = np.full(ny,particlePressure)

    forces = np.zeros_like(positions)

    temperatures = np.full(ny,particleTemperature)


    return positions,velocities,masses,pressures,forces,densities,temperatures

def intializeParticlesFromCombustionChamber(combustionChamberRadius,
                                            particleMass,
                                            particlePressure,
                                            particleTemperature,
                                            particleDensity,
                                            particleVelocityX,
                                            ny=10):

    nx = 1
    n_total = nx * ny

    dy = 2.0 * combustionChamberRadius / ny
    dx = dy  
    i, j = np.mgrid[0:nx, 0:ny]             
    i = i.ravel()                           
    j = j.ravel()

    positions = np.zeros((2, n_total))
    positions[0, :] = i * dx
    positions[1, :] = (j - (ny - 1) / 2.0) * dy

    masses    = np.full(n_total, particleMass)
    pressures = np.full(n_total, particlePressure)
    densities = np.full(n_total, particleMass * n_total / (2.0 * combustionChamberRadius))
    velocities = np.zeros_like(positions)
    velocities[0, :] = particleVelocityX
    forces = np.zeros_like(positions)
    temperatures = np.full(ny,particleTemperature)

    return positions, velocities, masses, pressures, forces, densities, temperatures