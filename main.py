'''
Author : Tristan Wawak

'''

import Utility as util
import particleGenerator as pG
import numpy as np
import geometry as geo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matscipy.neighbours import neighbour_list
import SmoothedParticleHydrodynamics as SPH

#######################################################################
# GEOMETRY
######################################################################

inletXaxisPosition = 0.0
throatXaxisPosition= 5*2.54*1e-2
exitXaxisPosition= 10*2.54*1e-2
inletRadius = 2.26*1e-2
throatRadius = 1.4224*1e-2
exitRadius = 1.7526*1e-2
profileType = "cosine"


########################################################################
#  Particle Initial Properties
##################################################################

pressureEndlimit = 0.0689476*0.89*10e5
viscosity = 1.83e-5
mass = 1
pressure = 0.0689476*10e5
density = 1.2
machini = 0.2
temp = 55
Ny = 500

velocityX = SPH.calcul_vx_depuis_mach(machini,0,temp)
deltaT = (exitXaxisPosition+throatXaxisPosition)/(velocityX)*0.005

cutoff =  3*(2 * inletRadius / Ny)

WallStiffness = mass*pressure/deltaT**2

c_crit = 2 * np.sqrt(WallStiffness * mass)

#######################################################################
# Initial Conditions of system
#########################################################

positionsInitial, velocitiesInitial, massesInitial, pressuresInitial, forcesInitial, densitiesInitial, temperaturesInitial = pG.intializeParticlesFromCombustionChamber(inletRadius,mass,pressure,temp,density,velocityX,Ny)

positions = positionsInitial
print(positions)
velocities = velocitiesInitial
masses = massesInitial
forces = forcesInitial
pressures = pressuresInitial
densities = densitiesInitial
temperatures = temperaturesInitial

#######################################################################
# Plot stuff
#########################################################

time_history = []

# Mach aux 3 sections
mach_inlet_history = []
mach_throat_history = []
mach_exit_history = []

# Pression statique aux 3 sections
p_inlet_history = []
p_throat_history = []
p_exit_history = []

dx_window = 2.5
L_window = 2 * dx_window

# Tolérance 
y_tol_centerline = inletRadius * 0.1 

inject_every_h = cutoff


#######################################################################
# MAIN LOOP
#########################################################

for iteration in range(400):

    current_time = iteration * deltaT

    util.verlet_predictor(positions,velocities,masses,forces,deltaT)

    if(iteration>=0):

        positions    = np.concatenate((positions,    positionsInitial),    axis=1)
        velocities   = np.concatenate((velocities,   velocitiesInitial),   axis=1)
        masses       = np.concatenate((masses,       massesInitial),       axis=0)
        forces       = np.concatenate((forces,       forcesInitial),       axis=1)
        pressures    = np.concatenate((pressures,    pressuresInitial),    axis=0)
        densities    = np.concatenate((densities,    densitiesInitial),    axis=0)
        temperatures = np.concatenate((temperatures, temperaturesInitial), axis=0)

    inBoundNode = positions[0, :] < (exitXaxisPosition)
    positions  = positions[:, inBoundNode]
    velocities = velocities[:, inBoundNode]
    masses     = masses[inBoundNode]
    forces     = forces[:, inBoundNode]
    pressures  = pressures[inBoundNode]
    densities  = densities[inBoundNode]
    temperatures = temperatures[inBoundNode]

    N = len(temperatures)

    positions, velocities = geo.reflectParticlesOnWall(positions, velocities,inletXaxisPosition,throatXaxisPosition,exitXaxisPosition,inletRadius,throatRadius,exitRadius,profileType)

    forces = np.zeros_like(forces)

    h = cutoff

    i, j, d, r = SPH.get_neighbours(positions, cutoff=h)
    print(f"{len(i)} paires de voisins trouvées")

    densities = SPH.compute_density(N, masses, i, j, d, h)
    pressures = SPH.pression(densities,pressureEndlimit,density)*1e-7
    temperatures = SPH.isentropique(temp,densities,density)
    forces += SPH.compute_pressure_forces(N, masses, densities, pressures, i, j, d, r, h)
    forces += viscosity*SPH.compute_viscosity_forces(N, masses, densities, velocities, i, j, d, r, h)

    if not np.all(np.isfinite(positions)):
        bad = np.where(~np.isfinite(positions).all(axis=0))[0]
        raise RuntimeError(f"Non-finite positions at particles {bad}")
    if not np.all(np.isfinite(velocities)):
        raise RuntimeError("Non-finite velocities")

    util.verlet_corrector(positions,velocities,masses,forces,deltaT)

    time_history.append(current_time)

    # Calcul du Mach
    mach = SPH.calcul_mach(velocities, temperatures)

    # Masque pour la ligne centrale
    mask_centerline = np.abs(positions[1, :]) < y_tol_centerline

    #ENTREE
    mask_inlet = mask_centerline & \
                 (positions[0, :] >= inletXaxisPosition) & \
                 (positions[0, :] < (inletXaxisPosition + dx_window))
    if np.any(mask_inlet):
        mach_inlet_history.append(np.mean(mach[mask_inlet]))
        p_inlet_history.append(np.mean(pressures[mask_inlet]))
    else:
        mach_inlet_history.append(0.0)
        p_inlet_history.append(0.0)

    mask_throat = mask_centerline & \
                  (positions[0, :] > (throatXaxisPosition - dx_window)) & \
                  (positions[0, :] < (throatXaxisPosition + dx_window))
    if np.any(mask_throat):
        mach_throat_history.append(np.mean(mach[mask_throat]))
        p_throat_history.append(np.mean(pressures[mask_throat]))
    else:
        mach_throat_history.append(0.0)
        p_throat_history.append(0.0)

    # SORTIE
    mask_exit = mask_centerline & \
                (positions[0, :] > (exitXaxisPosition - dx_window)) & \
                (positions[0, :] <= exitXaxisPosition)
    if np.any(mask_exit):
        mach_exit_history.append(np.mean(mach[mask_exit]))
        p_exit_history.append(np.mean(pressures[mask_exit]))
    else:
        mach_exit_history.append(0.0)
        p_exit_history.append(0.0)

    print(f"Step {iteration} completed with {positions.shape[1]} particles.")

    util.write_xyz(f"C:\\Users\\wawak\\Documents\\L3_mecanique\\S6\\LU3ME104\\ProjetFinal\\Programme\\vis\\traj_{iteration:04d}.xyz",positions)

#######################################################################
# Color map type NASA
#########################################################

_mach_colors = [
    (0.00, (0.00, 0.00, 0.00)),  # noir
    (0.10, (0.00, 0.00, 0.50)),  # bleu foncé
    (0.20, (0.00, 0.00, 1.00)),  # bleu
    (0.30, (0.00, 1.00, 1.00)),  # cyan
    (0.45, (0.00, 1.00, 0.00)),  # vert
    (0.55, (1.00, 1.00, 0.00)),  # jaune
    (0.65, (1.00, 0.50, 0.00)),  # orange
    (0.75, (1.00, 0.00, 0.00)),  # rouge
    (0.88, (1.00, 0.00, 1.00)),  # magenta
    (1.00, (1.00, 1.00, 1.00)),  # blanc
]
mach_cmap = LinearSegmentedColormap.from_list("mach_number", _mach_colors, N=512)

#######################################################################
# PLOT 
#########################################################

# PLOT 1 : Évolution temporelle du Mach aux 3 sections
fig1 = plt.figure("Mach vs temps", figsize=(10, 6))
ax1 = fig1.add_subplot(111)
ax1.plot(time_history, mach_inlet_history,  label='Entrée (x=0)',     color='green')
ax1.plot(time_history, mach_throat_history, label='Col (x=10 cm)',    color='blue')
ax1.plot(time_history, mach_exit_history,   label='Sortie (x=20 cm)', color='red')
ax1.set_xlabel('Temps (s)')
ax1.set_ylabel('Nombre de Mach (ligne centrale)')
ax1.set_title('Évolution temporelle du nombre de Mach aux 3 sections')
ax1.legend()
ax1.grid(True, linestyle='--')
fig1.tight_layout()

# PLOT 2 : Évolution temporelle de la pression aux 3 sections

fig2 = plt.figure("Pression vs temps", figsize=(10, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(time_history, p_inlet_history,  label='Pression Entrée', color='green')
ax2.plot(time_history, p_throat_history, label='Pression Col',    color='blue')
ax2.plot(time_history, p_exit_history,   label='Pression Sortie', color='red')
ax2.set_xlabel('Temps (s)')
ax2.set_ylabel('Pression statique (Pa)')
ax2.set_title('Évolution temporelle de la pression statique aux 3 sections')
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.legend()
ax2.grid(True, linestyle='--')
fig2.tight_layout()


# 
# Préparation des données pour les profils spatiaux finaux
# 
mask_cl_final = np.abs(positions[1, :]) < y_tol_centerline

x_cl  = positions[0, mask_cl_final]
vx_cl = velocities[0, mask_cl_final]
vy_cl = velocities[1, mask_cl_final]
T_cl  = temperatures[mask_cl_final]
p_cl_all = pressures[mask_cl_final]

n_bins = 25
x_edges = np.linspace(inletXaxisPosition, exitXaxisPosition, n_bins + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

mach_profile = np.full(n_bins, np.nan)
p_profile    = np.full(n_bins, np.nan)

gamma = 1.4
r_gas = 287.0  # J/(kg.K)

for k in range(n_bins):
    in_bin = (x_cl >= x_edges[k]) & (x_cl < x_edges[k+1])
    if np.any(in_bin):
        vx_mean = np.mean(vx_cl[in_bin])
        vy_mean = np.mean(vy_cl[in_bin])
        T_mean  = np.mean(T_cl[in_bin])
        v_mean  = np.sqrt(vx_mean**2 + vy_mean**2)
        c_mean  = np.sqrt(gamma * r_gas * T_mean)
        mach_profile[k] = v_mean / c_mean
        p_profile[k]    = np.mean(p_cl_all[in_bin])

valid = ~np.isnan(mach_profile)


# 
# PLOT 3 : Profil spatial du Mach (ligne centrale, état final)
# 
fig3 = plt.figure("Profil Mach axial", figsize=(10, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(x_centers[valid], mach_profile[valid], 'o-', color='blue', markersize=5)
ax3.axvline(throatXaxisPosition, color='k', linestyle=':', label='Col')
ax3.set_xlabel('Position axiale x (m)')
ax3.set_ylabel('Nombre de Mach (moyenné par bande)')
ax3.set_title('Profil du nombre de Mach sur la ligne centrale (état final)')
if np.any(valid):
    ax3.set_ylim(np.nanmin(mach_profile) - 0.05, np.nanmax(mach_profile) + 0.1)
ax3.legend()
ax3.grid(True, linestyle='--')
fig3.tight_layout()

# PLOT 4 : Profil spatial de la pression (ligne centrale, état final)

fig4 = plt.figure("Profil pression axial", figsize=(10, 6))
ax4 = fig4.add_subplot(111)
ax4.plot(x_centers[valid], p_profile[valid], 'o-', color='red', markersize=5)
ax4.axvline(throatXaxisPosition, color='k', linestyle=':', label='Col')
ax4.set_xlabel('Position axiale x (m)')
ax4.set_ylabel('Pression statique moyenne (Pa)')
ax4.set_title('Profil de pression statique sur la ligne centrale (état final)')
if np.any(valid):
    ax4.set_ylim(np.nanmin(p_profile) * 0.9, np.nanmax(p_profile) * 1.1)
ax4.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.legend()
ax4.grid(True, linestyle='--')
fig4.tight_layout()

# PLOT 5 : Carte 2D du nombre de Mach dans la tuyère

# On calcule le Mach pour toutes les particules (pas seulement la ligne centrale)
mach_all = SPH.calcul_mach(velocities, temperatures)

# Filtrage : on ne garde que les particules à l'intérieur du domaine [inlet, exit]
mask_domain = (positions[0, :] >= inletXaxisPosition) & \
              (positions[0, :] <= exitXaxisPosition)

x_all = positions[0, mask_domain]
y_all = positions[1, mask_domain]
m_all = mach_all[mask_domain]

fig5 = plt.figure("Carte Mach 2D", figsize=(12, 6))
ax5 = fig5.add_subplot(111)

# Scatter coloré par le Mach
sc = ax5.scatter(x_all, y_all, c=m_all, cmap=mach_cmap, s=8,
                 vmin=np.nanmin(m_all), vmax=np.nanmax(m_all))

# Tracé du contour de la tuyère (parois supérieure et inférieure)
x_wall = np.linspace(inletXaxisPosition, exitXaxisPosition, 400)
r_wall = geo.nozzle_radius_profile(x_wall,
                               inletXaxisPosition, throatXaxisPosition, exitXaxisPosition,
                               inletRadius, throatRadius, exitRadius,
                               profile=profileType)
ax5.plot(x_wall,  r_wall, 'k-', linewidth=2)
ax5.plot(x_wall, -r_wall, 'k-', linewidth=2)

# Ligne du col
ax5.axvline(throatXaxisPosition, color='k', linestyle=':', linewidth=1, alpha=0.7,
            label='Col')

# Colorbar
cbar = fig5.colorbar(sc, ax=ax5, orientation='vertical', pad=0.02)
cbar.set_label('Nombre de Mach')

ax5.set_xlabel('Position axiale x (m)')
ax5.set_ylabel('Position radiale y (m)')
ax5.set_title('Carte 2D du nombre de Mach dans la tuyère (état final)')
ax5.set_aspect('equal', adjustable='box')
ax5.legend(loc='upper right')
ax5.grid(True, linestyle='--', alpha=0.3)
fig5.tight_layout()

# PLOT 6 : Carte 2D du Mach moyenné par tranches axiales

n_bins_2d = 40
x_edges_2d = np.linspace(inletXaxisPosition, exitXaxisPosition, n_bins_2d + 1)
x_centers_2d = 0.5 * (x_edges_2d[:-1] + x_edges_2d[1:])

mach_mean_per_bin = np.full(n_bins_2d, np.nan)

for k in range(n_bins_2d):
    in_bin = (x_all >= x_edges_2d[k]) & (x_all < x_edges_2d[k + 1])
    if np.any(in_bin):
        mach_mean_per_bin[k] = np.mean(m_all[in_bin])

fig6 = plt.figure("Carte Mach 2D moyennée", figsize=(12, 6))
ax6 = fig6.add_subplot(111)

# Bornes de la colormap
vmin_avg = np.nanmin(mach_mean_per_bin)
vmax_avg = np.nanmax(mach_mean_per_bin)
cmap = mach_cmap
norm = plt.Normalize(vmin=0, vmax=2)

for k in range(n_bins_2d):
    if np.isnan(mach_mean_per_bin[k]):
        continue
    x_left  = x_edges_2d[k]
    x_right = x_edges_2d[k + 1]
    r_left  = geo.nozzle_radius_profile(np.array([x_left]),
                                    inletXaxisPosition, throatXaxisPosition, exitXaxisPosition,
                                    inletRadius, throatRadius, exitRadius,
                                    profile=profileType)[0]
    r_right = geo.nozzle_radius_profile(np.array([x_right]),
                                    inletXaxisPosition, throatXaxisPosition, exitXaxisPosition,
                                    inletRadius, throatRadius, exitRadius,
                                    profile=profileType)[0]
    color = cmap(norm(mach_mean_per_bin[k]))
    poly = Polygon([[x_left,  -r_left],
                    [x_right, -r_right],
                    [x_right,  r_right],
                    [x_left,   r_left]],
                   closed=True, facecolor=color, edgecolor='none')
    ax6.add_patch(poly)

# Tracé du contour de la tuyère par-dessus
ax6.plot(x_wall,  r_wall, 'k-', linewidth=2)
ax6.plot(x_wall, -r_wall, 'k-', linewidth=2)


ax6.axvline(throatXaxisPosition, color='k', linestyle=':', linewidth=1, alpha=0.7,
            label='Col')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar6 = fig6.colorbar(sm, ax=ax6, orientation='vertical', pad=0.02)
cbar6.set_label('Nombre de Mach (moyenné par tranche)')

ax6.set_xlim(inletXaxisPosition, exitXaxisPosition)
ax6.set_ylim(-inletRadius * 1.1, inletRadius * 1.1)
ax6.set_xlabel('Position axiale x (m)')
ax6.set_ylabel('Position radiale y (m)')
ax6.set_title('Carte 2D du nombre de Mach moyenné par tranche axiale (état final)')
ax6.set_aspect('equal', adjustable='box')
ax6.legend(loc='upper right')
ax6.grid(True, linestyle='--', alpha=0.3)
fig6.tight_layout()

plt.show()