import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c 

sys.path.append('../../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from wakeSolver import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 55
Ny = 55
Nz = 108
#dt = 5.707829241e-12 # CST

# Embedded boundaries
stl_cavity = 'cavity.stl' 
stl_pipe = 'beampipe.stl'

# Materials
stl_solids = {'cavity': stl_cavity, 'pipe': stl_pipe}
stl_materials = {'cavity': 'vacuum', 'pipe':  'vacuum'}
background = [100, 1.0, 100] # lossy metal [ε_r, µ_r, σ]

# Domain bounds
surf = pv.read(stl_cavity) + pv.read(stl_pipe)
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# Set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)
#grid.inspect()

# ------------ Beam source ----------------
# Beam parameters
beta = 0.5          # beam relativistic beta 
sigmaz = beta*6e-2  # [m] -> multiplied by beta to have f_max cte
q = 1e-9            # [C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/(beta*c)  # injection time offset [s] 

# Simulation
wakelength = 21. #[m]
add_space = 5   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=True, results_folder='results_beta05_sigma100_wl21/')

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

solver = SolverFIT3D(grid, wake, #dt=2.3277666273886625e-12*2,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg=background)
# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':None, 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=False,
                    **plotkw)

# Run only electromagnetic time-domain simulation
runEM = False
if runEM:
    from sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)

    solver.emsolve(Nt=500, source=beam, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=True, **plotkw)
    
#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
# CST wake
#cstWP = wake.read_txt('cst/WP.txt')
#cstZ = wake.read_txt('cst/Z.txt')
wake.f = np.abs(wake.f)

fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5) # , label='FIT+Wakis'
#ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
ax[0].set_xlabel('s [cm]')
ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
ax[0].legend()

ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5) # , label='FIT+Wakis'
#ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
ax[1].set_xlabel('f [GHz]')
ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
ax[1].legend()

#fig.suptitle('Benchmark with CST Wakefield Solver')
fig.tight_layout()
fig.savefig('results_beta05_sigma100_wl21_longerpipe/wake_and_impedance.png')

plt.show()
