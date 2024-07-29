"""Validates the adjoint gradient of a metalens in 3d."""

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt

Air = mp.Medium(index=1.0)
SiN = mp.Medium(index=1.5)


resolution = 20 
Lpml = 0.5 
pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]
Sourcespace = 0.5

design_region_width_x = 0.5
design_region_width_y = 0.5 
design_region_height = 0.5 

Sx = design_region_width_x
Sy = design_region_width_y
Sz = Lpml + design_region_height + Sourcespace + 1 + Lpml
cell_size = mp.Vector3(Sx, Sy, Sz)

wavelengths = np.array([0.5])
frequencies = 1/wavelengths
nf = len(frequencies) 

design_region_resolution = int(resolution)

fcen = 1 / 0.5
width = 0.1
fwidth = width * fcen

source_center = [0, 0, Sz / 2 - Lpml - Sourcespace / 2 ] 
source_size = mp.Vector3(Sx, Sy, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center,),]

Nx = int(round(design_region_resolution * design_region_width_x)) 
Ny = int(round(design_region_resolution * design_region_width_y)) 
Nz = int(round(design_region_resolution * design_region_height))

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, SiN, grid_type="U_MEAN",do_averaging=False)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height/2),
        size=mp.Vector3(design_region_width_x, design_region_width_y, design_region_height),
    ),
)

geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
]

sim = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, 
    resolution=resolution,
    k_point = mp.Vector3(0,0,0)
)

monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width_x/2, design_region_width_y/2, -Sz/2 + Lpml + 0.5/resolution), mp.Vector3(0.01,0.01,0) 

FourierFields_0_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ex,yee_grid=True)


ob_list = [FourierFields_0_x]


def J(fields):
    return npa.mean(npa.abs(fields[:,1]) ** 2)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-3,
)

# Ensure reproducible results.                                                                                
rng = np.random.RandomState(9861548)

# Random design region.                                                                                       
# initial_design_region = 0.9 * rng.rand(NX * NY * Nz)                                                             

# Constant design region.                                                                                     
initial_design_region = 0.9 * np.ones((Nx * Ny * Nz))

# Random perturbation for design region.                                                                      
max_perturbation = 1e-5
random_perturbation = (max_perturbation *
                        rng.rand(Nx * Ny * Nz))

unperturbed_val, unperturbed_grad = opt(
    [initial_design_region],
    need_gradient=True
)

perturbed_val, _ = opt(
    [initial_design_region + random_perturbation],
    need_gradient=False
)

adjoint_directional_deriv = ((random_perturbation[None, :] @
                                unperturbed_grad).flatten())
finite_diff_directional_deriv = perturbed_val - unperturbed_val

print(f"directional-derivative:, {finite_diff_directional_deriv} "
        f"(finite difference), {adjoint_directional_deriv} (adjoint)")