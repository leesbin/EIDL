import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from matplotlib import pyplot as plt

import wandb
wandb.init(project = "gradient_test", settings=wandb.Settings(silent=True))

mp.verbosity(0)

Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1.0)

resolution = 40
pixel=1/resolution

design_region_width = 4
design_region_height = 1
pml_size = 1.0

Sx = 2 * pml_size + design_region_width
Sy = 2 * pml_size + design_region_height + 2 + 6*pixel
cell_size = mp.Vector3(Sx, Sy)

wavelengths = 2
frequencies = np.array([1/wavelengths])

nf = len(frequencies) # number of frequencies

design_region_resolution = int(resolution)

pml_layers = [mp.PML(pml_size)]

fcen = 1 / 2
width = 0.2
fwidth = width * fcen
source_center = [0, -(design_region_height/2+1), 0]
source_size = mp.Vector3(Sx, 0, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

Nx = 21
Ny = 6

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)

geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    resolution=resolution,
)

monitor_position, monitor_size = mp.Vector3(0,design_region_height/2+1), mp.Vector3(0.01,0)
FourierFields = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ez,yee_grid=True)
ob_list = [FourierFields]

def J(fields):
    return npa.mean(npa.abs(fields[:,1]) ** 2) # The index 1 corresponds to the point at the center of our monitor.

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-2
)
opt.plot2D(True)



evaluation_history = []
n = Nx * Ny  # number of parameters
x = np.ones((n,)) * 0.5

num_betas = 5
LR=0.01

for iters in range(num_betas):
    print(iters)
    f0, dJ_du = opt([x])#[mapping(x, eta_i, cur_beta)])
    adjoint_gradient = dJ_du
    adjoint_gradient=np.array(adjoint_gradient).flatten()
    reshaped_gradients = adjoint_gradient.reshape(adjoint_gradient.shape[0], -1) #열벡터로 변환

    evaluation_history.append(np.real(f0))
    #sensitivity[0]=dJ_du

    # Computing the norm of each reshaped row
    norms = np.linalg.norm(reshaped_gradients, axis=1)

    # Calculating the mean of the norms
    adjgrad_norm = norms.mean()
    print(adjoint_gradient)
    #print(type(adjoint_gradient))
    
    x_image = x.reshape(Nx, Ny, 1)
    learning_rate = LR / adjgrad_norm

    wandb.log({
                    'fom ': f0,
                    'adjgrad norm': adjgrad_norm,
                    'learning rate': learning_rate,
                    "generated": [wandb.Image(x_image, caption=f"Iteration {iters}")]
    
                }),
    
    x = x + learning_rate * adjoint_gradient
    x[x>1] = 1
    x[x<0] = 0



np.savetxt("/root/design_result/gd/lastdesign.txt", x)
np.savetxt("/root/design_result/gd/dJ_du.txt",dJ_du)
np.savetxt("/root/design_result/gd/adjoint_gradient.txt",adjoint_gradient)
