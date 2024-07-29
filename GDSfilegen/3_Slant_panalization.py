# Find TE20 mode and LN waveguide validation 
# # LN Straight Waveguide Simulation

# ## 1. Simulation Environment

import scipy
import math
import numpy as np
import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import nlopt
import DFT_loader
import os

#mp.verbosity(1)

design_dir = "./3_Slant_penalization_result/"

# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

seed = 240
np.random.seed(seed)

SiO2= mp.Medium(index=1.45)
My_LiNbO3= mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)) # X cut

Sx= 1.0# 높이: 위아래 pml 0부터 LN, 그아래는 SiO2
Sy= 3.0# 너비: 좌우 pml, 폭 3um                         
Sz= 3.0# 길이 (진행방향): 3 

resolution = 50
Lpml= 10.0/ resolution

# 공간 비례 변수 (pml 뺀 유효 공간)
X_min= 0.5*-Sx
Y_min= 0.5*-Sy
Z_min= 0.5*-Sz

X_max= 0.5*Sx
Y_max= 0.5*Sy
Z_max= 0.5*Sz

# 높이 (x: thickness)
LNsub_h= 0.1 # LN substrate
SiO2_h= 0.2 # SiO2 substrate
LNwg_h= 0.5 # LN waveguide

# 폭 (y: width)
min_g= 0.424 # minimum gap
w_top= 0.798 # top width (output waveguide)
w_bot= w_top+ min_g #bottom width (1222 nm) 

input_w_top= 0.6
input_w_bot= input_w_top + min_g

# waveguide 길이
waveguide_length= 0.5 # 입출력단 길이 

# Adjoint source profile<-------------------------------시뮬레이션 조건 바뀌면 항시 체크
Adjoint_center= mp.Vector3(X_min+ SiO2_h+ (LNsub_h+ LNwg_h)/2, 0, Z_max- 0.3)
Adjoint_size= mp.Vector3(LNsub_h + LNwg_h + 0.2, 1.4, 0)
Ex_only= False
capture_setup= 1
flux_normalization= 1
is_final_check= True

# 설계 공간
design_region_x = LNwg_h
design_region_y = Sy
design_region_z = 2
design_region_resolution = int(resolution)

# 전체공간
cell= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, Sz+ 2*Lpml)
pml_layers = [mp.PML(thickness=Lpml)]#, direction=mp.Y)]

fcen= 1.0/ 0.775 # 775 nm
fwidth = 0.1 * fcen

# Fundamental source (TE00)
source_center = mp.Vector3(X_min+ SiO2_h+ (LNsub_h+ LNwg_h)/2, 0, Z_min+ 0.3)
source_size = mp.Vector3(LNsub_h+ LNwg_h+ 0.2, 2.0, 0)#

kpoint = mp.Vector3(0,0,1)
src = mp.GaussianSource(frequency= fcen, fwidth= fwidth)

source1 = [
    mp.EigenModeSource(
        src,
        size= source_size,
        center= source_center,
        direction= mp.NO_DIRECTION,
        eig_kpoint= kpoint,
        eig_band= 2,
        eig_match_freq= True,
    ),
]

Nx = int(design_region_resolution * design_region_x) + 1
Ny = int(design_region_resolution * design_region_y) + 1
Nz = int(design_region_resolution * design_region_z) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, My_LiNbO3, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(X_min+ SiO2_h+ LNsub_h + LNwg_h/2, 0, 0),
        size=mp.Vector3(design_region_x, design_region_y, design_region_z),
    ),
)

# 구조물 Z축을 따라 pml 영역까지 배치
geometry= [
    # mp.Block(#SiO2 substrate
    #     center= mp.Vector3(X_min+ 0.5*SiO2_h, 0, 0),
    #     material= SiO2,
    #     size= mp.Vector3(SiO2_h, Sy, Sz)
    # ),
    mp.Block(#LiNbO3 substrate
        center= mp.Vector3(X_min+ SiO2_h+ 0.5*LNsub_h, 0, 0),
        material= My_LiNbO3,
        size= mp.Vector3(LNsub_h, Sy, Sz+ 2*Lpml) 
    ),    
    mp.Prism(#Input LiNbO3 waveguide
        vertices = [
            mp.Vector3(X_min+ SiO2_h+ LNsub_h, input_w_bot/2, Z_min- Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h, -input_w_bot/2, Z_min- Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, -input_w_top/2, Z_min- Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, input_w_top/2, Z_min- Lpml)
        ],
        height= waveguide_length+ Lpml,
        axis= mp.Vector3(0, 0, 1),
        sidewall_angle= 0,
        material= My_LiNbO3,
    ),
    mp.Prism(#Output LiNbO3 waveguide
        vertices = [
            mp.Vector3(X_min+ SiO2_h+ LNsub_h, w_bot/2, Z_max+ Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h, -w_bot/2, Z_max+ Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, -w_top/2, Z_max+ Lpml),
            mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, w_top/2, Z_max+ Lpml)
        ],
        height= waveguide_length+ Lpml,
        axis= mp.Vector3(0, 0, -1),
        sidewall_angle= 0,
        material= My_LiNbO3,
    ),
    mp.Block(#Design region
        center=design_region.center, size=design_region.size, material=design_variables
    ),  
]
####################################################################################################
sim = mp.Simulation(
    cell_size= cell,
    resolution= resolution,
    boundary_layers= pml_layers,
    sources= source1,
    eps_averaging= False,
    geometry= geometry,
    default_material=SiO2,
    extra_materials=[My_LiNbO3],
    symmetries=[mp.Mirror(mp.Y, phase=-1)],
)


""" Optimization """
# Get TE20 profile [Ex, Ey, Ez]
if Ex_only: # Consider Main components of TE20 field
    Target_field=DFT_loader.load_target_field()

    # TE20 mode (Ex dft-field < we need to consider the resolution)
    FoM_x = mpa.FourierFields(
        sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ex ) #<------------------FoM calc
    # input of J
    ob_list = [FoM_x]

    # Figure of Merit function
    def J(mode_x):
        A_field=mode_x[0]
        X=npa.abs(npa.sum(A_field*Target_field[0].conjugate()))**2/(npa.sum(npa.abs(Target_field[0])**2)*(npa.sum(npa.abs(A_field)**2)))
        return X

else: # Consider all components of TE20 field
    Target_field=DFT_loader.load_target_field(is_consider_Ex_only=False)

    # TE20 mode (Ex, Ey and Ez dft-field < we need to consider the resolution)
    FoM_x = mpa.FourierFields(
        sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ex ) #<------------------FoM calc
    FoM_y = mpa.FourierFields(
        sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ey )
    FoM_z = mpa.FourierFields(
        sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ez )

    # input of J
    ob_list = [FoM_x, FoM_y, FoM_z]

    # Figure of Merit function
    def J(mode_x, mode_y, mode_z):
        x_field=mode_x[0]
        y_field=mode_y[0]
        z_field=mode_z[0]

        X=npa.abs(npa.sum(x_field*Target_field[0].conjugate()))**2
        Y=npa.abs(npa.sum(y_field*Target_field[1].conjugate()))**2    
        Z=npa.abs(npa.sum(z_field*Target_field[2].conjugate()))**2

        Xp=X/(npa.sum(npa.abs(Target_field[0])**2)*(npa.sum(npa.abs(x_field)**2)))
        Yp=Y/(npa.sum(npa.abs(Target_field[1])**2)*(npa.sum(npa.abs(y_field)**2)))    
        Zp=Z/(npa.sum(npa.abs(Target_field[2])**2)*(npa.sum(npa.abs(z_field)**2)))
        print('Current purity: X=', Xp,', Y=', Yp, ', Z=', Zp)
        Purity = str(Xp) + '\n' + str(Yp) + '\n' + str(Zp)
        with open(design_dir+"last_purity.txt", "w") as text_file:
            text_file.write(Purity)
        return npa.sum(X+Y+Z)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
    maximum_run_time=150,
    decay_by=1e-6,
)

evaluation_history = []
cur_iter = [0]
numevl = 1

def f(v, gradient, beta):
    global numevl
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([mapping(v, eta_i, beta)])  # compute objective and gradient
    # f0, dJ_du = opt()
    
    # Adjoint gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, dJ_du
        )  # backprop

    evaluation_history.append(np.real(f0))

    #np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights)
    
    numevl += 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))

    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)

minimum_length = 0.1 # 구조 최소크기(아일랜드 제거)
eta_e = 0.66 # 
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, 0.55)
eta_i = 0.5 # penalization reference
eta_d = 1 - eta_e 

def eta_z(z,eta_d,eta_e,bottom,top): # slanted
    return eta_d + ((z-bottom)/(top-bottom)) * (eta_e - eta_d)

def mapping(x, eta, beta):
    x_copy = (x.reshape(Nx, Ny*Nz))
    
    number = Nx-1 # 0:bot Nx:top
    x1 = x_copy[int(number)]

    # projection
    z = 0
    x2 = []

    while z < Nx:
       
    # number = int(z)
    # x1 = x_copy[int(number)]
    # Define spatial arrays used to generate bit masks
        y_g = np.linspace(-design_region_y / 2, design_region_y / 2, Ny)
        z_g = np.linspace(-design_region_z / 2, design_region_z / 2, Nz)
        Y_g, Z_g = np.meshgrid(y_g, z_g, sparse=True, indexing="ij")

                # Define the core mask
        down_wg_mask = (Z_g == -design_region_z / 2) & (np.abs(Y_g) <= (((input_w_top-input_w_bot)/Nx)*z + input_w_bot) / 2)#input
        up_wg_mask = (Z_g == design_region_z / 2) & (np.abs(Y_g) <= (((w_top-w_bot)/Nx)*z + w_bot) / 2)#output

        Li_mask = down_wg_mask | up_wg_mask

                # Define the cladding mask
        border_mask = (
            (Y_g == design_region_y / 2)
            | (Y_g == -design_region_y / 2)
            | (Z_g == -design_region_z / 2)
            | (Z_g == design_region_z / 2)
        )
        Air_mask = border_mask.copy()
        Air_mask[Li_mask] = False
        
        x1 = npa.where(Li_mask.flatten(), 1, npa.where(Air_mask.flatten(), 0, x1))
        
        # filter
        filtered_field = mpa.conic_filter(
            x1,
            filter_radius,
            design_region_y,
            design_region_z,
            design_region_resolution,
        )
        
        z_slice = mpa.tanh_projection(filtered_field, beta, eta_z(z, eta_d, eta_i, 0, Nx))
        
        z_slice = (npa.flipud(z_slice) + z_slice) / 2# <------------------2로 나눈이유?
        
        z_slice = z_slice.flatten()
        
        x2 = npa.concatenate((x2,z_slice),axis=0) 
        z = z + 1

    x = x2.flatten()

    # interpolate to actual materials
    return x

algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny * Nz  # number of parameters

# Initial guess - 초기 시작값 0.5
#x = np.random.uniform(0.45, 0.55, n)
x = np.ones((n,)) * 0.5
# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny * Nz))
ub = np.ones((Nx * Ny * Nz))

cur_beta = 2
beta_scale = 2
num_betas = 6
update_factor = 30
#ftol = 1e-3 

for iters in range(num_betas):
    print("current beta: ", cur_beta)
    
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    if cur_beta >=64:
        solver.set_max_objective(lambda a, g: f(a, g, float("inf")))
        solver.set_maxeval(1) # Set the maximum number of function evaluations
    else:    
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
   # solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration
    
plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("result.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

np.savetxt(design_dir+"evaluation.txt", evaluation_history)
np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)

# if is_final_check:

    #     #######################################
    #     Input_Ex=sim.add_dft_fields(
    #         [mp.Ex],
    #         fcen,
    #         0,
    #         1,
    #         center=source_center,
    #         size = source_size,
    #     )
    #     Input_Ey=sim.add_dft_fields(
    #         [mp.Ey],
    #         fcen,
    #         0,
    #         1,
    #         center=source_center,
    #         size = source_size,
    #     ) 
    #     Input_Ez=sim.add_dft_fields(
    #         [mp.Ez],
    #         fcen,
    #         0,
    #         1,
    #         center=source_center,
    #         size = source_size,
    #     )
    #     ######################################
    #     Output_Ex=sim.add_dft_fields(
    #         [mp.Ex],
    #         fcen,
    #         0,
    #         1,
    #         center=Adjoint_center,
    #         size = Adjoint_size,
    #     )
    #     Output_Ey=sim.add_dft_fields(
    #         [mp.Ey],
    #         fcen,
    #         0,
    #         1,
    #         center=Adjoint_center,
    #         size = Adjoint_size,
    #     ) 
    #     Output_Ez=sim.add_dft_fields(
    #         [mp.Ez],
    #         fcen,
    #         0,
    #         1,
    #         center=Adjoint_center,
    #         size = Adjoint_size,
    #     )

    #     # run the final simulation
    #     sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0, 150))

    #     Ex_Namei=f"Ex_fund_field"
    #     Ey_Namei=f"Ey_fund_field"
    #     Ez_Namei=f"Ez_fund_field"
    #     sim.output_dft(Input_Ez,Ez_Namei)
    #     sim.output_dft(Input_Ex,Ex_Namei)
    #     sim.output_dft(Input_Ey,Ey_Namei)

    #     Exo_Namei=f"Ex_opt_field"
    #     Eyo_Namei=f"Ey_opt_field"
    #     Ezo_Namei=f"Ez_opt_field"
    #     sim.output_dft(Output_Ex,Exo_Namei)
    #     sim.output_dft(Output_Ey,Eyo_Namei)
    #     sim.output_dft(Output_Ez,Ezo_Namei)
