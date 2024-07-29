
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
# import DFT_loader
import os

design_dir = "./A/"
mp.verbosity(1)
# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

SiO2= mp.Medium(index=1.45)
My_LiNbO3= mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)) # X cut
#My_LiNbO3= DFT_loader.My_LiNbO3_pso()

Cone_rad = 0.3 # 0.1 0.15 0.2 0.25 0.3
Free= False
Cone= True

Sy = 5
Prop_length= 5 

#############################
Geometry_profile= True      #
Ex_only= False              #
#############################
#############################
Main_Parameters= True       #
Parameter_activation= True  #
Monitor_Profile= True       #
Source_profile= True        #########################################################################
#                        ####  Set main parameters  ####                                            #
if Main_Parameters:                                                                                 #-Parameters for Simulation--%
    resolution= 50                                                                                  #-Resolution-----|
    Lpml= 10.0/ resolution                                                                          #-Tickness of PML|
                                                                                                    #
    # Hights (x: thickness)                                                                         #-Hights (X)-----|
    SiO2_h= 0.7 # SiO2 padding                                                                      #                |
    LNsub_h= 0.1 # LN substrate                                                                     #                |
    LNwg_h= 0.5 # LN waveguide                                                                      #----------------|
                                                                                                    #
    # Width (y: width)                                                                              #-Waveguide (Y)--|
    min_g= 0.44 # minimum gap                                                                       #                |
                                                                                                    #                |
    input_w_top= 0.6  # top width (input waveguide)                                                 #                |
    input_w_bot= input_w_top + min_g                                                                #                |
                                                                                                    #                |
    w_top= 0.8 # top width (output waveguide)<---------------------------                           #                |
    w_bot= w_top+ min_g #bottom width (1222 nm)                                                     #----------------|
                                                                                                    #
    # Waveguide length (z: length)                                                                  #
    waveguide_length= 0.5 # 입출력단 길이                                                            #-Propagation (Z)|
    pml_2_src= 0.3                                                                                  #                |
    mon_2_pml= 1/resolution                                                                                  #                |
                                                                               #----------------|--> 3 to 4
                                                                                                    #
    fcen= 1.0/ 0.775 # 775 nm                                                                       #-Frequancy------|
    fwidth = 0.2 * fcen                                                                             #----------------|
#                     ####  Set main components and monitors  ####                                  #
if Parameter_activation:                                                                            #--Variables for simulation--%                          
    Sx = SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h                                                            #-XYZ length--|
    # Width: w/o pml, 3um                                                                   #             |--> 3 to 4
    Sz = waveguide_length+ Prop_length+ waveguide_length # 길이 (진행방향)                           #-------------|
                                                                                                    #
    # Overall volume                                                                                #
    cell= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, Sz+ 2*Lpml)                                            #-Total Volume|
    pml_layers = [mp.PML(thickness=Lpml)]#, direction=mp.Y)]                                        #-PML---------|
                                                                                                    #
    # 공간 비례 변수 (pml 제외 유효 공간)                                                             #     
    X_min= 0.5*-Sx                                                                                  #-Min points--|
    Y_min= 0.5*-Sy                                                                                  #             |
    Z_min= 0.5*-Sz                                                                                  #-------------|
                                                                                                    #
    X_max= 0.5*Sx                                                                                   #-Max points--|
    Y_max= 0.5*Sy                                                                                   #             |
    Z_max= 0.5*Sz                                                                                   #-------------|
                                                                                                    #
    wg_hc= X_min+ SiO2_h+ (LNsub_h+ LNwg_h)/2 # middle of LNOI                                      #-LNOI center-|
                                                                                                    #
    # design region                                                                                 #                      
    design_region_x = LNwg_h                                                                        #-Design size-|
    design_region_y = Sy                                                                            #             |
    design_region_z = Sz - (waveguide_length*2)                                                     #             |
    design_region_resolution = int(resolution)                                                      #-------------|
                                                                                                    #
    Nx = int(design_region_resolution * design_region_x)+ 1                                         #-Grid points-|
    Ny = int(design_region_resolution * design_region_y)+ 1                                         #             |
    Nz = int(design_region_resolution * design_region_z)+ 1                                         #-------------|
if Monitor_Profile:                                                                                 #---Center & Size of Monitor-%
    # Adjoint source profile<-------------------------------시뮬레이션 조건 바뀌면 체크               #
    Adjoint_center= mp.Vector3(wg_hc, 0, Z_max- mon_2_pml)                                          #-Adjoint-----|
    Adjoint_size= mp.Vector3(Sx, 3.0, 0)                                                            #-------------|
                                                                                                    #
    # Fundamental source (TE00)                                                                     #
    source_center= mp.Vector3(wg_hc, 0, Z_min+ pml_2_src)                                           #-Source------|
    source_size= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, 0)                                              #-------------|
                                                                                                    #
    dft_monitor_center= mp.Vector3(0, 0, Z_max- mon_2_pml)                                          #-Output------|
    dft_monitor_size= mp.Vector3(Sx, Sy, 0)                                                         #-------------|
                                                                                                    #
    Incidence_center= mp.Vector3(wg_hc, 0, Z_min+ waveguide_length)                                 #-Input-------|
    Incidence_size= mp.Vector3(LNsub_h + LNwg_h + SiO2_h, Sy/2, 0)                                  #-------------|
    Eff_center= Adjoint_center                                                                      #-Transmission|
    Eff_size= mp.Vector3(LNsub_h + LNwg_h + SiO2_h, Sy/2, 0)                                        #-------------|
if Source_profile:                                                                                  #----Sources for simulation--%
    kpoint = mp.Vector3(0,0,1)                                                                      #-K vector----|
    src = mp.GaussianSource(frequency= fcen, fwidth= fwidth)#, is_integrated=True)                  #-Unit source-|
                                                                                                    #
    source1 = [                                                                                     #-TE00 (775)--|
        mp.EigenModeSource(                                                                         #             |
            src,                                                                                    #             |
            size= source_size,                                                                      #             |
            center= source_center,                                                                  #             |
            direction= mp.NO_DIRECTION,                                                             #             |
            eig_kpoint= kpoint,                                                                     #             |
            eig_band= 2,                                                                            #             |
            eig_match_freq= True,                                                                   #             |
        ),                                                                                          #             |
    ]                                                                                               #-------------|
    source2 = [                                                                                     #-TE20 (775)--|
        mp.EigenModeSource(                                                                         #             |
            src,                                                                                    #             |
            size= source_size,                                                                      #             |
            center= source_center,                                                                  #             |
            direction= mp.NO_DIRECTION,                                                             #             |
            eig_kpoint= kpoint,                                                                     #             |
            eig_band= 8,                                                                            #             |
            eig_parity=mp.NO_PARITY,                                                                #             |
            eig_match_freq= True,                                                                   #             |
        ),                                                                                          #             |
    ]                                                                                               #-------------|
    source3 = [                                                                                     #-Reverse src-|
        mp.EigenModeSource(                                                                         #             |
            src,                                                                                    #             |
            size= source_size,                                                                      #             |
            center= dft_monitor_center,                                                             #             |
            direction= mp.NO_DIRECTION,                                                             #             |
            eig_kpoint= mp.Vector3(0,0,-1),                                                         #             |
            eig_band= 8,                                                                            #             |
            eig_parity=mp.NO_PARITY,                                                                #             |
            eig_match_freq= True,                                                                   #             |
        ),                                                                                          #             |
    ]                                                                                               #-------------|
########################    Should be equal to that of Optimization    ############################## 

if Geometry_profile:
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
        mp.Block(#LiNbO3 substrate
            center= mp.Vector3(X_min+ SiO2_h+ 0.5*LNsub_h, 0, 0),
            material= My_LiNbO3,
            size= mp.Vector3(LNsub_h, Sy+ 2*Lpml, Sz+ 2*Lpml) 
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

""" Adjoint optimization """
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
    FoM_x = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ex ) #<------------------FoM calc
    FoM_y = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ey )
    FoM_z = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ez )
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
    minimum_run_time=10,
    decay_by=1e-6,
)

import Sub_Mapping
eta_Ref= 0.5



def mapping(x, beta):
    if Free:
        x= Sub_Mapping.TwoD_freeform(x, beta, Nx, Ny, Nz, eta_Ref)
    elif Cone:
        x= Sub_Mapping.TwoD_Filtered(x, beta, Nx, design_region_y, design_region_z, design_region_resolution, 0.25, Cone_rad)
    return x



evaluation_history = []
cur_iter = [0]
numevl = 1

def f(v, gradient, beta):
    global numevl
    print("Current iteration: {}".format(cur_iter[0] + 1))
    f0, dJ_du = opt([mapping(v, beta)])  # compute objective and gradient
    #np.savetxt(design_dir+"Test"+str(numevl) +".txt", dJ_du) <---- adjoint gradient
    print(dJ_du.size)

    if beta == npa.inf:
        evaluation_history.append(np.real(f0))
        numevl += 1
        print("First FoM: {}".format(evaluation_history[0]))
        print("Last FoM: {}".format(np.real(f0)))
        np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
        print("-----------------Optimization Complete------------------")
        return np.real(f0)
    else:
        if gradient.size > 0:
            gradient[:] = tensor_jacobian_product(mapping, 0)(
                v, beta, dJ_du,
            )  # backprop
            print(gradient.size)
            #np.savetxt(design_dir+"grad"+str(numevl) +".txt", gradient) <---- gradient for nlopt
        evaluation_history.append(np.real(f0))
        #np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights) <---- geometry
        numevl += 1
        print("First FoM: {}".format(evaluation_history[0]))
        print("Current FoM: {}".format(np.real(f0)))
        cur_iter[0] = cur_iter[0] + 1
        return np.real(f0)

algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
n = Ny * Nz  # number of parameters

# Initial guess - 초기 시작값 0.5
#x = np.random.uniform(0.45, 0.55, n)
x = np.ones((n,)) * 0.5

#np.savetxt(design_dir+"lastdesign.txt", x)
#lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Ny * Nz))
ub = np.ones((Ny * Nz))

beta_init = 2
cur_beta = beta_init
beta_scale = 2
num_betas = 5
update_factor = 20

for iters in range(num_betas + 1):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds

    if cur_beta >= (beta_init * (beta_scale ** (num_betas))):
        cur_beta = npa.inf
        print("current beta: ", cur_beta)
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(1) # Set the maximum number of function evaluations
    else:
        print("current beta: ", cur_beta)
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
        cur_beta = cur_beta * beta_scale
    # solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)

    
plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"result.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

np.savetxt(design_dir+"evaluation.txt", evaluation_history)
np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)

