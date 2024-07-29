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
import Sub_Mapping

# 'Top'
design_dir = "./A/"
mp.verbosity(1)
# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

SiO2= mp.Medium(index=1.45)
My_LiNbO3= mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)) # X cut
#My_LiNbO3= DFT_loader.My_LiNbO3_pso()

#############################
Geometry_profile= True      #
#############################
#############################
Main_Parameters= True       #
Parameter_activation= True  #
Monitor_Profile= True       #
Source_profile= True        #########################################################################
#                        ####  Set main parameters  ####                                            #
if Main_Parameters:                                                                                 #-Parameters for Simulation--%
    resolution= int(50)                                                                             #-Resolution-----|
    Lpml= round(10.0/ resolution, 2)                                                                #-Tickness of PML|
                                                                                                    #
    # Hights (x: thickness)                                                                         #-Hights (X)-----|
    SiO2_h= round(0.7, 2) # SiO2 padding                                                            #                |
    LNsub_h= round(0.1, 2) # LN substrate                                                           #                |
    LNwg_h= round(0.5, 2) # LN waveguide                                                            #----------------|
                                                                                                    #
    # Width (y: width)                                                                              #-Waveguide (Y)--|
    min_g= round(0.44, 2) # minimum gap                                                             #                |
                                                                                                    #                |
    input_w_top= round(0.6, 2)  # top width (input waveguide)                                       #                |
    input_w_bot= round(input_w_top + min_g, 2)                                                      #                |
                                                                                                    #                |
    w_top= round(0.8, 2) # top width (output waveguide)<---------------------------                 #                |
    w_bot= round(w_top+ min_g, 2) #bottom width (1222 nm)                                           #----------------|
                                                                                                    #
    # Waveguide length (z: length)                                                                  #
    waveguide_length= round(0.5, 2) # 입출력단 길이                                                  #-Propagation (Z)|
    pml_2_src= round(0.3, 2)                                                                        #                |
    mon_2_pml= round(0.3, 2)                                                                        #                |
    Prop_length= round(9.6, 2)                                                                      #----------------|--> 3 to 4
                                                                                                    #
    fcen= 1.0/ 0.775 # 775 nm                                                                       #-Frequancy------|
    fwidth = 0.2 * fcen                                                                             #----------------|
#                     ####  Set main components and monitors  ####                                  #
if Parameter_activation:                                                                            #--Variables for simulation--%                          
    Sx = round(SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h, 2)                                                  #-XYZ length--|
    Sy = round(3.6, 2)# Width: w/o pml, 3um                                                         #             |--> 3 to 4
    Sz = round(waveguide_length+ Prop_length+ waveguide_length, 2) # 길이 (진행방향)                 #-------------|
                                                                                                    #
    # Overall volume                                                                                #
    cell= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, Sz+ 2*Lpml)                                            #-Total Volume|
    pml_layers = [mp.PML(thickness=Lpml)]#, direction=mp.Y)]                                        #-PML---------|
                                                                                                    #
    # 공간 비례 변수 (pml 제외 유효 공간)                                                             #     
    X_min= round(0.5*-Sx, 2)                                                                        #-Min points--|
    Y_min= round(0.5*-Sy, 2)                                                                        #             |
    Z_min= round(0.5*-Sz, 2)                                                                        #-------------|
                                                                                                    #
    X_max= round(0.5*Sx, 2)                                                                         #-Max points--|
    Y_max= round(0.5*Sy, 2)                                                                         #             |
    Z_max= round(0.5*Sz, 2)                                                                         #-------------|
                                                                                                    #
    wg_hc= round(X_min+ SiO2_h+ (LNsub_h+ LNwg_h)/2, 2)# middle of LNOI                             #-LNOI center-|
                                                                                                    #
    # design region                                                                                 #                      
    design_region_x = LNwg_h                                                                        #-Design size-|
    design_region_y = Sy                                                                            #             |
    design_region_z = Prop_length                                                                   #             |
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
Target_field=DFT_loader.load_target_field()
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

# Fixed mapping parameters for LNOI
Not_waveguide= False 
Min_s_top= 0.1  # 100 nm
eta_Ref= 0.05   # Fixed
eta_g= 0.45     # Fixed
Number_of_local_region= 4 # High NL lead to large angle of sidewalls due to the rounded pixels

evaluation_history = []
cur_iter = [0]
numevl = 1
FoM_old = 0
FoM_oold = 0
FoM_cur = 0
Symmetry_sim = False
Sym_geo_width = False
Sym_geo_length = False
# This is only for speacial case: Y mirror symmetry & EVEN number of design region pixel
if sim.symmetries:
    print('Y Mirror symmetry')
    width_grid = np.linspace(-design_region_y /2, design_region_y /2, Ny)                          
    length_grid = np.linspace(-design_region_z /2, design_region_z /2, Nz)                
    W_g, L_g = np.meshgrid(width_grid, length_grid, sparse=True, indexing="ij")           
    Half_line= ((W_g == W_g[round((Ny-1)/2)])& (np.abs(L_g)<design_region_z))
    Symmetry_sim= True
    Sym_geo_width= True
else:
    print('No symmetries in simulation')
    Symmetry_sim= False  

def mapping(x, beta):
    Map_idx = 3
    if Map_idx == 0:
        print('Step 0: Optimization without ANY constrain')
        x = Sub_Mapping.GrayScale_Grating(
            Width_symmetry= Sym_geo_width,
            Length_symmetry= Sym_geo_length,
            x= x, 
            N_width= Ny, 
            N_length= Nz, 
            N_height= Nx,            
        )
    elif Map_idx == 1:
        print('Step 1: Optimization with Binarization')
        x = Sub_Mapping.Binarization(
            Width_symmetry= Sym_geo_width,
            Length_symmetry= Sym_geo_length,
            x= x, 
            N_width= Ny, 
            N_length= Nz, 
            N_height= Nx,
            beta= beta,             
        )
    elif Map_idx == 2:
        print('Step 2: Optimization with MFS constrain')
        x = Sub_Mapping.TwoD_Filtered(
            Width_symmetry= Sym_geo_width,
            Length_symmetry= Sym_geo_length,
            x = x, 
            DR_width = design_region_y, 
            DR_length = design_region_z,
            DR_res = design_region_resolution, 
            N_height = Nx,
            Cone_rad = 0.05,
            beta = beta,
        )
    else:
        print('Step 3: Optimization with ALL constrain')
        min_g_59= 0.6
        x_copy = Sub_Mapping.get_reference_layer(
            Symmetry_in_Sim= Symmetry_sim,
            Width_symmetry= Sym_geo_width,
            Length_symmetry= Sym_geo_length,
            DR_width= design_region_y, 
            N_width= Ny, 
            DR_length= design_region_z, 
            N_length= Nz, 
            DR_res= design_region_resolution,
            Min_size_top= Min_s_top,
            Min_gap= min_g_59,
            input_w_top= input_w_top,
            w_top= w_top,
            x= x,
            eta_Ref= eta_Ref,
            beta= beta,               
        )
        x= Sub_Mapping.Slant_sidewall(
            Reference_layer= x_copy,
            DR_width= design_region_y, 
            DR_length= design_region_z, 
            DR_res= design_region_resolution,
            beta= beta,
            eta_gap= eta_g,
            Min_gap= min_g,
            N_height= Nx,
            Number_of_local_region= Number_of_local_region,
        )
    if Not_waveguide:
        x= (x.reshape(Nz, Nx*Ny)).transpose()
    x= npa.clip(x, 0.0, 1.0)
    return x.flatten()

n = Ny * Nz  # number of parameters
EPS_Step = 0.02

# Unit update function via forward and adjoint simulation
def f(v, cur_beta, flag: bool = False):
    global numevl
    X=mapping(v, cur_beta)
    print("Current beta: ", cur_beta)
    print("Current iteration: {}".format(cur_iter[0] + 1))
    if cur_beta == npa.inf:
        f0, dJ_du = opt(rho_vector=[X], need_gradient=False)  # compute objective and gradient
        evaluation_history.append(np.real(f0))
        numevl += 1
        print("First FoM: {}".format(evaluation_history[0]))
        print("Last FoM: {}".format(np.real(f0)))
        np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
        print("-----------------Optimization Complete------------------")
        return np.real(f0), v, X
    else:
        f0, dJ_du = opt(rho_vector=[X])  # compute objective and gradient
        if v.size > 0:
            gradient = v*0
            if flag:
                cur_beta=cur_beta*2
            gradient[:] = tensor_jacobian_product(mapping, 0)(
                v, cur_beta, dJ_du,
            )  # backprop
        evaluation_history.append(np.real(f0))
        #np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights) <---- geometry
        numevl += 1
        print("First FoM: {}".format(evaluation_history[0]))
        print("Current FoM: {}".format(np.real(f0)))
        
        dv = (EPS_Step)*((gradient)/(npa.clip(npa.max(npa.abs(gradient)), 1e-9, npa.inf)))
        #This is only for speacial case: Y mirror symmetry & EVEN number of designregion pixel
        if sim.symmetries: 
            delta_v= npa.reshape(dv,(Ny,Nz))
            delta_half= npa.where(Half_line, 0, delta_v)                                                 
            delta_v= ((npa.flipud(delta_half)) + delta_v)
            delta_v= delta_v.flatten()
            Updated_v = v + delta_v
        #######################################################################################
        else:
            Updated_v = v + dv
        Updated_v = npa.clip(Updated_v, 0, 1)
        cur_iter[0] = cur_iter[0] + 1
        return np.real(f0), Updated_v, X

x = np.ones((n,)) * 0.5
beta_init = 1
cur_beta = beta_init
beta_scale = 2
num_betas = 6
update_factor = 100

# Main optimization loop
outer_iter = 0
while outer_iter < (num_betas + 1):
    if outer_iter == num_betas: # Stop condition
        cur_beta = npa.inf
        Max_iter = 1
    else:
        Max_iter = update_factor
    inner_iter = 0      
    while inner_iter < Max_iter:
        FoM, x, Geo= f(x, cur_beta) # FoM, sliced geo, total geo

        print(np.max(Geo))
        print(np.min(Geo))

        FoM_oold = FoM_old
        FoM_old = FoM_cur
        FoM_cur = FoM
        
        FD_1 = FoM_old - FoM_oold
        FD_2 = FoM_cur - FoM_old
        
        if inner_iter > 5:
            if FD_2 == FD_1:
                print('Gradient is Zero')
                break 
            elif (FD_2/FD_1) < 1e-5:  # FoM tol
                np.savetxt(design_dir+"Geometry_iter"+str(numevl)+"_beta_"+str(cur_beta)+".txt", design_variables.weights)
                Bi_idx = npa.sum(npa.where(((1e-5<Geo)&(Geo<(1-1e-5))), 1, 0))
                print('Local optimum')
                if Bi_idx == 0:     # Binarization tol
                    print('Binarized')
                    outer_iter == num_betas -1         
                break
        inner_iter = inner_iter +1
    outer_iter = outer_iter +1 
    cur_beta = cur_beta * beta_scale         

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