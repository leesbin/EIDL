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
#import DFT_loader
import os
import Sub_Mapping

# 'Top'
design_dir = "./A/"
mp.verbosity(1)
# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)
os.chdir(design_dir)

Air= mp.Medium(index=1.0)
SiO2= mp.Medium(index=1.4)
SiN = mp.Medium(index=2.1)
#My_LiNbO3= mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)) # X cut
#My_LiNbO3= DFT_loader.My_LiNbO3_pso()

#############################
Geometry_profile= True      #
Ex_only= False              #
Incidence= True             #
Geometry= True              #
Multi_layer= True
in_and_out=1
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
    # Width (x: width)                                                                              #-Width (X)------|
    X_Width= 1.0                                                                                    #----------------|
                                                                                                    #
    # Width (y: width)                                                                              #-Width (Y)------|
    Y_Width= 1.0                                                                                    #----------------|
                                                                                                    #
    # Propagation length (z: thickness)                                                             #
    Single_layer= 0.12
    Etch_stop_layer= 0.04

    src_2_geo= (1/resolution)*9 # top gap & Focal length                                                         #-Propagation (Z)|
    pml_2_src= (1/resolution)*5                                                                             #                |
    mon_2_pml= (1/resolution)*10                                                                                #                |
    Total_thickness= 0.8                                                                            #----------------|--> 3 to 4
                                                                                                    #
    fcen= 1.0/ 0.5 # 500 nm                                                                         #-Frequancy------|
    fwidth = 0.2 * fcen                                                                             #----------------|
#                     ####  Set main components and monitors  ####                                  #
if Parameter_activation:                                                                            #--Variables for simulation--%                          
    Sx = X_Width                                                                                    #-XYZ length--|
    Sy = Y_Width# Width: w/o pml, 3um                                                               #             |--> 3 to 4
    Sz = pml_2_src+ src_2_geo+ Total_thickness+ src_2_geo+ mon_2_pml # Thick (2.1 um)               #-------------|
                                                                                                    #
    # Overall volume                                                                                #
    cell= mp.Vector3(Sx, Sy, Sz+ 2*Lpml)                                                            #-Total Volume|
    pml_layers = [mp.PML(thickness=Lpml, direction=mp.Z)]                                           #-PML---------|
   
    
    #Bloch_K=  mp.Vector3(z=fcen*1.0).rotate(mp.Vector3(y=1), np.radians(20))
    Bloch_K= mp.Vector3()                                                                         #
    # 공간 비례 변수 (pml 제외 유효 공간)                                                             #     
    X_min= 0.5*-Sx                                                                                  #-Min points--|
    Y_min= 0.5*-Sy                                                                                  #             |
    Z_min= 0.5*-Sz                                                                                  #-------------|
                                                                                                    #
    X_max= 0.5*Sx                                                                                   #-Max points--|
    Y_max= 0.5*Sy                                                                                   #             |
    Z_max= 0.5*Sz                                                                                   #-------------|
                                                                                                    #
    Layer1_h= Z_max- pml_2_src- src_2_geo- Single_layer/2 # middle of layer                                  #-Layer center-|
    Layer2_h= Z_max- pml_2_src- src_2_geo- Single_layer*1.5 - Etch_stop_layer # middle of layer                  #-Layer center-|
    Layer3_h= Z_max- pml_2_src- src_2_geo- Single_layer*2.5 - Etch_stop_layer*2 # middle of layer                  #-Layer center-|
    Layer4_h= Z_max- pml_2_src- src_2_geo- Single_layer*3.5 - Etch_stop_layer*3 # middle of layer                  #-Layer center-|
    Layer5_h= Z_max- pml_2_src- src_2_geo- Single_layer*4.5 - Etch_stop_layer*4 # middle of layer                  #-Layer center-|
    ESL_center_const= -(Single_layer+Etch_stop_layer)/2
                                                                                                    #
    # design region                                                                                 #                      
    design_region_x = Sx                                                                            #-Design size-|
    design_region_y = Sy                                                                            #             |
    design_region_z = Single_layer                                                                  #             |
    design_region_resolution = int(resolution)                                                      #-------------|
                                                                                                    #
    Nx = int(design_region_resolution * design_region_x)+ 1                                         #-Grid points-|
    Ny = int(design_region_resolution * design_region_y)+ 1                                         #             |
    Nz = int(design_region_resolution * design_region_z)+ 1                                         #-------------|
if Monitor_Profile:                                                                                 #---Center & Size of Monitor-%
    # Adjoint source profile<-------------------------------시뮬레이션 조건 바뀌면 체크               #
    Adjoint_center= mp.Vector3(0, 0, Z_min+ mon_2_pml)                                              #-Adjoint-----|
    Adjoint_size= mp.Vector3(0.04, 0.04, 0)                                                         #-------------|
                                                                                                    
    # Fundamental source (TE00)                                                                     #
    source_center= mp.Vector3(0, 0, Z_max- pml_2_src)                                               #-Source------|
    source_size= mp.Vector3(Sx, Sy, 0)                                                              #-------------|
                                                                                                    
    output_monitor_center= mp.Vector3(0, 0, Z_min+ mon_2_pml)                                       #-Output(F)---|
    output_monitor_size= mp.Vector3(Sx, Sy, 0)                                                      #-------------|
    
    dft_monitor_center= mp.Vector3(0, 0, 0)                                                         #-Output(V)---|
    dft_monitor_size= mp.Vector3(Sx, 0, Sz)                                                         #-------------|
                                                                                                    
    Incidence_center= mp.Vector3(0, 0, Z_max - (pml_2_src) - (1/resolution)*2 )                                            #-Input-------|
    Incidence_size= mp.Vector3(Sx, Sy, 0)                                                           #-------------|

    Transmission_center= mp.Vector3(0, 0, Z_min+ mon_2_pml)                                         #-Transmission|
    Transmission_size= mp.Vector3(Sx, Sy, 0)                                                        #-------------|

    Eff_center= Adjoint_center                                                                      #-Efficiency--|
    Eff_size= mp.Vector3(0.04, 0.04, 0)                                                             #-------------|

if Source_profile:                                                                                  #----Sources for simulation--%
    kpoint = mp.Vector3(0,0,1)                                                                      #-K vector----|
    # def pw_amp(k, x0):
    #     def _pw_amp(x,y):
    #         return cmath.exp(1j * 2 * math.pi * k.dot(x + x0,y+x0, z+x0))

    #     return _pw_amp
    src = mp.GaussianSource(frequency= fcen, fwidth= fwidth, is_integrated=True)                  #-Unit source-|         


    # source1 = []
    # for i in range(round(resolution*Sx)):
    #     for j in range(round(resolution*Sy)):
    #         # if i%3==1 or i%3==2:
    #         #     continue
    #         print(i)
    #         source1.append(
    #             mp.Source(
    #                 src,
    #                 component=mp.Ex,
    #                 center=mp.Vector3(
    #                     x = X_min+ Sx*(i/(Sx*resolution))+0.5/resolution, 
    #                     y = Y_min+ Sy*(j/(Sy*resolution))+0.5/resolution, 
    #                     z= Z_max-pml_2_src
    #                     ),
    #                 amplitude = np.exp(1j * 2 * math.pi * Bloch_K.x*(X_min+ Sx*(i/(Sx*resolution))+0.5/resolution)),
    #                         ),
    #                     )                                                                                      
    source1 = [                                                                                     #-TE00 (775)--|
        mp.Source(                                                                                  #             |
            src,                                                                                    #             |
            size= source_size,                                                                      #             |
            component=mp.Ex,                                                                        #             |
            center= source_center,
            #amp_func= pw_amp(Bloch_K, source_center)                                                                  #             |
        ),                                                                                          #             |
        # mp.Source(                                                                                  #             |
        #     src,                                                                                    #             |
        #     size= source_size,                                                                      #             |
        #     component=mp.Ey,                                                                        #             |
        #     center= source_center,                                                                  #             |
        # )
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
    Plane= np.ones((Nx * Ny * Nz,)) * 0
    grating = npa.reshape(Plane,(Nx,Ny,Nz))
    #grating[int(Nx/2)-10:int(Nx/2)+10, :]= 1
    #grating[1, :, :]= 1
    #grating[int((Nx-1)/2)+5, :, :]= 1
    grating[0, :, :]= 1
    grating= grating.flatten()
    #design= Sub_Mapping.TwoD_freeform_gray(grating, Nx, Ny, Nz) 
    structure_weight_0 = np.loadtxt('modified_lastdesign.txt')

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN")
    design_variables.update_weights(structure_weight_0) 
    # np.savetxt("modified_lastdesign.txt", design_variables.weights)
    print(grating)
    if Multi_layer:
        DR_1 = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, Layer1_h),
                size=mp.Vector3(design_region_x, design_region_y, design_region_z),
            ),
        )
        ESL_1 = mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Layer1_h- ESL_center_const),
            material= SiO2,
            size= mp.Vector3(Sx, Sy, Etch_stop_layer) 
        )
        DR_2 = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, Layer2_h),
                size=mp.Vector3(design_region_x, design_region_y, design_region_z),
            ),
        )
        ESL_2 = mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Layer2_h- ESL_center_const),
            material= SiO2,
            size= mp.Vector3(Sx, Sy, Etch_stop_layer) 
        )
        DR_3 = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, Layer3_h),
                size=mp.Vector3(design_region_x, design_region_y, design_region_z),
            ),
        )
        ESL_3 = mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Layer3_h- ESL_center_const),
            material= SiO2,
            size= mp.Vector3(Sx, Sy, Etch_stop_layer) 
        )
        DR_4 = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, Layer4_h),
                size=mp.Vector3(design_region_x, design_region_y, design_region_z),
            ),
        )
        ESL_4 = mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Layer4_h- ESL_center_const),
            material= SiO2,
            size= mp.Vector3(Sx, Sy, Etch_stop_layer) 
        )
        DR_5 = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, Layer5_h),
                size=mp.Vector3(design_region_x, design_region_y, design_region_z),
            ),
        )
        ESL_5 = mp.Block(#SiO2 substrate
            center= mp.Vector3(0, 0, Layer5_h- ESL_center_const),
            material= SiO2,
            size= mp.Vector3(Sx, Sy, Etch_stop_layer) 
        )        
    # 구조물 Z축을 따라 pml 영역까지 배치
    geometry= [
        #ESL_1,    
        mp.Block(#Design region
            center=DR_1.center, size=DR_1.size, material=design_variables
        ),  
        # mp.Block(#Design region
        #     center=DR_5.center, size=DR_5.size, material=design_variables
        # ),  
    ]
###########################   Simulation #################################################
if Incidence:
    sim = mp.Simulation(
        cell_size= cell,
        resolution= resolution,
        boundary_layers= pml_layers,
        k_point= Bloch_K,
        sources= source1,
        eps_averaging= False,
        default_material=Air, # 빈공간
        # extra_materials=[SiO2,SiN],
    )
    input_fr= mp.FluxRegion(center= Incidence_center, size= Incidence_size)
    input_flux= sim.add_flux(fcen, fwidth, 1, input_fr)
    if in_and_out == 1:
        Input_Ex_H=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=source_center, size = output_monitor_size)
        Input_Ey_H=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=source_center, size = output_monitor_size) 
        Input_Ez_H=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=source_center, size = output_monitor_size)

        Input_Ex_V=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
        Input_Ey_V=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size) 
        Input_Ez_V=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
    #
    sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
    incidence_flux = mp.get_fluxes(input_flux)
    incidence_flux_data = sim.get_flux_data(input_flux)
    if in_and_out == 1: 
        Ex_Namei=f"Ex_Input_H_field"
        Ey_Namei=f"Ey_Input_H_field"
        Ez_Namei=f"Ez_Input_H_field"
        sim.output_dft(Input_Ex_H,Ex_Namei)
        sim.output_dft(Input_Ey_H,Ey_Namei)
        sim.output_dft(Input_Ez_H,Ez_Namei)

        Ex_Namei=f"Ex_Input_V_field"
        Ey_Namei=f"Ey_Input_V_field"
        Ez_Namei=f"Ez_Input_V_field"
        sim.output_dft(Input_Ex_V,Ex_Namei)
        sim.output_dft(Input_Ey_V,Ey_Namei)
        sim.output_dft(Input_Ez_V,Ez_Namei)
        #
##
if Geometry:
    sim.reset_meep()
    sim = mp.Simulation(
        cell_size= cell,
        resolution= resolution,
        boundary_layers= pml_layers,
        k_point= Bloch_K,
        sources= source1,
        eps_averaging= False,
        geometry= geometry,
        default_material=Air, # 빈공간
        # extra_materials=[SiO2,SiN],
    )
    # reflected flux
    refl_fr = mp.FluxRegion(center= Incidence_center, size= Incidence_size)
    refl = sim.add_flux(fcen, fwidth, 1, refl_fr)    
    sim.load_minus_flux_data(refl, incidence_flux_data)
    # transmitted flux
    tran_fr = mp.FluxRegion(center= Transmission_center, size= Transmission_size)
    tran = sim.add_flux(fcen, fwidth, 1, tran_fr)
    # fucused flux
    focused_fr = mp.FluxRegion(center= Adjoint_center, size= Adjoint_size)
    focused = sim.add_flux(fcen, fwidth, 1, focused_fr)
    #
    if in_and_out == 1:
        # DFT 모니터 (출력단 E(x,y,z)) 
        Output_Ex_H=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=output_monitor_center, size = output_monitor_size)
        Output_Ey_H=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=output_monitor_center, size = output_monitor_size) 
        Output_Ez_H=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=output_monitor_center, size = output_monitor_size)

        Output_Ex_V=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
        Output_Ey_V=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size) 
        Output_Ez_V=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
    #
    # run the final simulation
    sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
    #
    tran_flux = mp.get_fluxes(tran)
    Ts = []
    Ts = np.append(Ts, tran_flux[0] / incidence_flux[0])

    refl_flux = mp.get_fluxes(refl)
    Rs = []
    Rs = np.append(Rs, -refl_flux[0] / incidence_flux[0])

    focused_flux = mp.get_fluxes(focused)
    Fs = []
    Fs = np.append(Fs, focused_flux[0] / incidence_flux[0])    
    #
    if in_and_out == 1: 
        Exo_Namei=f"Ex_Output_H_field"
        Eyo_Namei=f"Ey_Output_H_field"
        Ezo_Namei=f"Ez_Output_H_field"
        sim.output_dft(Output_Ex_H,Exo_Namei)
        sim.output_dft(Output_Ey_H,Eyo_Namei)
        sim.output_dft(Output_Ez_H,Ezo_Namei)

        Exo_Namei=f"Ex_Output_V_field"
        Eyo_Namei=f"Ey_Output_V_field"
        Ezo_Namei=f"Ez_Output_V_field"
        sim.output_dft(Output_Ex_V,Exo_Namei)
        sim.output_dft(Output_Ey_V,Eyo_Namei)
        sim.output_dft(Output_Ez_V,Ezo_Namei)

    print('transmission =',Ts[0]*100,'%')
    print('reflection =',Rs[0]*100,'%')
    print('loss =',(1-(Ts[0]+Rs[0]))*100,'%')
    print('Focused efficiency =',Fs[0]*100,'%')
    loss_val=[Ts[0]*100, Rs[0]*100, (1-(Ts[0]+Rs[0]))*100, Fs[0]*100]
    np.savetxt('TReff.txt',loss_val)
##
sim.reset_meep()
os.chdir("..")
# ####################################################################################################
# sim = mp.Simulation(
#     cell_size= cell,
#     resolution= resolution,
#     boundary_layers= pml_layers,
#     sources= source1,
#     eps_averaging= False,
#     geometry= geometry,
#     default_material=SiO2,
#     extra_materials=[My_LiNbO3],
#     symmetries=[mp.Mirror(mp.Y, phase=-1)],
# )

# """ Adjoint optimization """
# # Get TE20 profile [Ex, Ey, Ez]
# if Ex_only: # Consider Main components of TE20 field
#     Target_field=DFT_loader.load_target_field()

#     # TE20 mode (Ex dft-field < we need to consider the resolution)
#     FoM_x = mpa.FourierFields(
#         sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ex ) #<------------------FoM calc
#     # input of J
#     ob_list = [FoM_x]

#     # Figure of Merit function
#     def J(mode_x):
#         A_field=mode_x[0]
#         X=npa.abs(npa.sum(A_field*Target_field[0].conjugate()))**2/(npa.sum(npa.abs(Target_field[0])**2)*(npa.sum(npa.abs(A_field)**2)))
#         return X
# else: # Consider all components of TE20 field
#     Target_field=DFT_loader.load_target_field(is_consider_Ex_only=False)
#     # TE20 mode (Ex, Ey and Ez dft-field < we need to consider the resolution)
#     FoM_x = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ex ) #<------------------FoM calc
#     FoM_y = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ey )
#     FoM_z = mpa.FourierFields(sim, mp.Volume(center= Adjoint_center, size= Adjoint_size), mp.Ez )
#     # input of J
#     ob_list = [FoM_x, FoM_y, FoM_z]
#     # Figure of Merit function
#     def J(mode_x, mode_y, mode_z):
#         x_field=mode_x[0]
#         y_field=mode_y[0]
#         z_field=mode_z[0]

#         X=npa.abs(npa.sum(x_field*Target_field[0].conjugate()))**2
#         Y=npa.abs(npa.sum(y_field*Target_field[1].conjugate()))**2    
#         Z=npa.abs(npa.sum(z_field*Target_field[2].conjugate()))**2

#         Xp=X/(npa.sum(npa.abs(Target_field[0])**2)*(npa.sum(npa.abs(x_field)**2)))
#         Yp=Y/(npa.sum(npa.abs(Target_field[1])**2)*(npa.sum(npa.abs(y_field)**2)))    
#         Zp=Z/(npa.sum(npa.abs(Target_field[2])**2)*(npa.sum(npa.abs(z_field)**2)))
#         print('Current purity: X=', Xp,', Y=', Yp, ', Z=', Zp)
#         Purity = str(Xp) + '\n' + str(Yp) + '\n' + str(Zp)
#         with open(design_dir+"last_purity.txt", "w") as text_file:
#             text_file.write(Purity)

#         return npa.sum(X+Y+Z)

# opt = mpa.OptimizationProblem(
#     simulation=sim,
#     objective_functions=J,
#     objective_arguments=ob_list,
#     design_regions=[design_region],
#     fcen=fcen,
#     df=0,
#     nf=1,
#     minimum_run_time=10,
#     decay_by=1e-6,
# )

# Min_s_top= 0.1 
# eta_Ref= 0.05
# eta_g= 0.45
# Number_of_local_region= 4 # This should be odd number for 'Mid' 

# def mapping(x, beta):
#     x_copy = Sub_Mapping.get_reference_layer(
#         DR_width= design_region_y, 
#         N_width= Ny, 
#         DR_length= design_region_z, 
#         N_length= Nz, 
#         DR_res= design_region_resolution,
#         Min_size_top= Min_s_top,
#         Min_gap= min_g,
#         input_w_top= input_w_top,
#         input_w_bot= input_w_bot,
#         w_top= w_top,
#         w_bot= w_bot,
#         x= x,
#         eta_Ref= eta_Ref,
#         beta= beta,               
#     )
#     x= Sub_Mapping.Slant_sidewall(
#         Reference_layer= x_copy,
#         DR_width= design_region_y, 
#         DR_length= design_region_z, 
#         DR_res= design_region_resolution,
#         beta= beta,
#         eta_gap= eta_g,
#         Min_gap= min_g,
#         N_height= Nx,
#         Number_of_local_region= Number_of_local_region,
#     )
#     return x



# evaluation_history = []
# cur_iter = [0]
# numevl = 1

# def f(v, gradient, beta):
#     global numevl
#     print("Current iteration: {}".format(cur_iter[0] + 1))
#     f0, dJ_du = opt([mapping(v, beta)])  # compute objective and gradient
#     #np.savetxt(design_dir+"Test"+str(numevl) +".txt", dJ_du) <---- adjoint gradient
#     print(dJ_du.size)
#     if beta == npa.inf:
#         evaluation_history.append(np.real(f0))
#         numevl += 1
#         print("First FoM: {}".format(evaluation_history[0]))
#         print("Last FoM: {}".format(np.real(f0)))
#         np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
#         print("-----------------Optimization Complete------------------")
#         return np.real(f0)
#     else:
#         if gradient.size > 0:
#             gradient[:] = tensor_jacobian_product(mapping, 0)(
#                 v, beta, dJ_du,
#             )  # backprop
#             print(gradient.size)
#             #np.savetxt(design_dir+"grad"+str(numevl) +".txt", gradient) <---- gradient for nlopt
#         evaluation_history.append(np.real(f0))
#         #np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights) <---- geometry
#         numevl += 1
#         print("First FoM: {}".format(evaluation_history[0]))
#         print("Current FoM: {}".format(np.real(f0)))
#         cur_iter[0] = cur_iter[0] + 1
#         return np.real(f0)

# algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# n = Ny * Nz  # number of parameters

# # Initial guess - 초기 시작값 0.5
# #x = np.random.uniform(0.45, 0.55, n)
# x = np.ones((n,)) * 0.5

# #np.savetxt(design_dir+"lastdesign.txt", x)
# #lower and upper bounds (상한 : 1, 하한 : 0)
# lb = np.zeros((Ny * Nz))
# ub = np.ones((Ny * Nz))

# beta_init = 2
# cur_beta = beta_init
# beta_scale = 2
# num_betas = 5
# update_factor = 20

# for iters in range(num_betas + 1):
#     solver = nlopt.opt(algorithm, n)
#     solver.set_lower_bounds(lb) # lower bounds
#     solver.set_upper_bounds(ub) # upper bounds

#     if cur_beta >= (beta_init * (beta_scale ** (num_betas))):
#         cur_beta = npa.inf
#         print("current beta: ", cur_beta)
#         solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
#         solver.set_maxeval(1) # Set the maximum number of function evaluations
#     else:
#         print("current beta: ", cur_beta)
#         solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
#         solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
#         cur_beta = cur_beta * beta_scale
#     # solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
#     x[:] = solver.optimize(x)

    
# plt.figure()

# plt.plot(evaluation_history, "o-")
# plt.grid(True)
# plt.xlabel("Iteration")
# plt.ylabel("FoM")
# plt.savefig(design_dir+"result.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# np.savetxt(design_dir+"evaluation.txt", evaluation_history)
# np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)

