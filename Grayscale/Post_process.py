""" Postprocess """
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
from mpi4py import MPI
import nlopt
import os
import h5py

comm= MPI.COMM_WORLD
rank= comm.Get_rank()

def Foward_prop (Wavelength, design):

    SiO2= mp.Medium(index=1.45)
    #My_LiNbO3= DFT_loader.My_LiNbO3_pso()
    My_LiNbO3= mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)) # X cut

    ##########################
    Geometry_profile= True
    Waveguide= True
    Geometry= True

    Capture_setup= False
    Target_source= True
    Reverse= True
    Top_view=0
    in_and_out=1
    #############################
    Main_Parameters= True       #
    Parameter_activation= True  #
    Monitor_Profile= True       #
    Source_profile= True        #########################################################################
    #                        ####  Set main parameters  ####                                            #
    if Main_Parameters:                                                                                 #-Parameters for Simulation--%
        resolution= int(100)                                                                            #-Resolution-----|
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
        design_region_resolution = int(50)                                                              #-------------|
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
        design_variables.update_weights(design) 
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
        Waveguide_input= [
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
                height= Sz+ 2*Lpml,
                axis= mp.Vector3(0, 0, 1),
                sidewall_angle= 0,
                material= My_LiNbO3,
            ),        
        ]
        Waveguide= [
            mp.Block(#LiNbO3 substrate
                center= mp.Vector3(X_min+ SiO2_h+ 0.5*LNsub_h, 0, 0),
                material= My_LiNbO3,
                size= mp.Vector3(LNsub_h, Sy+ 2*Lpml, Sz+ 2*Lpml) 
            ),   
            mp.Prism(#Output LiNbO3 waveguide
                vertices = [
                    mp.Vector3(X_min+ SiO2_h+ LNsub_h, w_bot/2, Z_min- Lpml),
                    mp.Vector3(X_min+ SiO2_h+ LNsub_h, -w_bot/2, Z_min- Lpml),
                    mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, -w_top/2, Z_min- Lpml),
                    mp.Vector3(X_min+ SiO2_h+ LNsub_h+ LNwg_h, w_top/2, Z_min- Lpml)
                ],
                height= Sz+ 2*Lpml,
                axis= mp.Vector3(0, 0, 1),
                sidewall_angle= 0,
                material= My_LiNbO3,
            ),        
        ]
    ###########################   Simulation #################################################
    if Waveguide:
        sim = mp.Simulation(
            cell_size= cell,
            resolution= resolution,
            boundary_layers= pml_layers,
            sources= source1,
            eps_averaging= False,
            geometry= Waveguide_input,
            extra_materials=[My_LiNbO3],
            default_material=SiO2, # 이거 Air vs LiNbO3
            symmetries=[mp.Mirror(mp.Y, phase=-1)],
        )
        input_fr= mp.FluxRegion(center= Incidence_center, size= Incidence_size)
        input_flux= sim.add_flux(fcen, fwidth, 1, input_fr)
        if in_and_out == 1:
            Input_Ex=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=source_center, size = dft_monitor_size)
            Input_Ey=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=source_center, size = dft_monitor_size) 
            Input_Ez=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=source_center, size = dft_monitor_size)
        #
        sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
        incidence_flux = mp.get_fluxes(input_flux)
        if in_and_out == 1: 
            Ex_Namei=f"Ex_fund_field"
            Ey_Namei=f"Ey_fund_field"
            Ez_Namei=f"Ez_fund_field"
            sim.output_dft(Input_Ez,Ez_Namei)
            sim.output_dft(Input_Ex,Ex_Namei)
            sim.output_dft(Input_Ey,Ey_Namei)
            #
            # Load DFT Field (Input)
            print(Ey_Namei+'.h5')
            hf = h5py.File(Ey_Namei+'.h5', 'r')
            A=hf.get('ey_0.r')
            R=np.array(A) # Real value
            #
            B=hf.get('ey_0.i')
            I=np.array(B)*1j # Imaginary value
            #
            R_Target_Field=R+I # Complex field
    ##
    if Geometry:
        sim.reset_meep()
        sim = mp.Simulation(
            cell_size= cell,
            resolution= resolution,
            boundary_layers= pml_layers,
            sources= source1,
            eps_averaging= False,
            geometry= geometry,
            extra_materials=[My_LiNbO3],
            default_material=SiO2, # 이거 Air vs LiNbO3
            symmetries=[mp.Mirror(mp.Y, phase=-1)],
        )
        # transmitted flux
        tran_fr = mp.FluxRegion(center= Eff_center, size= Eff_size)
        tran = sim.add_flux(fcen, fwidth, 1, tran_fr)
        #
        if Top_view==1:
            Yz_Ex=[0]*int(Nx+2)
            Yz_Ey=[0]*int(Nx+2)
            Yz_Ez=[0]*int(Nx+2)
            for i in range(0,Nx+2): # Z=source to Z=Sz-1, 100EA
                if i== 0:
                    Null=0.5
                else:
                    Null=1 
                Target_center=mp.Vector3(X_min+ SiO2_h+ Null*(LNsub_h)+ i*(0.5)/(Nx), 0, 0)
                Target_size=mp.Vector3(0, Sy, Sz)
                Yz_Ex[i]=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=Target_center, size = Target_size)
                Yz_Ey[i]=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=Target_center, size = Target_size) 
                Yz_Ez[i]=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=Target_center, size = Target_size) 
        if in_and_out == 1:
            # DFT 모니터 (출력단 E(x,y,z)) 
            Output_Ex=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
            Output_Ey=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size) 
            Output_Ez=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
        #
        if Capture_setup and rank==0:
            plt.figure()
            sim.plot2D(output_plane = mp.Volume(center=(0,0,0), size = mp.Vector3(mp.inf, mp.inf, 0)))
            plt.savefig("LN_converter_xy_plane.png")
            plt.cla()
            plt.clf()
            plt.close()

            plt.figure()
            sim.plot2D(output_plane = mp.Volume(center=(X_min+ SiO2_h+ LNsub_h+ LNwg_h, 0, 0), size = mp.Vector3(0, Sy+ 2*Lpml, mp.inf)))
            plt.savefig("LN_converter_yz_plane.png")
            plt.cla()
            plt.clf()
            plt.close()

            plt.figure()
            sim.plot2D(output_plane = mp.Volume(center=(0, 0, 0), size = mp.Vector3(mp.inf, 0, mp.inf)))
            plt.savefig("LN_converter_xz_plane.png")
            plt.cla()
            plt.clf()
            plt.close()
        # run the final simulation
        sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
        #
        Mode_profile = sim.get_eigenmode(
                frequency=fcen,
                where=mp.Volume(center=Adjoint_center, size=source_size),
                direction= mp.NO_DIRECTION,
                kpoint=kpoint,
                band_num=8,        
                parity=mp.NO_PARITY,
                match_frequency=True,
                eigensolver_tol=1e-12,
        )
        print(Mode_profile.k[2])
        wg_tran_flux = mp.get_fluxes(tran)
        Ts = []
        Ts = np.append(Ts, wg_tran_flux[0] / incidence_flux[0])
        np.savetxt('Mode_index.txt',Mode_profile.k/fcen) 
        #
        # .h5로 출력
        if Top_view==1:
            for i in range(0,Nx+2): 
                Ex_Namei=f"Ex_{i}_field"
                Ey_Namei=f"Ey_{i}_field"
                Ez_Namei=f"Ez_{i}_field"
                sim.output_dft(Yz_Ex[i],Ex_Namei)
                sim.output_dft(Yz_Ey[i],Ey_Namei)
                sim.output_dft(Yz_Ez[i],Ez_Namei)
        if in_and_out == 1: 
            Exo_Namei=f"Ex_opt_field"
            Eyo_Namei=f"Ey_opt_field"
            Ezo_Namei=f"Ez_opt_field"
            sim.output_dft(Output_Ex,Exo_Namei)
            sim.output_dft(Output_Ey,Eyo_Namei)
            sim.output_dft(Output_Ez,Ezo_Namei)
    ##
    if Reverse:
        sim.reset_meep()
        sim = mp.Simulation(
            cell_size= cell,
            resolution= resolution,
            boundary_layers= pml_layers,
            sources= source3,
            eps_averaging= False,
            geometry= geometry,
            extra_materials=[My_LiNbO3],
            default_material=SiO2, # 이거 Air vs LiNbO3
            symmetries=[mp.Mirror(mp.Y, phase=-1)],
        )        
        # transmitted flux (reverse)
        R_tran_fr = mp.FluxRegion(center= Incidence_center, size= Eff_size)
        R_tran = sim.add_flux(fcen, fwidth, 1, R_tran_fr)
        #
        if in_and_out == 1:
            # DFT 모니터 (출력단 E(x,y,z)) 
            R_Output_Ex=sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=source_center, size = dft_monitor_size)
            R_Output_Ey=sim.add_dft_fields([mp.Ey], fcen, 0, 1, center=source_center, size = dft_monitor_size) 
            R_Output_Ez=sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=source_center, size = dft_monitor_size)
        #
        sim.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
        Rev_tran_flux = mp.get_fluxes(R_tran)
        #
        if in_and_out == 1: 
            R_Exo_Namei=f"Ex_rev_field"
            R_Eyo_Namei=f"Ey_rev_field"
            R_Ezo_Namei=f"Ez_rev_field"
            sim.output_dft(R_Output_Ex,R_Exo_Namei)
            sim.output_dft(R_Output_Ey,R_Eyo_Namei)
            sim.output_dft(R_Output_Ez,R_Ezo_Namei)
            #
            # Load DFT Field (Reverse opt)
            print(R_Eyo_Namei+'.h5')
            hf = h5py.File(R_Eyo_Namei+'.h5', 'r')
            A=hf.get('ey_0.r')
            R=np.array(A) # Real value
            #
            B=hf.get('ey_0.i')
            I=np.array(B)*1j # Imaginary value
            #
            Reverse_Field=R+I # Complex field
            # Reverse process
            R_FoM_Y= npa.abs(npa.sum(Reverse_Field*R_Target_Field))**2 
            R_Purity= R_FoM_Y/((npa.sum(npa.abs(R_Target_Field)**2))*(npa.sum(npa.abs(Reverse_Field)**2)))            
    ##
    if Target_source:
        simT = mp.Simulation(
            cell_size= cell,
            resolution= resolution,
            boundary_layers= pml_layers,
            sources= source2,
            eps_averaging= False,
            geometry= Waveguide,
            extra_materials=[My_LiNbO3],
            default_material=SiO2, # 이거 Air vs LiNbO3
            symmetries=[mp.Mirror(mp.Y, phase=-1)],
        )
        Rev_fr= mp.FluxRegion(center= Incidence_center, size= Incidence_size)
        Rev_flux= simT.add_flux(fcen, fwidth, 1, Rev_fr)        
        # DFT 모니터 (입력단 E(x,y,z)) 
        if in_and_out == 1:
            Target_Ex=simT.add_dft_fields([mp.Ex], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
            Target_Ey=simT.add_dft_fields([mp.Ey], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size) 
            Target_Ez=simT.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_monitor_center, size = dft_monitor_size)
        if Capture_setup and rank==0:
            plt.figure()
            simT.plot2D(output_plane = mp.Volume(center=(0,0,0), size = mp.Vector3(mp.inf, mp.inf, 0)))
            plt.savefig("LN_waveguide_xy_plane.png")
            plt.cla()
            plt.clf()
            plt.close()

            plt.figure()
            simT.plot2D(output_plane = mp.Volume(center=(X_min+ SiO2_h+ LNsub_h+ LNwg_h, 0, 0), size = mp.Vector3(0, Sy+ 2*Lpml, mp.inf)))
            plt.savefig("LN_waveguide_yz_plane.png")
            plt.cla()
            plt.clf()
            plt.close()

            plt.figure()
            simT.plot2D(output_plane = mp.Volume(center=(0, 0, 0), size = mp.Vector3(mp.inf, 0, mp.inf)))
            plt.savefig("LN_waveguide_xz_plane.png")
            plt.cla()
            plt.clf()
            plt.close()
        #
        simT.run(until_after_sources= mp.stop_when_dft_decayed(1e-6, 0))
        Reverse_flux = mp.get_fluxes(Rev_flux)
        Rev_Ts = []
        Rev_Ts = np.append(Rev_Ts, - Rev_tran_flux[0] / Reverse_flux[0])
        if in_and_out == 1: 
            ExT_Namei=f"Ex_target_field"
            EyT_Namei=f"Ey_target_field"
            EzT_Namei=f"Ez_target_field"
            simT.output_dft(Target_Ex,ExT_Namei)
            simT.output_dft(Target_Ey,EyT_Namei)
            simT.output_dft(Target_Ez,EzT_Namei)
            #
            # Load DFT Field (target)
            print(EyT_Namei+'.h5')
            hf = h5py.File(EyT_Namei+'.h5', 'r')
            A=hf.get('ey_0.r')
            R=np.array(A) # Real value
            #
            B=hf.get('ey_0.i')
            I=np.array(B)*1j # Imaginary value
            #
            Target_Field=R+I # Complex field
            #
            # Load DFT Field (opt)
            print(Eyo_Namei+'.h5')
            hf = h5py.File(Eyo_Namei+'.h5', 'r')
            A=hf.get('ey_0.r')
            R=np.array(A) # Real value
            #
            B=hf.get('ey_0.i')
            I=np.array(B)*1j # Imaginary value
            #
            Output_Field=R+I # Complex field
            # Forward process
            FoM_Y= npa.abs(npa.sum(Output_Field*Target_Field))**2 
            Purity= FoM_Y/((npa.sum(npa.abs(Target_Field)**2))*(npa.sum(npa.abs(Output_Field)**2)))
    #
    print('transmission =',Ts[0]*100,'%')
    print('Purity =',Purity*100,'%')
    print('Conversion efficiency =', Purity*Ts[0]*100,'%')
    loss_val=[Purity*Ts[0]*100, Purity*100, Ts[0]*100, (1- Ts[0])*100]
    np.savetxt('WGeff.txt',loss_val)
    #
    print('Reverse transmission =',Rev_Ts[0]*100,'%')
    print('Reverse Purity =',R_Purity*100,'%')
    print('Reverse efficiency =', R_Purity*Rev_Ts[0]*100,'%')
    rev_val=[R_Purity*Rev_Ts[0]*100, R_Purity*100, Rev_Ts[0]*100]
    np.savetxt('Reveff.txt',rev_val)
    #
    sim.reset_meep()
    simT.reset_meep()
    os.chdir("..")


Spectral= False
direction=['A', 'B', 'C', 'D']
sub_direction1=['Single', 'PSO']

for dir in [0]:
    design_dir=direction[dir]+"/"
    if not os.path.exists(design_dir):
        try:
            os.makedirs(design_dir)
        except:
            print("already exist")
    os.chdir(direction[dir])
    Temp_design=np.loadtxt('lastdesign.txt')
    if Spectral:
        for dir1 in [1]:
            sub_dir1=sub_direction1[dir1]+"/"
            if not os.path.exists(sub_dir1):
                try:
                    os.makedirs(sub_dir1)
                except:
                    print("already exist")  
            os.chdir(sub_direction1[dir1])
            for temp_lambda in range(765,791,1):
                num=str(int(temp_lambda))
                sub_dir2=num+"/"
                if not os.path.exists(sub_dir2):
                    try:
                        os.makedirs(sub_dir2)
                    except:
                        print("already exist")  
                os.chdir(num)                
                Foward_prop(Wavelength= temp_lambda/1000, design= Temp_design)
            os.chdir("..")
        os.chdir("..")
    else:
        # sub_dir1=sub_direction1[0]+"/"
        # if not os.path.exists(sub_dir1):
        #     try:
        #         os.makedirs(sub_dir1)
        #     except:
        #         print("already exist")
        # os.chdir(sub_direction1[0])
        Foward_prop(Wavelength= 0.775, design= Temp_design)
        os.chdir("..")
