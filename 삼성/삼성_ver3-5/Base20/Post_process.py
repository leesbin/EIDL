'''
SPDX-FileCopyrightText: © Sangbin Lee (kairus0320@gmail.com)
SPDX-License-Identifier: Proprietary License
    - This code was developed by the EIDL at Hanyang University and is subject to the following conditions.
    - Hynix_multilayer_3D_CIS is only limited to SK Hynix CIS pixel team and also it is only limited to co-research use.
    - Prohibit unauthorized third-party sharing.
    - Prohibit other research that reworks the code.
'''
'''
# ----------------------------------------------------------------------------
# -                        EIDL: sites.google.com/view/eidl                  -
# ----------------------------------------------------------------------------
# Author: Sangbin Lee (EIDL, Hanyang University)
# Date: 2024-03-18
​
###------------- Code description -------------###
3D CIS simulation using Meep-Adjoint

'''
# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os

mp.verbosity(1)

# Simulation Condtion ###############################################
Norm = True # FoM normalization                                     #
Dipersion_model = False # dispersion model                          #
#####################################################################

# Fab constraint ####################################################
FL = True # Focal layer                                             #
Air_Conditions = False # Air grid                                   #
DTI = False # DTI layer                                             #
AR = False # AR coating                                             #
#####################################################################

# PostProcess Condtion ##############################################
simcondition = True                                                 #
designplot = True                                                   #
DFTfield = False                                                     #
Opticaleff = True                                                   #  
layerplot = True                                                    #
Sensitivity = True                                                  #
Crosstalk = True                                                    #
Discrete = True                                                     #
#####################################################################

x_list = [0,1,2]

def make_dir(i):
    if i == 0:
        directory_name = "x-polarization"
    elif i == 1:
        directory_name = "y-polarization"
    elif i == 2:
        directory_name = "xy-polarization"
    else:
        print("잘못된 입력입니다.")
        return
    
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            print("디렉토리 생성 성공:", directory_name)
        else:
            print("디렉토리 이미 존재합니다:", directory_name)
    except OSError as e:
        print("디렉토리 생성 실패:", directory_name, "| 에러:", e)
    else:
        try:
            sub_folders = ['Crosstalk', 'Design', 'QE_data', 'Sensitivity_data', 'DFTfield', 'Discrete']
            for folder_name in sub_folders:
                sub_folder_path = os.path.join(directory_name, folder_name)
                if not os.path.exists(sub_folder_path):
                    os.makedirs(sub_folder_path)
                    print("폴더 생성 성공:", sub_folder_path)
                else:
                    print("폴더 이미 존재합니다:", sub_folder_path)
        except Exception as e:
            print("폴더 생성 실패:", directory_name, "| 에러:", e)

for i in x_list:
    make_dir(i)
    if i == 0:
        directory_name = "x-polarization"
    elif i == 1:
        directory_name = "y-polarization"
    elif i == 2:
        directory_name = "xy-polarization"

    if simcondition:  
        # scaling & refractive index#########################################
        um_scale = 1 # 1A = 1 um                                            #
                                                                            #
        Air = mp.Medium(index=1.0)                                          #
        f_SiO2 = mp.Medium(index=1.4)                                       #
        ARcoating = mp.Medium(index=1.7)                                    #
                                                                            #
        if Dipersion_model:                                                 #
            from Materials import SiN                                       #
            from Materials import SiO2                                      #
            from Materials import HfO2                                      # 
            from Materials import Si                                        #
            from Materials import Al2O3                                     #
        else:                                                               #
            Air = mp.Medium(index=1.0)                                      #
            SiO2 = mp.Medium(index=1.45) # n=1.4529 @ 0.55um                #
            TiO2 = mp.Medium(index=2.65) # n=2.6479 @ 0.55um                #
            Si = mp.Medium(index=3.00)                                      #
                                                                            #
                                                                            #
        #####################################################################

        # resolution ########################################################
        resolution = 50                                                     #
        o_grid = 1/resolution # 20nm                                        #
        design_region_resolution = int(resolution)                          #
        #####################################################################
        
        if True: ##### Set main parameters ############################################################################################################################                                                                                                                                  
            if AR:                                                                               
                ar_thktop = round(o_grid * 0, 2) # ARtop thickness 0 nm                          
                ar_thkbottom = round(o_grid * 0, 2) # ARbottom thickness 0 nm                    
            else:                                                                                
                ar_thktop = round(o_grid * 0, 2)                                                 
                ar_thkbottom = round(o_grid * 0, 2)                                              
            if FL:                                                                               
                fl_thickness = round(o_grid * 40, 2) # focal layer thickness 800 nm              
            else:                                                                                
                fl_thickness = round(o_grid * 0, 2)
            
            ml_thickness_1 = round(o_grid * 20, 2) # multi layer thickness 200 nm 
            ml_thickness_2 = round(o_grid * 20, 2) # multi layer thickness 200 nm 
            ml_thickness_3 = round(o_grid * 20, 2) # multi layer thickness 200 nm 
            ml_thickness_4 = round(o_grid * 20, 2) # multi layer thickness 200 nm  
            ml_thickness_5 = round(o_grid * 20, 2) # multi layer thickness 200 nm 
            el_thickness = round(o_grid * 0, 2) # etch layer thickness 0 nm
            
            if DTI:
                dti_size = round(o_grid * 3, 2) # DTI size 60 nm
                dti_thickness = round(o_grid * 20, 2) # DTI thickness 400 nm
            else:
                dti_size = round(o_grid * 0, 2) # DTI size 0 nm
                dti_thickness = round(o_grid * 0, 2) # DTI thickness 0 nm

            if Air_Conditions:
                air_thickness = round(o_grid * 15, 2)  # air thickness 300 nm
                air_size = round(o_grid * 3, 2) # air size 60 nm
            else:
                air_thickness = round(o_grid * 0, 2) # air thickness 0 nm
                air_size = round(o_grid * 0, 2) # air size 0 nm

            sp_size = round(o_grid * 30, 2) # subpixel size 600 nm
            sd_size = round(sp_size - dti_size, 2) # real SiPD size

        if True: #####  Set main components and monitors  #############################################################################################################
            Lpml = round(o_grid * 10, 2) # PML thickness 400 nm
            pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]

            pml_2_src = round(o_grid * 5, 2) # PML to source 100 nm
            src_2_geo = round(o_grid * 5, 2) # Source to geometry 100 nm
            mon_2_pml = round(o_grid * 10, 2) # Monitor to PML 200 nm

            # Design region size
            design_region_width_x = round(sp_size * 2 , 2) # Design region x 1120 nm
            design_region_width_y = round(sp_size * 2 , 2) # Design region y 1120 nm
            design_region_height = round(ml_thickness_1 + ml_thickness_2 + ml_thickness_3 + ml_thickness_4 + ml_thickness_5 + el_thickness*4, 2) # Design region z 1160 nm   

            # Overall cell size
            Sx = design_region_width_x
            Sy = design_region_width_y
            Sz = round(Lpml +  pml_2_src + src_2_geo + ar_thktop + design_region_height + fl_thickness + ar_thkbottom + mon_2_pml + Lpml, 2)
            cell_size = mp.Vector3(Sx, Sy, Sz)

                # Design Region mesh size
            Nx = int(round(design_region_resolution * design_region_width_x)) + 1
            Ny = int(round(design_region_resolution * design_region_width_y)) + 1 
            Nz = int(round(design_region_resolution * design_region_height)) + 1

        if True: #####  Set wavelengths and sources  ##################################################################################################################
            wavelengths = np.linspace(0.425*um_scale, 0.655*um_scale, 24) 
            frequencies = 1/wavelengths
            nf = len(frequencies) # number of frequencies

        # source 설정
            width = 2
            frequency = 1/(0.550*um_scale)
            fwidth = frequency * width
            src = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)

            source_center = [0, 0, round(Sz / 2 - Lpml - pml_2_src, 2) ] # Source 위치
            source_size = mp.Vector3(Sx, Sy, 0)
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]

        if True: #####  Set geometry  ################################################################################################################################
            # structure load
            structure_weight_0 = np.loadtxt('lastdesign1.txt')
            structure_weight_1 = np.loadtxt('lastdesign2.txt')
            structure_weight_2 = np.loadtxt('lastdesign3.txt')
            structure_weight_3 = np.loadtxt('lastdesign4.txt')
            structure_weight_4 = np.loadtxt('lastdesign5.txt')

            structure_weight_0 = mpa.tanh_projection(structure_weight_0, np.inf, 0.5)
            structure_weight_1 = mpa.tanh_projection(structure_weight_1, np.inf, 0.5)
            structure_weight_2 = mpa.tanh_projection(structure_weight_2, np.inf, 0.5)
            structure_weight_3 = mpa.tanh_projection(structure_weight_3, np.inf, 0.5)
            structure_weight_4 = mpa.tanh_projection(structure_weight_4, np.inf, 0.5)

            # 설계 영역과 물질을 바탕으로 설계 영역 설정
            design_variables_0 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
            design_variables_0.update_weights(structure_weight_0)
            design_region_0 = mpa.DesignRegion(
                design_variables_0,
                volume=mp.Volume(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1/2, 2)),
                    size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_1),
                ),
            )

            design_variables_1 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
            design_variables_1.update_weights(structure_weight_1)
            design_region_1 = mpa.DesignRegion(
                design_variables_1,
                volume=mp.Volume(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2/2, 2)),
                    size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_2),
                ),
            )

            design_variables_2 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
            design_variables_2.update_weights(structure_weight_2)
            design_region_2 = mpa.DesignRegion(
                design_variables_2,
                volume=mp.Volume(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3/2, 2)),
                    size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_3),
                ),
            )

            design_variables_3 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
            design_variables_3.update_weights(structure_weight_3)
            design_region_3 = mpa.DesignRegion(
                design_variables_3,
                volume=mp.Volume(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3 - el_thickness - ml_thickness_4/2, 2)),
                    size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_4),
                ),
            )

            design_variables_4 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
            design_variables_4.update_weights(structure_weight_4)
            design_region_4 = mpa.DesignRegion(
                design_variables_4,
                volume=mp.Volume(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3 - el_thickness - ml_thickness_4 - el_thickness - ml_thickness_5/2, 2)),
                    size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_5),
                ),
            )

            # design region과 동일한 size의 Block 생성
            geometry = [
                # # # Focal Layer
                # mp.Block(
                #     center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height - fl_thickness/2, 2)), size=mp.Vector3(Sx, Sy, fl_thickness), material=SiO2
                # ),
                # # SiPD
                # mp.Block(
                #     center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height - fl_thickness - ar_thkbottom - mon_2_pml/2 - Lpml/2, 2)), size=mp.Vector3(Sx, Sy, mon_2_pml + Lpml), material=Si
                # ),
                # design region과 동일한 size의 Block 생성
                mp.Block(
                    center=design_region_0.center, size=design_region_0.size, material=design_variables_0
                ),
                mp.Block(
                    center=design_region_1.center, size=design_region_1.size, material=design_variables_1
                ),
                mp.Block(
                    center=design_region_2.center, size=design_region_2.size, material=design_variables_2
                ),
                mp.Block(
                    center=design_region_3.center, size=design_region_3.size, material=design_variables_3
                ),
                mp.Block(
                    center=design_region_4.center, size=design_region_4.size, material=design_variables_4
                ),
            ]

         # Meep simulation environment
    
        sim = mp.Simulation(
            cell_size=cell_size, 
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0), # periodic boundary condition
            eps_averaging= False, 
            # extra_materials=[SiO2, SiN, Si, HfO2, Al2O3],/
        )

        if True: ##### Optimization Environment ############################################################################################################################
            # Monitor position
            monitor_position_1, monitor_size_1 = mp.Vector3(design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + mon_2_pml - 1/resolution), mp.Vector3(sp_size/2,sp_size/2,0) 
            monitor_position_2, monitor_size_2 = mp.Vector3(-design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + mon_2_pml - 1/resolution), mp.Vector3(sp_size/2,sp_size/2,0) 
            monitor_position_3, monitor_size_3 = mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + mon_2_pml - 1/resolution), mp.Vector3(sp_size/2,sp_size/2,0) 
            monitor_position_4, monitor_size_4 = mp.Vector3(design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + mon_2_pml - 1/resolution), mp.Vector3(sp_size/2,sp_size/2,0) 
            
            # FourierFields
            FourierFields_1_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ex,yee_grid=True)
            FourierFields_2_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ex,yee_grid=True)
            FourierFields_3_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ex,yee_grid=True)
            FourierFields_4_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Ex,yee_grid=True)

            FourierFields_1_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
            FourierFields_2_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
            FourierFields_3_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
            FourierFields_4_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Ey,yee_grid=True)


            FourierFields_1_x_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Hy,yee_grid=True)
            FourierFields_2_x_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Hy,yee_grid=True)
            FourierFields_3_x_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Hy,yee_grid=True)
            FourierFields_4_x_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Hy,yee_grid=True)

            FourierFields_1_y_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Hx,yee_grid=True)
            FourierFields_2_y_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Hx,yee_grid=True)
            FourierFields_3_y_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Hx,yee_grid=True)
            FourierFields_4_y_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Hx,yee_grid=True)

            ob_list = [FourierFields_1_x, FourierFields_1_y, FourierFields_2_x,FourierFields_2_y, FourierFields_3_x, FourierFields_3_y, FourierFields_4_x, FourierFields_4_y, FourierFields_1_x_h, FourierFields_1_y_h, FourierFields_2_x_h,FourierFields_2_y_h, FourierFields_3_x_h, FourierFields_3_y_h, FourierFields_4_x_h, FourierFields_4_y_h]

            fred = []
            fgreen = []
            fblue = []
            # J : Objective function
            # [frequency index, moniter index]
            if Norm:
                def J(fields_1_x, fields_1_y, fields_2_x, fields_2_y, fields_3_x, fields_3_y, fields_4_x, fields_4_y, fields_1_x_h, fields_1_y_h, fields_2_x_h, fields_2_y_h, fields_3_x_h, fields_3_y_h, fields_4_x_h, fields_4_y_h):
                    red_0 = -npa.sum(fields_1_x[10,:]*(npa.real(fields_1_x_h[10,:])-npa.imag(fields_1_x_h[10,:])*1j)) + npa.sum(fields_1_y[10,:]*(npa.real(fields_1_y_h[10,:])-npa.imag(fields_1_y_h[10,:])*1j))
                    red_1 = -npa.sum(fields_1_x[11,:]*(npa.real(fields_1_x_h[11,:])-npa.imag(fields_1_x_h[11,:])*1j)) + npa.sum(fields_1_y[11,:]*(npa.real(fields_1_y_h[11,:])-npa.imag(fields_1_y_h[11,:])*1j))
                    red_2 = -npa.sum(fields_1_x[12,:]*(npa.real(fields_1_x_h[12,:])-npa.imag(fields_1_x_h[12,:])*1j))  + npa.sum(fields_1_y[12,:]*(npa.real(fields_1_y_h[12,:])-npa.imag(fields_1_y_h[12,:])*1j))
                    red_3 = -npa.sum(fields_1_x[13,:]*(npa.real(fields_1_x_h[13,:])-npa.imag(fields_1_x_h[13,:])*1j)) + npa.sum(fields_1_y[13,:]*(npa.real(fields_1_y_h[13,:])-npa.imag(fields_1_y_h[13,:])*1j))
                    red_4 = -npa.sum(fields_1_x[14,:]*(npa.real(fields_1_x_h[14,:])-npa.imag(fields_1_x_h[14,:])*1j)) + npa.sum(fields_1_y[14,:]*(npa.real(fields_1_y_h[14,:])-npa.imag(fields_1_y_h[14,:])*1j))

                    CT_red_0 = -npa.sum(fields_1_x[0,:]*(npa.real(fields_1_x_h[0,:])-npa.imag(fields_1_x_h[0,:])*1j)) + npa.sum(fields_1_y[0,:]*(npa.real(fields_1_y_h[0,:])-npa.imag(fields_1_y_h[0,:])*1j))
                    CT_red_1 = -npa.sum(fields_1_x[1,:]*(npa.real(fields_1_x_h[1,:])-npa.imag(fields_1_x_h[1,:])*1j)) + npa.sum(fields_1_y[1,:]*(npa.real(fields_1_y_h[1,:])-npa.imag(fields_1_y_h[1,:])*1j))
                    CT_red_2 = -npa.sum(fields_1_x[2,:]*(npa.real(fields_1_x_h[2,:])-npa.imag(fields_1_x_h[2,:])*1j)) + npa.sum(fields_1_y[2,:]*(npa.real(fields_1_y_h[2,:])-npa.imag(fields_1_y_h[2,:])*1j))
                    CT_red_3 = -npa.sum(fields_1_x[3,:]*(npa.real(fields_1_x_h[3,:])-npa.imag(fields_1_x_h[3,:])*1j)) + npa.sum(fields_1_y[3,:]*(npa.real(fields_1_y_h[3,:])-npa.imag(fields_1_y_h[3,:])*1j))
                    CT_red_4 = -npa.sum(fields_1_x[4,:]*(npa.real(fields_1_x_h[4,:])-npa.imag(fields_1_x_h[4,:])*1j)) + npa.sum(fields_1_y[4,:]*(npa.real(fields_1_y_h[4,:])-npa.imag(fields_1_y_h[4,:])*1j))
                    CT_red_5 = -npa.sum(fields_1_x[5,:]*(npa.real(fields_1_x_h[5,:])-npa.imag(fields_1_x_h[5,:])*1j)) + npa.sum(fields_1_y[5,:]*(npa.real(fields_1_y_h[5,:])-npa.imag(fields_1_y_h[5,:])*1j))
                    CT_red_6 = -npa.sum(fields_1_x[6,:]*(npa.real(fields_1_x_h[6,:])-npa.imag(fields_1_x_h[6,:])*1j)) + npa.sum(fields_1_y[6,:]*(npa.real(fields_1_y_h[6,:])-npa.imag(fields_1_y_h[6,:])*1j))
                    CT_red_7 = -npa.sum(fields_1_x[7,:]*(npa.real(fields_1_x_h[7,:])-npa.imag(fields_1_x_h[7,:])*1j)) + npa.sum(fields_1_y[7,:]*(npa.real(fields_1_y_h[7,:])-npa.imag(fields_1_y_h[7,:])*1j))
                    CT_red_8 = -npa.sum(fields_1_x[8,:]*(npa.real(fields_1_x_h[8,:])-npa.imag(fields_1_x_h[8,:])*1j)) + npa.sum(fields_1_y[8,:]*(npa.real(fields_1_y_h[8,:])-npa.imag(fields_1_y_h[8,:])*1j))
                    CT_red_9 = -npa.sum(fields_1_x[9,:]*(npa.real(fields_1_x_h[9,:])-npa.imag(fields_1_x_h[9,:])*1j)) + npa.sum(fields_1_y[9,:]*(npa.real(fields_1_y_h[9,:])-npa.imag(fields_1_y_h[9,:])*1j))

                    green_0 = -npa.sum(fields_2_x[5,:]*(npa.real(fields_2_x_h[5,:])-npa.imag(fields_2_x_h[5,:])*1j)) + npa.sum(fields_2_y[5,:]*(npa.real(fields_2_y_h[5,:])-npa.imag(fields_2_y_h[5,:])*1j)) - npa.sum(fields_4_x[5,:]*(npa.real(fields_4_x_h[5,:])-npa.imag(fields_4_x_h[5,:])*1j)) + npa.sum(fields_4_y[5,:]*(npa.real(fields_4_y_h[5,:])-npa.imag(fields_4_y_h[5,:])*1j))
                    green_1 = -npa.sum(fields_2_x[6,:]*(npa.real(fields_2_x_h[6,:])-npa.imag(fields_2_x_h[6,:])*1j)) + npa.sum(fields_2_y[6,:]*(npa.real(fields_2_y_h[6,:])-npa.imag(fields_2_y_h[6,:])*1j)) - npa.sum(fields_4_x[6,:]*(npa.real(fields_4_x_h[6,:])-npa.imag(fields_4_x_h[6,:])*1j)) + npa.sum(fields_4_y[6,:]*(npa.real(fields_4_y_h[6,:])-npa.imag(fields_4_y_h[6,:])*1j))
                    green_2 = -npa.sum(fields_2_x[7,:]*(npa.real(fields_2_x_h[7,:])-npa.imag(fields_2_x_h[7,:])*1j)) + npa.sum(fields_2_y[7,:]*(npa.real(fields_2_y_h[7,:])-npa.imag(fields_2_y_h[7,:])*1j)) - npa.sum(fields_4_x[7,:]*(npa.real(fields_4_x_h[7,:])-npa.imag(fields_4_x_h[7,:])*1j)) + npa.sum(fields_4_y[7,:]*(npa.real(fields_4_y_h[7,:])-npa.imag(fields_4_y_h[7,:])*1j))
                    green_3 = -npa.sum(fields_2_x[8,:]*(npa.real(fields_2_x_h[8,:])-npa.imag(fields_2_x_h[8,:])*1j)) + npa.sum(fields_2_y[8,:]*(npa.real(fields_2_y_h[8,:])-npa.imag(fields_2_y_h[8,:])*1j)) - npa.sum(fields_4_x[8,:]*(npa.real(fields_4_x_h[8,:])-npa.imag(fields_4_x_h[8,:])*1j)) + npa.sum(fields_4_y[8,:]*(npa.real(fields_4_y_h[8,:])-npa.imag(fields_4_y_h[8,:])*1j))
                    green_4 = -npa.sum(fields_2_x[9,:]*(npa.real(fields_2_x_h[9,:])-npa.imag(fields_2_x_h[9,:])*1j)) + npa.sum(fields_2_y[9,:]*(npa.real(fields_2_y_h[9,:])-npa.imag(fields_2_y_h[9,:])*1j)) - npa.sum(fields_4_x[9,:]*(npa.real(fields_4_x_h[9,:])-npa.imag(fields_4_x_h[9,:])*1j)) + npa.sum(fields_4_y[9,:]*(npa.real(fields_4_y_h[9,:])-npa.imag(fields_4_y_h[9,:])*1j))

                    CT_green_0 = -npa.sum(fields_2_x[0,:]*(npa.real(fields_2_x_h[0,:])-npa.imag(fields_2_x_h[0,:])*1j)) + npa.sum(fields_2_y[0,:]*(npa.real(fields_2_y_h[0,:])-npa.imag(fields_2_y_h[0,:])*1j)) - npa.sum(fields_4_x[0,:]*(npa.real(fields_4_x_h[0,:])-npa.imag(fields_4_x_h[0,:])*1j)) + npa.sum(fields_4_y[0,:]*(npa.real(fields_4_y_h[0,:])-npa.imag(fields_4_y_h[0,:])*1j))
                    CT_green_1 = -npa.sum(fields_2_x[1,:]*(npa.real(fields_2_x_h[1,:])-npa.imag(fields_2_x_h[1,:])*1j)) + npa.sum(fields_2_y[1,:]*(npa.real(fields_2_y_h[1,:])-npa.imag(fields_2_y_h[1,:])*1j)) - npa.sum(fields_4_x[1,:]*(npa.real(fields_4_x_h[1,:])-npa.imag(fields_4_x_h[1,:])*1j)) + npa.sum(fields_4_y[1,:]*(npa.real(fields_4_y_h[1,:])-npa.imag(fields_4_y_h[1,:])*1j))
                    CT_green_2 = -npa.sum(fields_2_x[2,:]*(npa.real(fields_2_x_h[2,:])-npa.imag(fields_2_x_h[2,:])*1j)) + npa.sum(fields_2_y[2,:]*(npa.real(fields_2_y_h[2,:])-npa.imag(fields_2_y_h[2,:])*1j)) - npa.sum(fields_4_x[2,:]*(npa.real(fields_4_x_h[2,:])-npa.imag(fields_4_x_h[2,:])*1j)) + npa.sum(fields_4_y[2,:]*(npa.real(fields_4_y_h[2,:])-npa.imag(fields_4_y_h[2,:])*1j))
                    CT_green_3 = -npa.sum(fields_2_x[3,:]*(npa.real(fields_2_x_h[3,:])-npa.imag(fields_2_x_h[3,:])*1j)) + npa.sum(fields_2_y[3,:]*(npa.real(fields_2_y_h[3,:])-npa.imag(fields_2_y_h[3,:])*1j)) - npa.sum(fields_4_x[3,:]*(npa.real(fields_4_x_h[3,:])-npa.imag(fields_4_x_h[3,:])*1j)) + npa.sum(fields_4_y[3,:]*(npa.real(fields_4_y_h[3,:])-npa.imag(fields_4_y_h[3,:])*1j))
                    CT_green_4 = -npa.sum(fields_2_x[4,:]*(npa.real(fields_2_x_h[4,:])-npa.imag(fields_2_x_h[4,:])*1j)) + npa.sum(fields_2_y[4,:]*(npa.real(fields_2_y_h[4,:])-npa.imag(fields_2_y_h[4,:])*1j)) - npa.sum(fields_4_x[4,:]*(npa.real(fields_4_x_h[4,:])-npa.imag(fields_4_x_h[4,:])*1j)) + npa.sum(fields_4_y[4,:]*(npa.real(fields_4_y_h[4,:])-npa.imag(fields_4_y_h[4,:])*1j))
                    CT_green_5 = -npa.sum(fields_2_x[10,:]*(npa.real(fields_2_x_h[10,:])-npa.imag(fields_2_x_h[10,:])*1j)) + npa.sum(fields_2_y[10,:]*(npa.real(fields_2_y_h[10,:])-npa.imag(fields_2_y_h[10,:])*1j)) - npa.sum(fields_4_x[10,:]*(npa.real(fields_4_x_h[10,:])-npa.imag(fields_4_x_h[10,:])*1j)) + npa.sum(fields_4_y[10,:]*(npa.real(fields_4_y_h[10,:])-npa.imag(fields_4_y_h[10,:])*1j))
                    CT_green_6 = -npa.sum(fields_2_x[11,:]*(npa.real(fields_2_x_h[11,:])-npa.imag(fields_2_x_h[11,:])*1j)) + npa.sum(fields_2_y[11,:]*(npa.real(fields_2_y_h[11,:])-npa.imag(fields_2_y_h[11,:])*1j)) - npa.sum(fields_4_x[11,:]*(npa.real(fields_4_x_h[11,:])-npa.imag(fields_4_x_h[11,:])*1j)) + npa.sum(fields_4_y[11,:]*(npa.real(fields_4_y_h[11,:])-npa.imag(fields_4_y_h[11,:])*1j))
                    CT_green_7 = -npa.sum(fields_2_x[12,:]*(npa.real(fields_2_x_h[12,:])-npa.imag(fields_2_x_h[12,:])*1j)) + npa.sum(fields_2_y[12,:]*(npa.real(fields_2_y_h[12,:])-npa.imag(fields_2_y_h[12,:])*1j)) - npa.sum(fields_4_x[12,:]*(npa.real(fields_4_x_h[12,:])-npa.imag(fields_4_x_h[12,:])*1j)) + npa.sum(fields_4_y[12,:]*(npa.real(fields_4_y_h[12,:])-npa.imag(fields_4_y_h[12,:])*1j))
                    CT_green_8 = -npa.sum(fields_2_x[13,:]*(npa.real(fields_2_x_h[13,:])-npa.imag(fields_2_x_h[13,:])*1j)) + npa.sum(fields_2_y[13,:]*(npa.real(fields_2_y_h[13,:])-npa.imag(fields_2_y_h[13,:])*1j)) - npa.sum(fields_4_x[13,:]*(npa.real(fields_4_x_h[13,:])-npa.imag(fields_4_x_h[13,:])*1j)) + npa.sum(fields_4_y[13,:]*(npa.real(fields_4_y_h[13,:])-npa.imag(fields_4_y_h[13,:])*1j))
                    CT_green_9 = -npa.sum(fields_2_x[14,:]*(npa.real(fields_2_x_h[14,:])-npa.imag(fields_2_x_h[14,:])*1j)) + npa.sum(fields_2_y[14,:]*(npa.real(fields_2_y_h[14,:])-npa.imag(fields_2_y_h[14,:])*1j)) - npa.sum(fields_4_x[14,:]*(npa.real(fields_4_x_h[14,:])-npa.imag(fields_4_x_h[14,:])*1j)) + npa.sum(fields_4_y[14,:]*(npa.real(fields_4_y_h[14,:])-npa.imag(fields_4_y_h[14,:])*1j))

                    blue_0 = -npa.sum(fields_3_x[0,:]*(npa.real(fields_3_x_h[0,:])-npa.imag(fields_3_x_h[0,:])*1j)) + npa.sum(fields_3_y[0,:]*(npa.real(fields_3_y_h[0,:])-npa.imag(fields_3_y_h[0,:])*1j))
                    blue_1 = -npa.sum(fields_3_x[1,:]*(npa.real(fields_3_x_h[1,:])-npa.imag(fields_3_x_h[1,:])*1j)) + npa.sum(fields_3_y[1,:]*(npa.real(fields_3_y_h[1,:])-npa.imag(fields_3_y_h[1,:])*1j))
                    blue_2 = -npa.sum(fields_3_x[2,:]*(npa.real(fields_3_x_h[2,:])-npa.imag(fields_3_x_h[2,:])*1j)) + npa.sum(fields_3_y[2,:]*(npa.real(fields_3_y_h[2,:])-npa.imag(fields_3_y_h[2,:])*1j))
                    blue_3 = -npa.sum(fields_3_x[3,:]*(npa.real(fields_3_x_h[3,:])-npa.imag(fields_3_x_h[3,:])*1j)) + npa.sum(fields_3_y[3,:]*(npa.real(fields_3_y_h[3,:])-npa.imag(fields_3_y_h[3,:])*1j))
                    blue_4 = -npa.sum(fields_3_x[4,:]*(npa.real(fields_3_x_h[4,:])-npa.imag(fields_3_x_h[4,:])*1j)) + npa.sum(fields_3_y[4,:]*(npa.real(fields_3_y_h[4,:])-npa.imag(fields_3_y_h[4,:])*1j))

                    CT_blue_0 = -npa.sum(fields_3_x[5,:]*(npa.real(fields_3_x_h[5,:])-npa.imag(fields_3_x_h[5,:])*1j)) + npa.sum(fields_3_y[5,:]*(npa.real(fields_3_y_h[5,:])-npa.imag(fields_3_y_h[5,:])*1j))
                    CT_blue_1 = -npa.sum(fields_3_x[6,:]*(npa.real(fields_3_x_h[6,:])-npa.imag(fields_3_x_h[6,:])*1j)) + npa.sum(fields_3_y[6,:]*(npa.real(fields_3_y_h[6,:])-npa.imag(fields_3_y_h[6,:])*1j))
                    CT_blue_2 = -npa.sum(fields_3_x[7,:]*(npa.real(fields_3_x_h[7,:])-npa.imag(fields_3_x_h[7,:])*1j)) + npa.sum(fields_3_y[7,:]*(npa.real(fields_3_y_h[7,:])-npa.imag(fields_3_y_h[7,:])*1j))
                    CT_blue_3 = -npa.sum(fields_3_x[8,:]*(npa.real(fields_3_x_h[8,:])-npa.imag(fields_3_x_h[8,:])*1j)) + npa.sum(fields_3_y[8,:]*(npa.real(fields_3_y_h[8,:])-npa.imag(fields_3_y_h[8,:])*1j))
                    CT_blue_4 = -npa.sum(fields_3_x[9,:]*(npa.real(fields_3_x_h[9,:])-npa.imag(fields_3_x_h[9,:])*1j)) + npa.sum(fields_3_y[9,:]*(npa.real(fields_3_y_h[9,:])-npa.imag(fields_3_y_h[9,:])*1j))
                    CT_blue_5 = -npa.sum(fields_3_x[10,:]*(npa.real(fields_3_x_h[10,:])-npa.imag(fields_3_x_h[10,:])*1j)) + npa.sum(fields_3_y[10,:]*(npa.real(fields_3_y_h[10,:])-npa.imag(fields_3_y_h[10,:])*1j))
                    CT_blue_6 = -npa.sum(fields_3_x[11,:]*(npa.real(fields_3_x_h[11,:])-npa.imag(fields_3_x_h[11,:])*1j)) + npa.sum(fields_3_y[11,:]*(npa.real(fields_3_y_h[11,:])-npa.imag(fields_3_y_h[11,:])*1j))
                    CT_blue_7 = -npa.sum(fields_3_x[12,:]*(npa.real(fields_3_x_h[12,:])-npa.imag(fields_3_x_h[12,:])*1j)) + npa.sum(fields_3_y[12,:]*(npa.real(fields_3_y_h[12,:])-npa.imag(fields_3_y_h[12,:])*1j))
                    CT_blue_8 = -npa.sum(fields_3_x[13,:]*(npa.real(fields_3_x_h[13,:])-npa.imag(fields_3_x_h[13,:])*1j)) + npa.sum(fields_3_y[13,:]*(npa.real(fields_3_y_h[13,:])-npa.imag(fields_3_y_h[13,:])*1j))
                    CT_blue_9 = -npa.sum(fields_3_x[14,:]*(npa.real(fields_3_x_h[14,:])-npa.imag(fields_3_x_h[14,:])*1j)) + npa.sum(fields_3_y[14,:]*(npa.real(fields_3_y_h[14,:])-npa.imag(fields_3_y_h[14,:])*1j))


                    red = npa.real(red_0)/npa.real(r_0_dft_total) + npa.real(red_1)/npa.real(r_1_dft_total) + npa.real(red_2)/npa.real(r_2_dft_total)+ npa.real(red_3)/npa.real(r_3_dft_total) + npa.real(red_4)/npa.real(r_4_dft_total)
                    green = npa.real(green_0)/npa.real(g_0_dft_total) + npa.real(green_1)/npa.real(g_1_dft_total) + npa.real(green_2)/npa.real(g_2_dft_total) + npa.real(green_3)/npa.real(g_3_dft_total) + npa.real(green_4)/npa.real(g_4_dft_total)
                    blue = npa.real(blue_0)/npa.real(b_0_dft_total) + npa.real(blue_1)/npa.real(b_1_dft_total)  + npa.real(blue_2)/npa.real(b_2_dft_total)  + npa.real(blue_3)/npa.real(b_3_dft_total)  + npa.real(blue_4)/npa.real(b_4_dft_total) 

                    CT_red = npa.real(CT_red_0)/npa.real(b_0_dft_total) + npa.real(CT_red_1)/npa.real(b_1_dft_total) + npa.real(CT_red_2)/npa.real(b_2_dft_total) + npa.real(CT_red_3)/npa.real(b_3_dft_total) + npa.real(CT_red_4)/npa.real(b_4_dft_total) + npa.real(CT_red_5)/npa.real(g_0_dft_total) + npa.real(CT_red_6)/npa.real(g_1_dft_total) + npa.real(CT_red_7)/npa.real(g_2_dft_total) + npa.real(CT_red_8)/npa.real(g_3_dft_total) + npa.real(CT_red_9)/npa.real(g_4_dft_total)
                    CT_green = npa.real(CT_green_0)/npa.real(b_0_dft_total) + npa.real(CT_green_1)/npa.real(b_1_dft_total) + npa.real(CT_green_2)/npa.real(b_2_dft_total) + npa.real(CT_green_3)/npa.real(b_3_dft_total) + npa.real(CT_green_4)/npa.real(b_4_dft_total) + npa.real(CT_green_5)/npa.real(r_0_dft_total) + npa.real(CT_green_6)/npa.real(r_1_dft_total) + npa.real(CT_green_7)/npa.real(r_2_dft_total) + npa.real(CT_green_8)/npa.real(r_3_dft_total) + npa.real(CT_green_9)/npa.real(r_4_dft_total)
                    CT_blue = npa.real(CT_blue_0)/npa.real(g_0_dft_total) + npa.real(CT_blue_1)/npa.real(g_1_dft_total) + npa.real(CT_blue_2)/npa.real(g_2_dft_total) + npa.real(CT_blue_3)/npa.real(g_3_dft_total) + npa.real(CT_blue_4)/npa.real(g_4_dft_total) + npa.real(CT_blue_5)/npa.real(r_0_dft_total) + npa.real(CT_blue_6)/npa.real(r_1_dft_total) + npa.real(CT_blue_7)/npa.real(r_2_dft_total) + npa.real(CT_blue_8)/npa.real(r_3_dft_total) + npa.real(CT_blue_9)/npa.real(r_4_dft_total)

                    fred.append(red)
                    fgreen.append(green)
                    fblue.append(blue)

                    print("red: ", red, "green: ", green, "blue: ", blue)

                    OE = blue + green + red # Optical Efficiency
                    CT = CT_blue + CT_green + CT_red # Crosstalk
                    rate = 0

                    print(red)
                    print(green)
                    print(blue)
                    penalty = np.abs(blue - green) + np.abs(green - red) + np.abs(red - blue) # L1 norm

                    return (1-rate) * OE - rate *CT - penalty*0.5
            else :
                def J(fields_1_x, fields_1_y, fields_2_x, fields_2_y, fields_3_x, fields_3_y, fields_4_x, fields_4_y, fields_1_x_h, fields_1_y_h, fields_2_x_h, fields_2_y_h, fields_3_x_h, fields_3_y_h, fields_4_x_h, fields_4_y_h):
                    red = -npa.sum(fields_1_x[21:30,:]*(npa.real(fields_1_x_h[21:30,:])-npa.imag(fields_1_x_h[21:30,:])*1j))  + npa.sum(fields_1_y[21:30,:]*(npa.real(fields_1_y_h[21:30,:])-npa.imag(fields_1_y_h[21:30,:])*1j))
                    green = -npa.sum(fields_2_x[11:20,:]*(npa.real(fields_2_x_h[11:20,:])-npa.imag(fields_2_x_h[11:20,:])*1j)) + npa.sum(fields_2_y[11:20,:]*(npa.real(fields_2_y_h[11:20,:])-npa.imag(fields_2_y_h[11:20,:])*1j)) + npa.sum(fields_4_x[11:20,:]*(npa.real(fields_4_x_h[11:20,:])-npa.imag(fields_4_x_h[11:20,:])*1j)) + npa.sum(fields_4_y[11:20,:]*(npa.real(fields_4_y_h[11:20,:])-npa.imag(fields_4_y_h[11:20,:])*1j))
                    blue = -npa.sum(fields_3_x[1:10,:]*(npa.real(fields_3_x_h[1:10,:])-npa.imag(fields_3_x_h[1:10,:])*1j)) + npa.sum(fields_3_y[1:10,:]*(npa.real(fields_3_y_h[1:10,:])-npa.imag(fields_3_y_h[1:10,:])*1j))
                
                    CT_red = -npa.sum(fields_1_x[1:10,:]*(npa.real(fields_1_x_h[1:10,:])-npa.imag(fields_1_x_h[1:10,:])*1j))  + npa.sum(fields_1_y[1:10,:]*(npa.real(fields_1_y_h[1:10,:])-npa.imag(fields_1_y_h[1:10,:])*1j)) -npa.sum(fields_1_x[11:20,:]*(npa.real(fields_1_x_h[11:20,:])-npa.imag(fields_1_x_h[11:20,:])*1j))  + npa.sum(fields_1_y[11:20,:]*(npa.real(fields_1_y_h[11:20,:])-npa.imag(fields_1_y_h[11:20,:])*1j))
                    CT_green = -npa.sum(fields_2_x[1:10,:]*(npa.real(fields_2_x_h[1:10,:])-npa.imag(fields_2_x_h[1:10,:])*1j)) + npa.sum(fields_2_y[1:10,:]*(npa.real(fields_2_y_h[1:10,:])-npa.imag(fields_2_y_h[1:10,:])*1j)) + npa.sum(fields_4_x[1:10,:]*(npa.real(fields_4_x_h[1:10,:])-npa.imag(fields_4_x_h[1:10,:])*1j)) + npa.sum(fields_4_y[1:10,:]*(npa.real(fields_4_y_h[1:10,:])-npa.imag(fields_4_y_h[1:10,:])*1j)) 
                    -npa.sum(fields_2_x[21:30,:]*(npa.real(fields_2_x_h[21:30,:])-npa.imag(fields_2_x_h[21:30,:])*1j)) + npa.sum(fields_2_y[21:30,:]*(npa.real(fields_2_y_h[21:30,:])-npa.imag(fields_2_y_h[21:30,:])*1j)) + npa.sum(fields_4_x[21:30,:]*(npa.real(fields_4_x_h[21:30,:])-npa.imag(fields_4_x_h[21:30,:])*1j)) + npa.sum(fields_4_y[21:30,:]*(npa.real(fields_4_y_h[21:30,:])-npa.imag(fields_4_y_h[21:30,:])*1j))
                    CT_blue = -npa.sum(fields_3_x[11:20,:]*(npa.real(fields_3_x_h[11:20,:])-npa.imag(fields_3_x_h[11:20,:])*1j)) + npa.sum(fields_3_y[11:20,:]*(npa.real(fields_3_y_h[11:20,:])-npa.imag(fields_3_y_h[11:20,:])*1j)) -npa.sum(fields_3_x[21:30,:]*(npa.real(fields_3_x_h[21:30,:])-npa.imag(fields_3_x_h[21:30,:])*1j)) + npa.sum(fields_3_y[21:30,:]*(npa.real(fields_3_y_h[21:30,:])-npa.imag(fields_3_y_h[21:30,:])*1j))
                    
                    fred.append(npa.real(red))
                    fgreen.append(npa.real(green))
                    fblue.append(npa.real(blue))
                    OE = npa.real(blue + green + red)
                    CT = npa.real(CT_blue + CT_green + CT_red)
                    rate = 0

                    print(red)
                    print(green)
                    print(blue)

                    return (1-rate) * OE - rate *CT
                
        # optimization 설정
        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=ob_list,
            design_regions=[design_region_0, design_region_1, design_region_2, design_region_3, design_region_4],
            frequencies=[frequencies[1],frequencies[2],frequencies[3],frequencies[4],frequencies[5],frequencies[10],frequencies[11],frequencies[12],frequencies[13],frequencies[14],frequencies[18],frequencies[19],frequencies[20],frequencies[21],frequencies[22]],   
            decay_by=1e-3,
            # maximum_run_time=300,
        )       
    ###############################################################################################################################
    if designplot:
        # ## 3. Design plot
        opt.plot2D(False, output_plane = mp.Volume(size = (Sx, 0, design_region_height), center = (0,0,round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height/2, 2))),
                source_parameters={'alpha':0},
                    boundary_parameters={'alpha':0},
                )
        plt.axis("off")
        plt.savefig("./"+directory_name+"/Design/Lastdesignxz.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        opt.plot2D(False, output_plane = mp.Volume(size = (0, Sy, design_region_height), center = (0,0,round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height/2, 2))),
                source_parameters={'alpha':0},
                    boundary_parameters={'alpha':0},
                )
        # plt.axis("off")
        plt.xlabel("Width (μm)")
        plt.ylabel("Height (μm)")
        plt.savefig("./"+directory_name+"/Design/Lastdesignyz.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        opt.plot2D(True, output_plane = mp.Volume(size = (Sx, Sy, 0), center = (0, 0,round(-Sz/2 + Lpml + mon_2_pml + design_region_height/2,2))),
                source_parameters={'alpha':0},
                    boundary_parameters={'alpha':0},
                    monitor_parameters={'alpha':0},
                )
        plt.xlabel("x (μm)")
        plt.ylabel("y (μm)")
        plt.savefig("./"+directory_name+"/Design/Lastdesignxy.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure 

        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)),
                    source_parameters={'alpha':0},
                    # boundary_parameters={'alpha':0},
                    monitor_parameters={'alpha':0},
                )
        # plt.axis("off")
        plt.xlabel("Width (μm)")
        plt.ylabel("Height (μm)")
        plt.savefig("./"+directory_name+"/Design/Fullshot.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
    ###############################################################################################################################
    if DFTfield:
        width = 0.4

        fcen_red = 1/(0.625*um_scale)
        fwidth_red = fcen_red * width

        fcen_green = 1/(0.545*um_scale)
        fwidth_green = fcen_green * width

        fcen_blue = 1/(0.455*um_scale)
        fwidth_blue = fcen_blue * width

        # blue pixel
        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        src = mp.GaussianSource(frequency=frequencies[3], fwidth=fwidth_blue, is_integrated=True)

        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                      
        opt.sim.change_sources(source)

        plt.figure()

        tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

        tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[3], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

        Ex_Namei="Ex_b_field"
        Ey_Namei="Ey_b_field"
        Ez_Namei="Ez_b_field"

        Hx_Namei="Hx_b_field"
        Hy_Namei="Hy_b_field"
        Hz_Namei="Hz_b_field"

        pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)) #pt는 transmitted flux region과 동일

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        opt.sim.output_dft(tran_Ex,"./"+directory_name+"/DFTfield/"+Ex_Namei)
        opt.sim.output_dft(tran_Ey,"./"+directory_name+"/DFTfield/"+Ey_Namei)
        opt.sim.output_dft(tran_Ez,"./"+directory_name+"/DFTfield/"+Ez_Namei)

        opt.sim.output_dft(tran_Hx,"./"+directory_name+"/DFTfield/"+Hx_Namei)
        opt.sim.output_dft(tran_Hy,"./"+directory_name+"/DFTfield/"+Hy_Namei)
        opt.sim.output_dft(tran_Hz,"./"+directory_name+"/DFTfield/"+Hz_Namei)


        # green pixel
        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        src = mp.GaussianSource(frequency=frequencies[12], fwidth=fwidth_green, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
        opt.sim.change_sources(source)

        plt.figure()

        tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

        tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[12], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

        Ex_Namei="Ex_g_field"
        Ey_Namei="Ey_g_field"
        Ez_Namei="Ez_g_field"

        Hx_Namei="Hx_g_field"
        Hy_Namei="Hy_g_field"
        Hz_Namei="Hz_g_field"

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        opt.sim.output_dft(tran_Ex,"./"+directory_name+"/DFTfield/"+Ex_Namei)
        opt.sim.output_dft(tran_Ey,"./"+directory_name+"/DFTfield/"+Ey_Namei)
        opt.sim.output_dft(tran_Ez,"./"+directory_name+"/DFTfield/"+Ez_Namei)

        opt.sim.output_dft(tran_Hx,"./"+directory_name+"/DFTfield/"+Hx_Namei)
        opt.sim.output_dft(tran_Hy,"./"+directory_name+"/DFTfield/"+Hy_Namei)
        opt.sim.output_dft(tran_Hz,"./"+directory_name+"/DFTfield/"+Hz_Namei)


        # red pixel
        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        src = mp.GaussianSource(frequency=frequencies[20], fwidth=fwidth_red, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
        opt.sim.change_sources(source)

        plt.figure()

        tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

        tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
        tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[20], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)


        Ex_Namei="Ex_r_field"
        Ey_Namei="Ey_r_field"
        Ez_Namei="Ez_r_field"

        Hx_Namei="Hx_r_field"
        Hy_Namei="Hy_r_field"
        Hz_Namei="Hz_r_field"

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        opt.sim.output_dft(tran_Ex,"./"+directory_name+"/DFTfield/"+Ex_Namei)
        opt.sim.output_dft(tran_Ey,"./"+directory_name+"/DFTfield/"+Ey_Namei)
        opt.sim.output_dft(tran_Ez,"./"+directory_name+"/DFTfield/"+Ez_Namei)

        opt.sim.output_dft(tran_Hx,"./"+directory_name+"/DFTfield/"+Hx_Namei)
        opt.sim.output_dft(tran_Hy,"./"+directory_name+"/DFTfield/"+Hy_Namei)
        opt.sim.output_dft(tran_Hz,"./"+directory_name+"/DFTfield/"+Hz_Namei)
    ###############################################################################################################################
    if Opticaleff:
        # ## 5. Optical Efficiency
        opt.sim.reset_meep()

        # simulation 1 : geometry가 없는 구조
        geometry_1 = [
            mp.Block(
                center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), material=Air
            )
        ]

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry_1,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )
        
        width = 2
        frequency = 1/(0.550*um_scale)
        fwidth = frequency * width
        fcen = (1/(0.30 * um_scale) + 1/(0.80 * um_scale))/2
        df = 1 /(0.30 * um_scale) - 1/(0.80 * um_scale)
        nfreq = 300

        src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5),mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        opt.sim.change_sources(source)


        # reflection moniter 설정

        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0,  round(Sz/2 - Lpml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        ) 
        refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

        # transmission moiniter 설정

        tran_t = mp.FluxRegion(
            center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        )
        tran_total = opt.sim.add_flux(fcen, df, nfreq, tran_t)


        # pt는 transmitted flux region과 동일

        pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2))

        #source가 끝난 후에 50 동안 계속 실행하며 component는 Ey, pt 설계의 끝에서 |Ey|^2의 값이 최대 값으로부터 1/1000 만큼 감쇠할때까지
        #추가적인 50 단위의 시간 실행 -> Fourier-transform 수렴예상

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        #데이터 저장

        straight_refl_data = opt.sim.get_flux_data(refl)
        total_flux = mp.get_fluxes(tran_total)
        flux_freqs = mp.get_flux_freqs(tran_total)


        opt.sim.reset_meep()

        # simulation 2 : geometry가 있는 구조

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )


        # 반사된 flux 구하기

        refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

        # 투과된 flux 구하기

        tran = opt.sim.add_flux(fcen, df, nfreq, tran_t)

        # 픽셀에 들어온 flux 구하기

        tran_p = mp.FluxRegion(
            center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0)
        )

        tran_pixel = opt.sim.add_flux(fcen, df, nfreq, tran_p)

        #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
        #the Fourier transforms of the scattered fields 에서 빼줍니다.

        opt.sim.load_minus_flux_data(refl, straight_refl_data)

        #각각 픽셀의 flux 구하기

        tran_r = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_gr = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_b = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_gb = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_red = opt.sim.add_flux(fcen, df, nfreq, tran_r)
        tran_greenr = opt.sim.add_flux(fcen, df, nfreq, tran_gr)
        tran_blue = opt.sim.add_flux(fcen, df, nfreq, tran_b)
        tran_greenb = opt.sim.add_flux(fcen, df, nfreq, tran_gb)

        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)))
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        # 데이터 저장

        refl_flux = mp.get_fluxes(refl)
        tran_flux = mp.get_fluxes(tran)
        tran_flux_p = mp.get_fluxes(tran_pixel)

        red_flux = mp.get_fluxes(tran_red)
        greenr_flux = mp.get_fluxes(tran_greenr)
        blue_flux = mp.get_fluxes(tran_blue)
        greenb_flux = mp.get_fluxes(tran_greenb)

        # totalflux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(nfreq):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] / total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr, "r", label="R")
            plt.plot(wl, Tgr, "g", label="Gb")
            plt.plot(wl, Tb, "b", label="B")
            plt.plot(wl, Tgb, color='limegreen', label="Gr")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.400, 0.400, 0.500, 0.500], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.500, 0.500, 0.600, 0.600], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.600, 0.600, 0.700, 0.700], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            plt.savefig("./"+directory_name+"/QE.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tg = []
        Tb = []
        Tg0 = []

        Trt = []
        Tgt = []
        Tbt = []
        Tg0t = []
        DTIloss=[]
        for d in range(nfreq):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / tran_flux_p[d])
            Tg = np.append(Tg, greenr_flux[d] /tran_flux_p[d])
            Tb = np.append(Tb, blue_flux[d] / tran_flux_p[d])
            Tg0 = np.append(Tg0, greenb_flux[d] / tran_flux_p[d])

            Trt = np.append(Trt, red_flux[d] / total_flux[d])
            Tgt = np.append(Tgt, greenr_flux[d] / total_flux[d])
            Tbt = np.append(Tbt, blue_flux[d] / total_flux[d])
            Tg0t = np.append(Tg0t, greenb_flux[d] / total_flux[d])

            DTIloss = np.append(DTIloss, (tran_flux_p[d] - (red_flux[d] + greenr_flux[d] + blue_flux[d] + greenb_flux[d]))/tran_flux_p[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr, "r",)
            plt.plot(wl, (Tg+Tg0), "g",)
            plt.plot(wl, Tb, "b",)
            plt.plot(wl, Trt, "r--",)
            plt.plot(wl, (Tgt+Tg0t), "g--",)
            plt.plot(wl, Tbt, "b--",)
            plt.plot(wl, DTIloss, "k",)
            #plt.plot(wl, Tg0, color='limegreen', label="Greenpixel2")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.400, 0.400, 0.500, 0.500], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.500, 0.500, 0.600, 0.600], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.600, 0.600, 0.700, 0.700], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.tick_params(axis='x', direction='in', pad = 8)
            plt.tick_params(axis='y', direction='in', pad = 10)
            #plt.show()
            plt.savefig("./"+directory_name+"/Optical Efficiency.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure



        wl = []
        Tr = []
        Tg = []
        Tb = []
        Tg0 = []

        Trt = []
        Tgt = []
        Tbt = []
        Tg0t = []
        DTIloss=[]
        for d in range(nfreq):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / tran_flux_p[d])
            Tg = np.append(Tg, greenr_flux[d] /tran_flux_p[d])
            Tb = np.append(Tb, blue_flux[d] / tran_flux_p[d])
            Tg0 = np.append(Tg0, greenb_flux[d] / tran_flux_p[d])

            Trt = np.append(Trt, red_flux[d] / total_flux[d])
            Tgt = np.append(Tgt, greenr_flux[d] / total_flux[d])
            Tbt = np.append(Tbt, blue_flux[d] / total_flux[d])
            Tg0t = np.append(Tg0t, greenb_flux[d] / total_flux[d])

            DTIloss = np.append(DTIloss, (tran_flux_p[d] - (red_flux[d] + greenr_flux[d] + blue_flux[d] + greenb_flux[d]))/tran_flux_p[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr, "r",)
            plt.plot(wl, Tg, "g",)
            plt.plot(wl, Tb, "b",)
            plt.plot(wl, Trt, "r--",)
            plt.plot(wl, Tgt, "g--",)
            plt.plot(wl, Tbt, "b--",)
            plt.plot(wl, DTIloss, "k",)
            #plt.plot(wl, Tg0, color='limegreen', label="Greenpixel2")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.400, 0.400, 0.500, 0.500], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.500, 0.500, 0.600, 0.600], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.600, 0.600, 0.700, 0.700], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.tick_params(axis='x', direction='in', pad = 8)
            plt.tick_params(axis='y', direction='in', pad = 10)
            #plt.show()
            plt.savefig("./"+directory_name+"/Green Optical Efficiency.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure



        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(nfreq):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] / total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr, "r", label="R")
            plt.plot(wl, (Tgr+Tgb), "g", label="Gb")
            plt.plot(wl, Tb, "b", label="B")
            # plt.plot(wl, Tgb, color='limegreen', label="Gr")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.400, 0.400, 0.500, 0.500], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.500, 0.500, 0.600, 0.600], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.600, 0.600, 0.700, 0.700], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            plt.savefig("./"+directory_name+"/GQE.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        np.savetxt("./"+directory_name+"/QE_data/R_raw_data.txt" , Tr* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gr_raw_data.txt" , Tgr* 400)
        np.savetxt("./"+directory_name+"/QE_data/B_raw_data.txt" , Tb* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gb_raw_data.txt" , Tgb* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gavg_raw_data.txt" , (Tgb+Tgr)* 200)



        # 투과율과 반사율

        Rs = []
        Ts = []
        for d in range(nfreq):
            Rs = np.append(Rs, -refl_flux[d] / total_flux[d])
            Ts = np.append(Ts, tran_flux[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Rs, "b", label="reflectance")
            plt.plot(wl, Ts, "r", label="transmittance")
            plt.plot(wl, 1 - Rs - Ts, "g", label="loss")
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Transmittance")
            plt.fill([0.40, 0.40, 0.50, 0.50], [0.0, 1.0, 1.0, 0.0], color='lightblue', alpha=0.5)
            plt.fill([0.50, 0.50, 0.60, 0.60], [0.0, 1.0, 1.0, 0.0], color='lightgreen', alpha=0.5)
            plt.fill([0.60, 0.60, 0.70, 0.70], [0.0, 1.0, 1.0, 0.0], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/QE_data/T_R.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure
    ###############################################################################################################################
    if layerplot:

        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        from matplotlib.colors import LinearSegmentedColormap
        gray_color = '#b9b9b9'
        black_color = '#404040'

        # Create a custom colormap from 'gray' to 'black'
        cmap = LinearSegmentedColormap.from_list('gray_to_black', [gray_color, black_color])


        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        plt.imshow(npa.rot90((structure_weight_0.reshape(Nx,Ny))), cmap=cmap)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.savefig("./"+directory_name+"/Design/design_layer_1.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.imshow(npa.rot90((structure_weight_1.reshape(Nx,Ny))), cmap=cmap)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.savefig("./"+directory_name+"/Design/design_layer_2.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        plt.imshow(npa.rot90((structure_weight_2.reshape(Nx,Ny))), cmap=cmap)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.savefig("./"+directory_name+"/Design/design_layer_3.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        plt.imshow(npa.rot90((structure_weight_3.reshape(Nx,Ny))), cmap=cmap)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.savefig("./"+directory_name+"/Design/design_layer_4.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        plt.imshow(npa.rot90((structure_weight_4.reshape(Nx,Ny))), cmap=cmap)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.savefig("./"+directory_name+"/Design/design_layer_5.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
    ###############################################################################################################################
    if Sensitivity:
        wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
        frequencies = 1/wavelengths
        nf = len(frequencies) # number of frequencies

        opt.sim.reset_meep()

        # simulation 1 : geometry가 없는 구조
        geometry_1 = [
            mp.Block(
                center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), material=Air
            )
        ]

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry_1,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )
        width = 2
        frequency = 1/(0.550*um_scale)
        fwidth = frequency * width
        fcen = (1/(0.30 * um_scale) + 1/(0.80 * um_scale))/2
        df = 1 /(0.30 * um_scale) - 1/(0.80 * um_scale)
        nfreq = 300

        src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5),mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        opt.sim.change_sources(source)

        # reflection moniter 설정
        refl = []
        tran_total = []
        straight_refl_data = []
        total_flux = []
        flux_freqs = []

        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0,  round(Sz/2 - Lpml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        ) 

        #transmission moiniter 설정

        tran_t = mp.FluxRegion(
            center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        )


        for d in range(31):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran_total = np.append(tran_total, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))


        #pt는 transmitted flux region과 동일

        pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2))

        #source가 끝난 후에 50 동안 계속 실행하며 component는 Ey, pt 설계의 끝에서 |Ey|^2의 값이 최대 값으로부터 1/1000 만큼 감쇠할때까지
        #추가적인 50 단위의 시간 실행 -> Fourier-transform 수렴예상

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        #데이터 저장


        for d in range(31):
            straight_refl_data =  np.append(straight_refl_data,mp.get_fluxes(refl[d]))
            total_flux = np.append(total_flux,mp.get_fluxes(tran_total[d]))
            flux_freqs = np.append(flux_freqs,mp.get_flux_freqs(tran_total[d]))

        opt.sim.reset_meep()

        # simulation 2 : geometry가 있는 구조

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        refl = []
        tran = []
        for d in range(31):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran = np.append(tran, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))

        #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
        #the Fourier transforms of the scattered fields 에서 빼줍니다.

        #각각 픽셀의 flux 구하기

        tran_gr = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_b = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_gb = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_r = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_red = []
        tran_greenr = []
        tran_blue = []
        tran_greenb = []


        for d in range(31):
            tran_red = np.append(tran_red, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_r))
            tran_greenr = np.append(tran_greenr, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gr))
            tran_blue = np.append(tran_blue, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_b))
            tran_greenb = np.append(tran_greenb, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gb))

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        # 데이터 저장

        sc_refl_flux = []
        tran_flux = []


        red_flux = []
        greenr_flux = []
        blue_flux = []
        greenb_flux = []

        for d in range(31):
            sc_refl_flux = np.append(sc_refl_flux,mp.get_fluxes(refl[d]))
            tran_flux = np.append(tran_flux,mp.get_fluxes(tran[d]))

            red_flux = np.append(red_flux,mp.get_fluxes(tran_red[d]))
            greenr_flux = np.append(greenr_flux,mp.get_fluxes(tran_greenr[d]))
            blue_flux = np.append(blue_flux,mp.get_fluxes(tran_blue[d]))
            greenb_flux = np.append(greenb_flux ,mp.get_fluxes(tran_greenb[d]))
            
        refl_flux= sc_refl_flux-straight_refl_data
        weight = [70.16, 76.10, 83.32, 90.69, 94.79, 102.17, 106.41, 114.93, 117.55, 125.90, 129.97, 133.48, 139.60, 141.13, 147.02, 151.21, 152.96, 156.20, 160.38, 161.84, 163.03, 167.11, 168.03, 168.02, 158.82, 87.90, 29.08, 10.24, 4.04, 2.03, 1.11]

        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(31):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] /total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr*400, "r", label="R")
            plt.plot(wl, Tgr*400, "g", label="Gr")
            plt.plot(wl, Tb*400, "b", label="B")
            plt.plot(wl, Tgb*400, color='limegreen', label="Gb")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.400, 0.400, 0.500, 0.500], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.500, 0.500, 0.600, 0.600], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.600, 0.600, 0.700, 0.700], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/QE_data/D_OE.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        np.savetxt("./"+directory_name+"/QE_data/R_discrete_data.txt" , Tr* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gr_discrete_data.txt" , Tgr* 400)
        np.savetxt("./"+directory_name+"/QE_data/B_discrete_data.txt" , Tb* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gb_discrete_data.txt" , Tgb* 400)
        np.savetxt("./"+directory_name+"/QE_data/Gavg_discrete_data.txt" , (Tgb+Tgr)* 200)


        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(31):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] * weight[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d]* weight[d] /total_flux[d])
            Tb = np.append(Tb, blue_flux[d]* weight[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d]* weight[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr* 4, "r", label="R")
            plt.plot(wl, Tgr* 4, "g", label="Gr")
            plt.plot(wl, Tb* 4, "b", label="B")
            plt.plot(wl, Tgb* 4, color='limegreen', label="Gb")
            
            # plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Sensitivity")
            # plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            # plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            # plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Sensitivity_data/Sensitivity.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure
        
        np.savetxt("./"+directory_name+"/Sensitivity_data/R_Sensitivity_data.txt" , Tr* 4)
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gr_Sensitivity_data.txt" , Tgr* 4)
        np.savetxt("./"+directory_name+"/Sensitivity_data/B_Sensitivity_data.txt" , Tb* 4)
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gb_Sensitivity_data.txt" , Tgb* 4)
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gavg_Sensitivity_data.txt" , (Tgb+Tgr)* 2)

        np.savetxt("./"+directory_name+"/Sensitivity_data/R_Sensitivity_sum_data.txt" , np.array([np.sum(Tr* 4)]))
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gr_Sensitivity_sum_data.txt" , np.array([np.sum(Tgr* 4)]))
        np.savetxt("./"+directory_name+"/Sensitivity_data/B_Sensitivity_sum_data.txt" , np.array([np.sum(Tb* 4)]))
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gb_Sensitivity_sum_data.txt" , np.array([np.sum(Tgb* 4)]))
        np.savetxt("./"+directory_name+"/Sensitivity_data/Gavg_Sensitivity_sum_data.txt" , np.array([np.sum((Tgb+Tgr)* 2)]))        
        np.savetxt("./"+directory_name+"/Sensitivity_data/GG_Difference.txt" , 100 * (Tgr-Tgb)/((Tgr+Tgb)/2))
    ###############################################################################################################################
    if Crosstalk:
        grid = 5
        wavelengths = np.linspace(0.425*um_scale, 0.655*um_scale, 24*grid) 
        frequencies = 1/wavelengths
        nf = len(frequencies) # number of frequencies

        opt.sim.reset_meep()

        # simulation 1 : geometry가 없는 구조
        geometry_1 = [
            mp.Block(
                center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), material=Air
            )
        ]

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry_1,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )
        width = 2
        frequency = 1/(0.550*um_scale)
        fwidth = frequency * width
        fcen = (1/(0.30 * um_scale) + 1/(0.80 * um_scale))/2
        df = 1 /(0.30 * um_scale) - 1/(0.80 * um_scale)
        nfreq = 300

        src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5),mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        opt.sim.change_sources(source)

        #reflection moniter 설정
        refl = []
        tran_total = []
        straight_refl_data = []
        total_flux = []
        flux_freqs = []

        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0,  round(Sz/2 - Lpml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        ) 

        #transmission moiniter 설정

        tran_t = mp.FluxRegion(
            center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        )


        for d in range(24*grid):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran_total = np.append(tran_total, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))

        #pt는 transmitted flux region과 동일

        pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2))

        #source가 끝난 후에 50 동안 계속 실행하며 component는 Ey, pt 설계의 끝에서 |Ey|^2의 값이 최대 값으로부터 1/1000 만큼 감쇠할때까지
        #추가적인 50 단위의 시간 실행 -> Fourier-transform 수렴예상

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        #데이터 저장


        for d in range(24*grid):
            straight_refl_data =  np.append(straight_refl_data,mp.get_fluxes(refl[d]))
            total_flux = np.append(total_flux,mp.get_fluxes(tran_total[d]))
            flux_freqs = np.append(flux_freqs,mp.get_flux_freqs(tran_total[d]))

        opt.sim.reset_meep()

        # simulation 2 : geometry가 있는 구조

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        refl = []
        tran = []
        for d in range(24*grid):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran = np.append(tran, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))

        #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
        #the Fourier transforms of the scattered fields 에서 빼줍니다.

        #각각 픽셀의 flux 구하기

        tran_gr = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_b = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_gb = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_r = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_red = []
        tran_greenr = []
        tran_blue = []
        tran_greenb = []


        for d in range(24*grid):
            tran_red = np.append(tran_red, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_r))
            tran_greenr = np.append(tran_greenr, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gr))
            tran_blue = np.append(tran_blue, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_b))
            tran_greenb = np.append(tran_greenb, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gb))

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        # 데이터 저장

        sc_refl_flux = []
        tran_flux = []
        tran_flux_p = [] 

        red_flux = []
        greenr_flux = []
        blue_flux = []
        greenb_flux = []

        for d in range(24*grid):
            sc_refl_flux = np.append(sc_refl_flux,mp.get_fluxes(refl[d]))
            tran_flux = np.append(tran_flux,mp.get_fluxes(tran[d]))

            red_flux = np.append(red_flux,mp.get_fluxes(tran_red[d]))
            greenr_flux = np.append(greenr_flux,mp.get_fluxes(tran_greenr[d]))
            blue_flux = np.append(blue_flux,mp.get_fluxes(tran_blue[d]))
            greenb_flux = np.append(greenb_flux ,mp.get_fluxes(tran_greenb[d]))
            
        refl_flux= sc_refl_flux-straight_refl_data

        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(24*grid):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] /total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])
        Ttg = Tgr+Tgb

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr, "r", label="Redpixel")
            plt.plot(wl, Ttg, "g", label="Greenpixel")
            plt.plot(wl, Tb, "b", label="Bluepixel")
            #plt.plot(wl, Tgb, color='limegreen', label="Greenpixel2")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Crosstalk Efficiency")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Crosstalk/crosstalk.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure
            
        g_b = np.sum(Ttg[0:6*grid])/np.sum(Tb[0:6*grid])
        b_g = np.sum(Tb[9*grid:15*grid])/np.sum(Ttg[9*grid:15*grid])
        r_b = np.sum(Tr[0:6*grid])/np.sum(Tb[0:6*grid])
        b_r = np.sum(Tb[17*grid:23*grid])/np.sum(Tr[17*grid:23*grid])
        r_g = np.sum(Tr[9*grid:15*grid])/np.sum(Ttg[9*grid:15*grid])
        g_r = np.sum(Ttg[17*grid:23*grid])/np.sum(Tr[17*grid:23*grid])

        crosstalk = [g_b, b_g, r_b, b_r, r_g, g_r]
        crosstalk_avg = np.mean([g_b, b_g, r_b, b_r, r_g, g_r]) * 100
        np.savetxt("./"+directory_name+"/Crosstalk/rgbcrosstalk_avg.txt" , np.array([crosstalk_avg]))
        np.savetxt("./"+directory_name+"/Crosstalk/rgbcrosstalk.txt" , np.array([crosstalk]))
        
        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(24*grid):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] /total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])
        Ttg = Tgr+Tgb

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tb/Tgb, "b", label="b_gb")
            plt.plot(wl, Tr/Tgb, "r", label="r_gb")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Crosstalk")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Crosstalk/Gb.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tb/Tgr, "b", label="b_gr")
            plt.plot(wl, Tr/Tgr, "r", label="r_gr")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Crosstalk")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Crosstalk/Gbr.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure
            
        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tb/Tr, "b", label="b_r")
            plt.plot(wl, Tgr/Tr, "g", label="gr_r")
            plt.plot(wl, Tgb/Tr, "g--", label="gb_r")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Crosstalk")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Crosstalk/R.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr/Tb, "r", label="b_b")
            plt.plot(wl, Tgr/Tb, "g", label="gr_b")
            plt.plot(wl, Tgb/Tb, "g--", label="gb_b")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Crosstalk")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Crosstalk/B.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        gb_b = np.sum(Tgb[0:6*grid])/np.sum(Tb[0:6*grid])
        b_gb = np.sum(Tb[9*grid:15*grid])/np.sum(Tgb[9*grid:15*grid])
        gr_b = np.sum(Tgr[0:6*grid])/np.sum(Tb[0:6*grid])
        b_gr = np.sum(Tb[9*grid:15*grid])/np.sum(Tgr[9*grid:15*grid])
        r_b = np.sum(Tr[0:6*grid])/np.sum(Tb[0:6*grid])
        b_r = np.sum(Tb[17*grid:23*grid])/np.sum(Tr[17*grid:23*grid])
        r_gb = np.sum(Tr[9*grid:15*grid])/np.sum(Tgb[9*grid:15*grid])
        gb_r = np.sum(Tgb[17*grid:23*grid])/np.sum(Tr[17*grid:23*grid])
        r_gr = np.sum(Tr[9*grid:15*grid])/np.sum(Tgr[9*grid:15*grid])
        gr_r = np.sum(Tgr[17*grid:23*grid])/np.sum(Tr[17*grid:23*grid])

        ggcrosstalk = [gb_b, b_gb, gr_b, b_gr, r_b, b_r, r_gb, gb_r, r_gr, gr_r]
        ggcrosstalk_avg = np.mean([gb_b, b_gb, gr_b, b_gr, r_b, b_r, r_gb, gb_r, r_gr, gr_r]) * 100 
        np.savetxt("./"+directory_name+"/Crosstalk/ggcrosstalk_avg.txt" , np.array([ggcrosstalk_avg]))
        np.savetxt("./"+directory_name+"/Crosstalk/gcrosstalk.txt" , np.array([ggcrosstalk]))
    ###############################################################################################################################
    if Discrete:
        wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 61) 
        frequencies = 1/wavelengths
        nf = len(frequencies) # number of frequencies

        opt.sim.reset_meep()

        # simulation 1 : geometry가 없는 구조
        geometry_1 = [
            mp.Block(
                center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), material=Air
            )
        ]

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry_1,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )
        width = 2
        frequency = 1/(0.550*um_scale)
        fwidth = frequency * width
        fcen = (1/(0.30 * um_scale) + 1/(0.80 * um_scale))/2
        df = 1 /(0.30 * um_scale) - 1/(0.80 * um_scale)
        nfreq = 300

        src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
        if i == 0:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5)]
        elif i == 1:
            source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        elif i == 2:
            source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude=5),mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude=5)]
        opt.sim.change_sources(source)

        # reflection moniter 설정
        refl = []
        tran_total = []
        straight_refl_data = []
        total_flux = []
        flux_freqs = []

        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0,  round(Sz/2 - Lpml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        ) 

        #transmission moiniter 설정

        tran_t = mp.FluxRegion(
            center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0)
        )


        for d in range(61):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran_total = np.append(tran_total, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))


        #pt는 transmitted flux region과 동일

        pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2))

        #source가 끝난 후에 50 동안 계속 실행하며 component는 Ey, pt 설계의 끝에서 |Ey|^2의 값이 최대 값으로부터 1/1000 만큼 감쇠할때까지
        #추가적인 50 단위의 시간 실행 -> Fourier-transform 수렴예상

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        #데이터 저장


        for d in range(61):
            straight_refl_data =  np.append(straight_refl_data,mp.get_fluxes(refl[d]))
            total_flux = np.append(total_flux,mp.get_fluxes(tran_total[d]))
            flux_freqs = np.append(flux_freqs,mp.get_flux_freqs(tran_total[d]))

        opt.sim.reset_meep()

        # simulation 2 : geometry가 있는 구조

        opt.sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            default_material=Air,
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        refl = []
        tran = []
        for d in range(61):
            refl = np.append(refl, opt.sim.add_flux(1/wavelengths[d], 0, 1, refl_fr))
            tran = np.append(tran, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_t))

        #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
        #the Fourier transforms of the scattered fields 에서 빼줍니다.

        #각각 픽셀의 flux 구하기

        tran_gr = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_b = mp.FluxRegion(
            center=mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_gb = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, -design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )
        tran_r = mp.FluxRegion(
            center=mp.Vector3(design_region_width_x/4, design_region_width_y/4, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2)), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
        )

        tran_red = []
        tran_greenr = []
        tran_blue = []
        tran_greenb = []


        for d in range(61):
            tran_red = np.append(tran_red, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_r))
            tran_greenr = np.append(tran_greenr, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gr))
            tran_blue = np.append(tran_blue, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_b))
            tran_greenb = np.append(tran_greenb, opt.sim.add_flux(1/wavelengths[d], 0, 1, tran_gb))

        opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))

        # 데이터 저장

        sc_refl_flux = []
        tran_flux = []


        red_flux = []
        greenr_flux = []
        blue_flux = []
        greenb_flux = []

        for d in range(61):
            sc_refl_flux = np.append(sc_refl_flux,mp.get_fluxes(refl[d]))
            tran_flux = np.append(tran_flux,mp.get_fluxes(tran[d]))

            red_flux = np.append(red_flux,mp.get_fluxes(tran_red[d]))
            greenr_flux = np.append(greenr_flux,mp.get_fluxes(tran_greenr[d]))
            blue_flux = np.append(blue_flux,mp.get_fluxes(tran_blue[d]))
            greenb_flux = np.append(greenb_flux ,mp.get_fluxes(tran_greenb[d]))
            
        refl_flux= sc_refl_flux-straight_refl_data

        # 입사 flux 대비 효율

        wl = []
        Tr = []
        Tgr = []
        Tb = []
        Tgb = []
        for d in range(61):
            wl = np.append(wl, 1 / (flux_freqs[d] * um_scale))
            Tr = np.append(Tr, red_flux[d] / total_flux[d])
            Tgr = np.append(Tgr, greenr_flux[d] /total_flux[d])
            Tb = np.append(Tb, blue_flux[d] / total_flux[d])
            Tgb = np.append(Tgb, greenb_flux[d] / total_flux[d])

        if mp.am_master():
            plt.figure(dpi=150)
            plt.plot(wl, Tr*400, "r", label="R")
            plt.plot(wl, Tgr*400, "g", label="Gr")
            plt.plot(wl, Tb*400, "b", label="B")
            plt.plot(wl, Tgb*400, color='limegreen', label="Gb")
            
            plt.axis([0.40, 0.70, 0, 1])
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("Efficiency")
            plt.fill([0.425, 0.425, 0.485, 0.485], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
            plt.fill([0.515, 0.515, 0.575, 0.575], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
            plt.fill([0.595, 0.595, 0.655, 0.655], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
            plt.legend(loc="upper right")
            #plt.show()
            plt.savefig("./"+directory_name+"/Discrete/D_OE.png")
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

        np.savetxt("./"+directory_name+"/Discrete/R_discrete_5nm_data.txt" , Tr* 400)
        np.savetxt("./"+directory_name+"/Discrete/Gr_discrete_5nm_data.txt" , Tgr* 400)
        np.savetxt("./"+directory_name+"/Discrete/B_discrete_5nm_data.txt" , Tb* 400)
        np.savetxt("./"+directory_name+"/Discrete/Gb_discrete_5nm_data.txt" , Tgb* 400)
        np.savetxt("./"+directory_name+"/Discrete/Gavg_discrete_5nm_data.txt" , (Tgb+Tgr)* 200)

        np.savetxt("./"+directory_name+"/Discrete/MAX_R_discrete_5nm_data.txt" , np.array([np.max(Tr[41:61]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/MAX_Gr_discrete_5nm_data.txt" , np.array([np.max(Tgr[21:41]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/MAX_B_discrete_5nm_data.txt" , np.array([np.max(Tb[0:21]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/MAX_Gb_discrete_5nm_data.txt" , np.array([np.max(Tgb[21:41]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/MAX_Gavg_discrete_5nm_data.txt" , np.array([np.max((Tgb[21:41]+Tgr[21:41])* 200)]))

        np.savetxt("./"+directory_name+"/Discrete/mean_R_discrete_5nm_data.txt" , np.array([np.mean(Tr[40:53]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/mean_Gr_discrete_5nm_data.txt" , np.array([np.mean(Tgr[24:37]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/mean_B_discrete_5nm_data.txt" , np.array([np.mean(Tb[6:19]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/mean_Gb_discrete_5nm_data.txt" , np.array([np.mean(Tgb[24:37]* 400)]))
        np.savetxt("./"+directory_name+"/Discrete/mean_Gavg_discrete_5nm_data.txt" , np.array([np.mean((Tgb[24:37]+Tgr[24:37])* 200)]))