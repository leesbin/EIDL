# Description: This code is for the optimization of the color router

# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os
import Mapping
import Optimizer

mp.verbosity(1)

# Simulation Condtion ###############################################
Norm = True # FoM normalization                                     #
Dipersion_model = False # dispersion model                          #
#####################################################################

# Fab constraint ####################################################
ML = True # Multi layer                                             #
FL = True # Focal layer                                             #
Air_Conditions = False # Air grid                                   #
DTI = False # DTI layer                                             #
AR = False # AR coating                                             #
#####################################################################

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
#####################################################################

# resolution ########################################################
resolution = 50                                                     #
o_grid = 1/resolution # 20nm                                        #
design_region_resolution = int(resolution)                          #
#####################################################################

# Multi layer size ##################################################
mlsize_list = [20] # 200 nm                                         #
#####################################################################

for i in mlsize_list:

##### Directory Generation ################################################
    directory_name = "Base" + str(i)                                      #
    try:                                                                  #
        if not os.path.exists(directory_name):                            #
            os.makedirs(directory_name)                                   #
            print("디렉토리 생성 성공:", directory_name)                   #
        else:                                                             #
            print("디렉토리 이미 존재합니다:", directory_name)              #
    except OSError as e:                                                  #
        print("디렉토리 생성 실패:", directory_name, "| 에러:", e)          #
###########################################################################

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
        
        ml_thickness_1 = round(o_grid * i, 2) # multi layer thickness 400 nm 
        ml_thickness_2 = round(o_grid * i, 2) # multi layer thickness 400 nm 
        ml_thickness_3 = round(o_grid * i, 2) # multi layer thickness 400 nm 
        ml_thickness_4 = round(o_grid * i, 2) # multi layer thickness 400 nm  
        ml_thickness_5 = round(o_grid * i, 2) # multi layer thickness 400 nm 
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
        design_region_width_x = round(sp_size * 2 , 2) # Design region x 1200 nm
        design_region_width_y = round(sp_size * 2 , 2) # Design region y 1200 nm
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
        wavelengths_pieces = np.linspace(0.4*um_scale, 0.7*um_scale, 31)
        wavelengths = np.array([
                wavelengths_pieces[1], wavelengths_pieces[3], wavelengths_pieces[5], wavelengths_pieces[7], wavelengths_pieces[9],
                wavelengths_pieces[11], wavelengths_pieces[13], wavelengths_pieces[15], wavelengths_pieces[17], wavelengths_pieces[19],
                wavelengths_pieces[21], wavelengths_pieces[23], wavelengths_pieces[25], wavelengths_pieces[27], wavelengths_pieces[29]
        ])
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

##### Fabrication Constraints ########################################################################
    minimum_length = 0.06                                                                            #
    eta_e = 0.75                                                                                     #
    filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)                           #
    pad_value = int(filter_radius*2/o_grid)                                                          #
    eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)     #
######################################################################################################

    if ML: # Multi Design Region  #################################################################################################################################
        design_variables_0 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
        design_region_0 = mpa.DesignRegion(
            design_variables_0,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_1),
            ),
        )

        design_variables_1 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
        design_region_1 = mpa.DesignRegion(
            design_variables_1,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_2),
            ),
        )

        design_variables_2 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
        design_region_2 = mpa.DesignRegion(
            design_variables_2,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_3),
            ),
        )

        design_variables_3 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
        design_region_3 = mpa.DesignRegion(
            design_variables_3,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3 - el_thickness - ml_thickness_4/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_4),
            ),
        )

        design_variables_4 = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
        design_region_4 = mpa.DesignRegion(
            design_variables_4,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - ml_thickness_1 - el_thickness - ml_thickness_2 - el_thickness - ml_thickness_3 - el_thickness - ml_thickness_4 - el_thickness - ml_thickness_5/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_5),
            ),
        )

    else: # One Design Region
        design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
        design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height/2, 2)),
                size=mp.Vector3(design_region_width_x, design_region_width_y, design_region_height),
            ),
        )

    if True: ##### Set geometries #################################################################################################################################
        geometry = [
            # mp.Block(
            #     center=mp.Vector3(0,0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height/2, 2)), 
            #     size=mp.Vector3(design_region_width_x, design_region_width_y, design_region_height), 
            #     material=SiO2
            # ),
            # Focal Layer
            mp.Block(
                center=mp.Vector3(0, 0, round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height - fl_thickness/2, 2)), size=mp.Vector3(Sx, Sy, fl_thickness), material=Air
            ),
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

    if Norm: ##### normalization flux calculation #################################################################################################################
        sim = mp.Simulation(
            cell_size=cell_size, 
            boundary_layers=pml_layers,
            sources=source,
            default_material=Air, # 빈공간
            resolution=resolution,
            k_point = mp.Vector3(0,0,0)
        )

        b_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        b_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        b_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        g_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        g_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[9], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[10], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[11], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[12], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[13], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

        r_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
        r_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[14], 0 , 1, center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + 1/resolution,2)), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)


        #################################################################################################################################################################################

        sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-9))

        b_0_dft_array_Ex = sim.get_dft_array(b_0_tran_Ex, mp.Ex, 0)
        b_0_dft_array_Ey = sim.get_dft_array(b_0_tran_Ey, mp.Ey, 0)
        b_0_dft_array_Ez = sim.get_dft_array(b_0_tran_Ez, mp.Ez, 0)

        b_0_dft_array_Hx = sim.get_dft_array(b_0_tran_Hx, mp.Hx, 0)
        b_0_dft_array_Hy = sim.get_dft_array(b_0_tran_Hy, mp.Hy, 0)
        b_0_dft_array_Hz = sim.get_dft_array(b_0_tran_Hz, mp.Hz, 0)

        b_0_dft_total = - np.sum(b_0_dft_array_Ex*np.conj(b_0_dft_array_Hy)) + np.sum(b_0_dft_array_Ey*np.conj(b_0_dft_array_Hx))

        b_1_dft_array_Ex = sim.get_dft_array(b_1_tran_Ex, mp.Ex, 0)
        b_1_dft_array_Ey = sim.get_dft_array(b_1_tran_Ey, mp.Ey, 0)
        b_1_dft_array_Ez = sim.get_dft_array(b_1_tran_Ez, mp.Ez, 0)

        b_1_dft_array_Hx = sim.get_dft_array(b_1_tran_Hx, mp.Hx, 0)
        b_1_dft_array_Hy = sim.get_dft_array(b_1_tran_Hy, mp.Hy, 0)
        b_1_dft_array_Hz = sim.get_dft_array(b_1_tran_Hz, mp.Hz, 0)

        b_1_dft_total = - np.sum(b_1_dft_array_Ex*np.conj(b_1_dft_array_Hy)) + np.sum(b_1_dft_array_Ey*np.conj(b_1_dft_array_Hx))

        b_2_dft_array_Ex = sim.get_dft_array(b_2_tran_Ex, mp.Ex, 0)
        b_2_dft_array_Ey = sim.get_dft_array(b_2_tran_Ey, mp.Ey, 0)
        b_2_dft_array_Ez = sim.get_dft_array(b_2_tran_Ez, mp.Ez, 0)

        b_2_dft_array_Hx = sim.get_dft_array(b_2_tran_Hx, mp.Hx, 0)
        b_2_dft_array_Hy = sim.get_dft_array(b_2_tran_Hy, mp.Hy, 0)
        b_2_dft_array_Hz = sim.get_dft_array(b_2_tran_Hz, mp.Hz, 0)

        b_2_dft_total = - np.sum(b_2_dft_array_Ex*np.conj(b_2_dft_array_Hy)) + np.sum(b_2_dft_array_Ey*np.conj(b_2_dft_array_Hx))

        b_3_dft_array_Ex = sim.get_dft_array(b_3_tran_Ex, mp.Ex, 0)
        b_3_dft_array_Ey = sim.get_dft_array(b_3_tran_Ey, mp.Ey, 0)
        b_3_dft_array_Ez = sim.get_dft_array(b_3_tran_Ez, mp.Ez, 0)

        b_3_dft_array_Hx = sim.get_dft_array(b_3_tran_Hx, mp.Hx, 0)
        b_3_dft_array_Hy = sim.get_dft_array(b_3_tran_Hy, mp.Hy, 0)
        b_3_dft_array_Hz = sim.get_dft_array(b_3_tran_Hz, mp.Hz, 0)

        b_3_dft_total = - np.sum(b_3_dft_array_Ex*np.conj(b_3_dft_array_Hy)) + np.sum(b_3_dft_array_Ey*np.conj(b_3_dft_array_Hx))

        b_4_dft_array_Ex = sim.get_dft_array(b_4_tran_Ex, mp.Ex, 0)
        b_4_dft_array_Ey = sim.get_dft_array(b_4_tran_Ey, mp.Ey, 0)
        b_4_dft_array_Ez = sim.get_dft_array(b_4_tran_Ez, mp.Ez, 0)

        b_4_dft_array_Hx = sim.get_dft_array(b_4_tran_Hx, mp.Hx, 0)
        b_4_dft_array_Hy = sim.get_dft_array(b_4_tran_Hy, mp.Hy, 0)
        b_4_dft_array_Hz = sim.get_dft_array(b_4_tran_Hz, mp.Hz, 0)

        b_4_dft_total = - np.sum(b_4_dft_array_Ex*np.conj(b_4_dft_array_Hy)) + np.sum(b_4_dft_array_Ey*np.conj(b_4_dft_array_Hx))

        g_0_dft_array_Ex = sim.get_dft_array(g_0_tran_Ex, mp.Ex, 0)
        g_0_dft_array_Ey = sim.get_dft_array(g_0_tran_Ey, mp.Ey, 0)
        g_0_dft_array_Ez = sim.get_dft_array(g_0_tran_Ez, mp.Ez, 0)

        g_0_dft_array_Hx = sim.get_dft_array(g_0_tran_Hx, mp.Hx, 0)
        g_0_dft_array_Hy = sim.get_dft_array(g_0_tran_Hy, mp.Hy, 0)
        g_0_dft_array_Hz = sim.get_dft_array(g_0_tran_Hz, mp.Hz, 0)

        g_0_dft_total = - np.sum(g_0_dft_array_Ex*np.conj(g_0_dft_array_Hy)) + np.sum(g_0_dft_array_Ey*np.conj(g_0_dft_array_Hx))

        g_1_dft_array_Ex = sim.get_dft_array(g_1_tran_Ex, mp.Ex, 0)
        g_1_dft_array_Ey = sim.get_dft_array(g_1_tran_Ey, mp.Ey, 0)
        g_1_dft_array_Ez = sim.get_dft_array(g_1_tran_Ez, mp.Ez, 0)

        g_1_dft_array_Hx = sim.get_dft_array(g_1_tran_Hx, mp.Hx, 0)
        g_1_dft_array_Hy = sim.get_dft_array(g_1_tran_Hy, mp.Hy, 0)
        g_1_dft_array_Hz = sim.get_dft_array(g_1_tran_Hz, mp.Hz, 0)

        g_1_dft_total = - np.sum(g_1_dft_array_Ex*np.conj(g_1_dft_array_Hy)) + np.sum(g_1_dft_array_Ey*np.conj(g_1_dft_array_Hx))

        g_2_dft_array_Ex = sim.get_dft_array(g_2_tran_Ex, mp.Ex, 0)
        g_2_dft_array_Ey = sim.get_dft_array(g_2_tran_Ey, mp.Ey, 0)
        g_2_dft_array_Ez = sim.get_dft_array(g_2_tran_Ez, mp.Ez, 0)

        g_2_dft_array_Hx = sim.get_dft_array(g_2_tran_Hx, mp.Hx, 0)
        g_2_dft_array_Hy = sim.get_dft_array(g_2_tran_Hy, mp.Hy, 0)
        g_2_dft_array_Hz = sim.get_dft_array(g_2_tran_Hz, mp.Hz, 0)

        g_2_dft_total = - np.sum(g_2_dft_array_Ex*np.conj(g_2_dft_array_Hy)) + np.sum(g_2_dft_array_Ey*np.conj(g_2_dft_array_Hx))

        g_3_dft_array_Ex = sim.get_dft_array(g_3_tran_Ex, mp.Ex, 0)
        g_3_dft_array_Ey = sim.get_dft_array(g_3_tran_Ey, mp.Ey, 0)
        g_3_dft_array_Ez = sim.get_dft_array(g_3_tran_Ez, mp.Ez, 0)

        g_3_dft_array_Hx = sim.get_dft_array(g_3_tran_Hx, mp.Hx, 0)
        g_3_dft_array_Hy = sim.get_dft_array(g_3_tran_Hy, mp.Hy, 0)
        g_3_dft_array_Hz = sim.get_dft_array(g_3_tran_Hz, mp.Hz, 0)

        g_3_dft_total = - np.sum(g_3_dft_array_Ex*np.conj(g_3_dft_array_Hy)) + np.sum(g_3_dft_array_Ey*np.conj(g_3_dft_array_Hx))

        g_4_dft_array_Ex = sim.get_dft_array(g_4_tran_Ex, mp.Ex, 0)
        g_4_dft_array_Ey = sim.get_dft_array(g_4_tran_Ey, mp.Ey, 0)
        g_4_dft_array_Ez = sim.get_dft_array(g_4_tran_Ez, mp.Ez, 0)

        g_4_dft_array_Hx = sim.get_dft_array(g_4_tran_Hx, mp.Hx, 0)
        g_4_dft_array_Hy = sim.get_dft_array(g_4_tran_Hy, mp.Hy, 0)
        g_4_dft_array_Hz = sim.get_dft_array(g_4_tran_Hz, mp.Hz, 0)

        g_4_dft_total = - np.sum(g_4_dft_array_Ex*np.conj(g_4_dft_array_Hy)) + np.sum(g_4_dft_array_Ey*np.conj(g_4_dft_array_Hx))

        r_0_dft_array_Ex = sim.get_dft_array(r_0_tran_Ex, mp.Ex, 0)
        r_0_dft_array_Ey = sim.get_dft_array(r_0_tran_Ey, mp.Ey, 0)
        r_0_dft_array_Ez = sim.get_dft_array(r_0_tran_Ez, mp.Ez, 0)

        r_0_dft_array_Hx = sim.get_dft_array(r_0_tran_Hx, mp.Hx, 0)
        r_0_dft_array_Hy = sim.get_dft_array(r_0_tran_Hy, mp.Hy, 0)
        r_0_dft_array_Hz = sim.get_dft_array(r_0_tran_Hz, mp.Hz, 0)

        r_0_dft_total = - np.sum(r_0_dft_array_Ex*np.conj(r_0_dft_array_Hy)) + np.sum(r_0_dft_array_Ey*np.conj(r_0_dft_array_Hx))

        r_1_dft_array_Ex = sim.get_dft_array(r_1_tran_Ex, mp.Ex, 0)
        r_1_dft_array_Ey = sim.get_dft_array(r_1_tran_Ey, mp.Ey, 0)
        r_1_dft_array_Ez = sim.get_dft_array(r_1_tran_Ez, mp.Ez, 0)

        r_1_dft_array_Hx = sim.get_dft_array(r_1_tran_Hx, mp.Hx, 0)
        r_1_dft_array_Hy = sim.get_dft_array(r_1_tran_Hy, mp.Hy, 0)
        r_1_dft_array_Hz = sim.get_dft_array(r_1_tran_Hz, mp.Hz, 0)

        r_1_dft_total = - np.sum(r_1_dft_array_Ex*np.conj(r_1_dft_array_Hy)) + np.sum(r_1_dft_array_Ey*np.conj(r_1_dft_array_Hx))

        r_2_dft_array_Ex = sim.get_dft_array(r_2_tran_Ex, mp.Ex, 0)
        r_2_dft_array_Ey = sim.get_dft_array(r_2_tran_Ey, mp.Ey, 0)
        r_2_dft_array_Ez = sim.get_dft_array(r_2_tran_Ez, mp.Ez, 0)

        r_2_dft_array_Hx = sim.get_dft_array(r_2_tran_Hx, mp.Hx, 0)
        r_2_dft_array_Hy = sim.get_dft_array(r_2_tran_Hy, mp.Hy, 0)
        r_2_dft_array_Hz = sim.get_dft_array(r_2_tran_Hz, mp.Hz, 0)

        r_2_dft_total = - np.sum(r_2_dft_array_Ex*np.conj(r_2_dft_array_Hy)) + np.sum(r_2_dft_array_Ey*np.conj(r_2_dft_array_Hx))

        r_3_dft_array_Ex = sim.get_dft_array(r_3_tran_Ex, mp.Ex, 0)
        r_3_dft_array_Ey = sim.get_dft_array(r_3_tran_Ey, mp.Ey, 0)
        r_3_dft_array_Ez = sim.get_dft_array(r_3_tran_Ez, mp.Ez, 0)

        r_3_dft_array_Hx = sim.get_dft_array(r_3_tran_Hx, mp.Hx, 0)
        r_3_dft_array_Hy = sim.get_dft_array(r_3_tran_Hy, mp.Hy, 0)
        r_3_dft_array_Hz = sim.get_dft_array(r_3_tran_Hz, mp.Hz, 0)

        r_3_dft_total = - np.sum(r_3_dft_array_Ex*np.conj(r_3_dft_array_Hy)) + np.sum(r_3_dft_array_Ey*np.conj(r_3_dft_array_Hx))

        r_4_dft_array_Ex = sim.get_dft_array(r_4_tran_Ex, mp.Ex, 0)
        r_4_dft_array_Ey = sim.get_dft_array(r_4_tran_Ey, mp.Ey, 0)
        r_4_dft_array_Ez = sim.get_dft_array(r_4_tran_Ez, mp.Ez, 0)

        r_4_dft_array_Hx = sim.get_dft_array(r_4_tran_Hx, mp.Hx, 0)
        r_4_dft_array_Hy = sim.get_dft_array(r_4_tran_Hy, mp.Hy, 0)
        r_4_dft_array_Hz = sim.get_dft_array(r_4_tran_Hz, mp.Hz, 0)

        r_4_dft_total = - np.sum(r_4_dft_array_Ex*np.conj(r_4_dft_array_Hy)) + np.sum(r_4_dft_array_Ey*np.conj(r_4_dft_array_Hx))
    
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
        # extra_materials=[SiO2, SiN, Si, HfO2, Al2O3],
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
        frequencies=frequencies,
        decay_by=1e-3,
        # maximum_run_time=300,
    )
    
    if True: ##### Structure Plot ############################################################################################################################
        opt.plot2D(True, output_plane = mp.Volume(size = (Sx, 0, Sz), center = (0, design_region_width_y/4,0)),
                # source_parameters={'alpha':0},
                #     boundary_parameters={'alpha':0},
                )
        # plt.axis("off")
        plt.savefig("Lastdesignxz.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure round(-Sz/2 + Lpml + pml_gap,2)

        opt.plot2D(False, output_plane = mp.Volume(size = (0, Sy, design_region_height), center = (0,0,round(Sz / 2 - Lpml -  pml_2_src - src_2_geo - ar_thktop - design_region_height/2, 2))),
                source_parameters={'alpha':0},
                    boundary_parameters={'alpha':0},
                )
        # plt.axis("off")
        plt.xlabel("Width (μm)")
        plt.ylabel("Height (μm)")
        plt.savefig("Designregion.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure round(-Sz/2 + Lpml + pml_gap,2)

        opt.plot2D(True, output_plane = mp.Volume(size = (Sx, Sy, 0), center = (0, 0,round(-Sz/2 + Lpml + mon_2_pml - 1/resolution,2))),
                # source_parameters={'alpha':0},
                #     boundary_parameters={'alpha':0},
                )
        # plt.axis("off")
        plt.savefig("Lastdesignxy.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure 

##### Parameters for record ########################################################################
    evaluation_history = [] # FOM                                                                  #
    lr_history=[] # Learning rate                                                                  #
    t_history=[] # Iteration number                                                                #
    binarization_history = [] # Binarization                                                       #
    adjgrad_history=[] # Adjoint gradient mean                                                     #
    beta_history = [] # Binarization parameter                                                     #
                                                                                                   #
    cur_iter = [0]                                                                                 #
    numevl = 1                                                                                     #
####################################################################################################
    
    # Optimization parameters
    lr = 0.02  # learning rate
    optimizer = Optimizer.AdamOptimizer(lr=lr)   # Adam optimizer
    Momentum_dJ_du = False 

    # Function for optimization
    def f(v, cur_beta):
        global numevl
        print("Current iteration: {}".format(cur_iter[0] + 1))

        # Apply mapping
        x1, x2, x3, x4, x5 = Mapping.multiregion_mapping(v,                             # design variables
                                                         eta_i,                         # threshold point of the linear filter
                                                         cur_beta,                      # binarization parameter
                                                         0,                             # separate the design variables for layers
                                                         Nx,                            # number of design variables in x direction
                                                         Ny,                            # number of design variables in y direction
                                                         filter_radius,                 # filter radius
                                                         design_region_width_x,         # design region width in x direction
                                                         design_region_width_y,         # design region width in y direction
                                                         design_region_resolution,      # design region resolution
                                                         pad_value,                     # padding value
                                                         o_grid                         # one grid size
                                                         )
        
        # Save the entire design variables
        X =Mapping.multiregion_mapping( v,                             # design variables
                                        eta_i,                         # threshold point of the linear filter
                                        cur_beta,                      # binarization parameter
                                        1,                             # the entire design variables
                                        Nx,                            # number of design variables in x direction
                                        Ny,                            # number of design variables in y direction
                                        filter_radius,                 # filter radius
                                        design_region_width_x,         # design region width in x direction
                                        design_region_width_y,         # design region width in y direction
                                        design_region_resolution,      # design region resolution
                                        pad_value,                     # padding value
                                        o_grid                         # one grid size
                                        )

        binarization_degree = np.sum(np.abs(X-0.5)) / (0.5 * X.size)

        if True: ##### compute objective and gradient ############################################################################################
            
            # Calculate the fom and the gradient
            f0, dJ_du = opt([x1, x2, x3, x4, x5]) 

            # Compute the entire gradient
            t_dJ_du = []
            t_dJ_du = npa.concatenate([np.sum(dJ_du[0],axis=1), np.sum(dJ_du[1],axis=1), np.sum(dJ_du[2],axis=1), np.sum(dJ_du[3],axis=1), np.sum(dJ_du[4],axis=1)])
            t_dJ_du = ((t_dJ_du.reshape(5,Nx*Ny)).transpose()).flatten()

            if Momentum_dJ_du: ##### dJ_du momentum #####################################################################################################
                updated_v, adam_lr, adam_t, m_dJ_du = optimizer.update(v, t_dJ_du)
            else:
                m_dJ_du = t_dJ_du
            
            # Adjoint gradient
            if v.size > 0:
                gradient = v*0
                gradient[:] = tensor_jacobian_product(Mapping.multiregion_mapping, 0)(
                    v, 
                    eta_i, 
                    cur_beta, 
                    1,
                    Nx,                            
                    Ny,                           
                    filter_radius,              
                    design_region_width_x,         
                    design_region_width_y,       
                    design_region_resolution,      
                    pad_value,                     
                    o_grid,         
                    m_dJ_du
                )  # backpropagation
        
        adjgrad_norm = gradient.mean()

        if Momentum_dJ_du: #### Update design #############################################################################################################
            updated_v = v + gradient * lr
            updated_v = np.clip(updated_v, 0.0, 1.0)
        else: 
            # gradient momentum
            updated_v, adam_lr, adam_t, m_gradient = optimizer.update(v, gradient)

        if True: ##### Record the Parameters #####################################################################################################
            evaluation_history.append(np.real(f0))
            lr_history.append(adam_lr)
            t_history.append(adam_t)
            binarization_history.append(binarization_degree)
            adjgrad_history.append(adjgrad_norm)
            beta_history.append(cur_beta) 

        if True: ##### Save the structure of the current iteration #################################################################################
            np.savetxt("./" + directory_name+"/structure_0_"+str(numevl) +".txt", design_variables_0.weights)
            np.savetxt("./" + directory_name+"/structure_1_"+str(numevl) +".txt", design_variables_1.weights)
            np.savetxt("./" + directory_name+"/structure_2_"+str(numevl) +".txt", design_variables_2.weights)
            np.savetxt("./" + directory_name+"/structure_3_"+str(numevl) +".txt", design_variables_3.weights)
            np.savetxt("./" + directory_name+"/structure_4_"+str(numevl) +".txt", design_variables_4.weights)
        
        numevl += 1

        cur_iter[0] = cur_iter[0] + 1
    
        max_fom=max(evaluation_history)
        print("First FoM: {}".format(evaluation_history[0]))
        print("Current FoM: {}".format(np.real(f0)))
        print("Max FoM:", max_fom)
        print("Index of Max FoM:", evaluation_history.index(max_fom))
        print("Index of Current FoM:", evaluation_history.index(np.real(f0))) 

        return np.real(f0), updated_v, X

    if True: ##### Optimization ############################################################################################################################
        n = Nx * Ny * 5  # number of design variables
        x = np.ones((n,)) * 0.5

        # Initialize parameters
        initial_beta = 1
        final_beta = 64
        Max_iter = 400
        inner_iter = 0

        while inner_iter < Max_iter:
            # Calculate current beta with exponential increase
            cur_beta = initial_beta * (final_beta / initial_beta) ** (inner_iter / (Max_iter - 1))
            
            # Call function f and unpack its return values
            FoM, x, Geo = f(x, cur_beta)  # FoM, design variables, total geo
            
            # Calculate binarization index
            Bi_idx = np.sum(np.where(((1e-3 < Geo) & (Geo < (1 - 1e-3))), 0, 1)) / Geo.size
            
            print('Local optimum')
            
            # Check if binarization condition is met
            if Bi_idx > 0.95:  # Binarization tolerance
                print('Binarized')
                
                # Set cur_beta to infinity and perform one more iteration
                cur_beta = np.inf
                FoM, x, Geo = f(x, cur_beta)  # Final iteration with cur_beta as infinity
                
                # Break the loop after final iteration
                break
            
            # Increment the iteration counter
            inner_iter += 1

    if True: ##### Save result plot ########################################################################################################################
        # Save the result of the optimization parameters
        np.savetxt("./" + directory_name+"/evaluation.txt", evaluation_history)
        np.savetxt("./" + directory_name+"/lr_history.txt", lr_history)
        np.savetxt("./" + directory_name+"/t_history.txt", t_history)
        np.savetxt("./" + directory_name+"/binarization_history.txt", binarization_history)
        np.savetxt("./" + directory_name+"/adjgrad_history.txt", adjgrad_history)
        np.savetxt("./" + directory_name+"/beta_history.txt", beta_history)

        def extract_elements(lst):
            # 결과를 저장할 리스트를 생성합니다.
            result = []

            # 리스트의 길이만큼 반복하면서 4의 배수 인덱스의 요소를 추출합니다.
            for i in range(0, len(lst), 17):
                result.append(lst[i])

            return result

        # RGB FoM plot

        fred = extract_elements(fred)
        fgreen = extract_elements(fgreen)
        fblue = extract_elements(fblue)

        np.savetxt("./" + directory_name+"/fred.txt", fred)
        np.savetxt("./" + directory_name+"/fgreen.txt", fgreen)
        np.savetxt("./" + directory_name+"/fblue.txt", fblue)

        plt.figure()

        plt.plot(fred, "r-")
        plt.plot(fgreen, "g-")
        plt.plot(fblue, "b-")
        plt.grid(False)
        plt.tick_params(axis='x', direction='in', pad = 5)
        plt.tick_params(axis='y', direction='in', pad = 10)
        plt.xlabel("Iteration")
        plt.ylabel("FoM")
        plt.savefig("./" + directory_name+"/FoMrgbresult.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        
        # FoM plot

        plt.figure()
        plt.plot(evaluation_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("FoM")
        plt.savefig("./" + directory_name+"/FoMresult.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        # Binarization plot

        plt.figure()
        plt.plot(binarization_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Binarization")
        plt.savefig("./" + directory_name+"/binarization_history.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        # Learning rate plot

        plt.figure()
        plt.plot(lr_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.savefig("./" + directory_name+"/lr_history.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        # Adjoint gradient plot

        plt.figure()
        plt.plot(adjgrad_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Adjgrad_Norm")
        plt.savefig("./" + directory_name+"/adjgrad_norm.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        # Binarization parameter plot
        
        plt.figure()
        plt.plot(beta_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Beta")
        plt.savefig("./" + directory_name+"/beta.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        # Save the last design variables
        np.savetxt("./" + directory_name+"/lastdesign1.txt", design_variables_0.weights)
        np.savetxt("./" + directory_name+"/lastdesign2.txt", design_variables_1.weights)
        np.savetxt("./" + directory_name+"/lastdesign3.txt", design_variables_2.weights)
        np.savetxt("./" + directory_name+"/lastdesign4.txt", design_variables_3.weights)
        np.savetxt("./" + directory_name+"/lastdesign5.txt", design_variables_4.weights)
