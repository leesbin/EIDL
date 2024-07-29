
# 접는용
simcondition = 1
designplot = 0
DFTfield = 0
Opticaleff = 1
layerplot = 0
# sen+GG
Sensitivity = 0
Crosstalk = 0
Discrete = 0

# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os

# os.chdir('/root/samsung/')

mp.verbosity(1)

x_list = [2] #[0,1,2]

# def make_dir(i):
#     if i == 0:
#         directory_name = "x-polarization"
#     elif i == 1:
#         directory_name = "y-polarization"
#     elif i == 2:
#         directory_name = "xy-polarization"
#     else:
#         print("잘못된 입력입니다.")
#         return
    
#     try:
#         if not os.path.exists(directory_name):
#             os.makedirs(directory_name)
#             print("디렉토리 생성 성공:", directory_name)
#         else:
#             print("디렉토리 이미 존재합니다:", directory_name)
#     except OSError as e:
#         print("디렉토리 생성 실패:", directory_name, "| 에러:", e)
#     else:
#         try:
#             sub_folders = ['Crosstalk', 'Design', 'QE_data', 'Sensitivity_data', 'DFTfield', 'Discrete']
#             for folder_name in sub_folders:
#                 sub_folder_path = os.path.join(directory_name, folder_name)
#                 if not os.path.exists(sub_folder_path):
#                     os.makedirs(sub_folder_path)
#                     print("폴더 생성 성공:", sub_folder_path)
#                 else:
#                     print("폴더 이미 존재합니다:", sub_folder_path)
#         except Exception as e:
#             print("폴더 생성 실패:", directory_name, "| 에러:", e)


D_M = 0 

# scaling & refractive index
um_scale = 1 # 1A = 1 um

Air = mp.Medium(index=1.0)
SiO2 = mp.Medium(index=1.45) # n=1.4529 @ 0.55um
TiO2 = mp.Medium(index=2.65) # n=2.6479 @ 0.55um
            
mlsize_list = [12] #[320nm, 480nm, 640nm] 밑에서 mlsize 짝수layer에서 위치 2로 나눠야해서 짝수여야함
flsize_list = [20] #[400nm, 800nm, 1200nm]
source_loc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
monitor_loc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
resolution = 25
o_grid = 1/resolution # 40nm

for p in monitor_loc:
    for r in source_loc:
        for i in x_list:

            if i == 0:
                directory_name = "Result_" + str(round(o_grid*p,2)) + "_" + str(round(o_grid*q,2)) + "_" + str(round(r,2))+"/x-polarization"
                try:
                    if not os.path.exists(directory_name):
                        os.makedirs(directory_name)
                        print("디렉토리 생성 성공:", directory_name)
                    else:
                        print("디렉토리 이미 존재합니다:", directory_name)

                except OSError as e:
                    print("디렉토리 생성 실패:", directory_name, "| 에러:", e)
            elif i == 1:
                directory_name = "Result_" + str(round(o_grid*p,2)) + "_" + str(round(o_grid*q,2)) + "_" + str(round(r,2))+"/y-polarization"
                try:
                    if not os.path.exists(directory_name):
                        os.makedirs(directory_name)
                        print("디렉토리 생성 성공:", directory_name)
                    else:
                        print("디렉토리 이미 존재합니다:", directory_name)

                except OSError as e:
                    print("디렉토리 생성 실패:", directory_name, "| 에러:", e)
            
            elif i == 2:
                directory_name = "Result_" + str(round(p,2)) + "_" + str(round(r,2))+"/xy-polarization"
                try:
                    if not os.path.exists(directory_name):
                        os.makedirs(directory_name)
                        print("디렉토리 생성 성공:", directory_name)
                    else:
                        print("디렉토리 이미 존재합니다:", directory_name)

                except OSError as e:
                    print("디렉토리 생성 실패:", directory_name, "| 에러:", e)


            if simcondition == 1 : 


                # 해상도 및 사이즈 설정
                
                fl_thickness = round(o_grid * 20, 2) # focal layer thickness 500 nm
                ml_thickness_1 = round(o_grid * 12, 2) # multi layer thickness 200 nm or 100 nm
                # ml_thickness_2 = round(o_grid * 8, 2) # multi layer thickness 200 nm or 100 nm
                # ml_thickness_3 = round(o_grid * 6, 2) # multi layer thickness 200 nm or 100 nm
                # ml_thickness_4 = round(o_grid * 6, 2) # multi layer thickness 200 nm or 100 nm 
                # ml_thickness_5 = round(o_grid * 6, 2) # multi layer thickness 200 nm or 100 nm

                Lpml = round(o_grid * 10, 2) # PML thickness 400 nm 
                pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]
                Sourcegap = round(o_grid * r, 2) # pml to Source length 80nm
                Monitorgap = round(o_grid * p, 2) # pml to Monitor length 80nm
                Sourcespace = round(o_grid * 25, 2) # Source propagation 1um

                # 설계 공간
                design_region_width_x = 1.28 # 디자인 영역 x 1280 nm
                design_region_width_y = 1.28 # 디자인 영역 y 1280 nm
                design_region_height = round(ml_thickness_1, 2) # + ml_thickness_3 + ml_thickness_4 + ml_thickness_5 # 디자인 영역 높이 z 
                # 전체 공간
                Sx = design_region_width_x
                Sy = design_region_width_y
                Sz = round(Lpml + Sourcegap + Sourcespace  + design_region_height + fl_thickness + Monitorgap + Lpml, 2)            
                cell_size = mp.Vector3(Sx, Sy, Sz)

                # 파장, 주파수 설정
                wavelengths = np.linspace(0.4*um_scale, 0.7*um_scale, 31) 
                frequencies = 1/wavelengths
                nf = len(frequencies) # number of frequencies

                # structure load
                structure_weight_0 = np.loadtxt('lastdesign1.txt')
                #structure_weight_1 = np.loadtxt('/root/samsung/Base_0.32_0.6_0.04_3/lastdesign2.txt')
                # structure_weight_2 = np.loadtxt('lastdesign3.txt')
                # structure_weight_3 = np.loadtxt('lastdesign4.txt')
                # structure_weight_4 = np.loadtxt('lastdesign5.txt')
                # structure_weight = np.where(structure_weight < 0.5, 0.0, 1.0)
                # np.savetxt('check.txt',structure_weight)

                # 파장, 주파수 설정
                wavelengths = np.linspace(0.4*um_scale, 0.7*um_scale, 31) 
                frequencies = 1/wavelengths
                nf = len(frequencies) # number of frequencies

                # Fabrication Constraints 설정

                minimum_length = 0.030 * um_scale # minimum length scale (microns)
                eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
                # eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
                # eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
                # filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
                filter_radius = o_grid * 2 # minimum length scale (microns) 80nm
                design_region_resolution = int(resolution)


                # source 설정
                width = 0.4

                fcen_red = 1/(0.65*um_scale)
                fwidth_red = fcen_red * width

                fcen_green = 1/(0.55*um_scale)
                fwidth_green = fcen_green * width

                fcen_blue = 1/(0.45*um_scale)
                fwidth_blue = fcen_blue * width

                src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)

                src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=fwidth_green, is_integrated=True)

                src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=fwidth_blue, is_integrated=True)

                source_center = [0, 0, round(Sz / 2 - Lpml - Sourcegap, 2) ] # Source 위치
                
                source_size = mp.Vector3(Sx, Sy, 0)

                source = [mp.Source(src_0, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_0, component=mp.Ey, size=source_size, center=source_center,),
                            mp.Source(src_1, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_1, component=mp.Ey, size=source_size, center=source_center,),
                            mp.Source(src_2, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_2, component=mp.Ey, size=source_size, center=source_center,)]


                # 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
                Nx = int(round(design_region_resolution * design_region_width_x)) + 1
                Ny = int(round(design_region_resolution * design_region_width_y)) + 1 
                Nz = int(round(design_region_resolution * design_region_height)) + 1


                # 설계 영역과 물질을 바탕으로 설계 영역 설정
                design_variables_0 = mp.MaterialGrid(mp.Vector3(Nx, Ny, 1), SiO2, TiO2, grid_type="U_MEAN",do_averaging=False)
                design_variables_0.update_weights(structure_weight_0)
                design_region_0 = mpa.DesignRegion(
                    design_variables_0,
                    volume=mp.Volume(
                        center=mp.Vector3(0, 0, round(Sz / 2 - Lpml - Sourcegap - Sourcespace -  0.5 * ml_thickness_1, 2)),
                        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_thickness_1),
                    ),
                )

            


                # design region과 동일한 size의 Block 생성
                geometry = [
                    mp.Block(
                        center=design_region_0.center, size=design_region_0.size, material=design_variables_0
                    ),
                ]

                # Meep simulation 세팅

                sim = mp.Simulation(
                    cell_size=cell_size, 
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=Air, # 빈공간
                    resolution=resolution,
                    k_point = mp.Vector3(0,0,0),
                    eps_averaging= False,
                    extra_materials=[SiO2,TiO2],
                )

                ###############################################################################################################################
                # ## 2. Optimization Environment

                # 모니터 위치와 크기 설정 (focal point)
                monitor_position_1, monitor_size_1 = mp.Vector3(design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), mp.Vector3(round(0.32,2),round(0.32,2),0)
                monitor_position_2, monitor_size_2 = mp.Vector3(-design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), mp.Vector3(round(0.32,2),round(0.32,2),0) 
                monitor_position_3, monitor_size_3 = mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), mp.Vector3(round(0.32,2),round(0.32,2),0) 
                monitor_position_4, monitor_size_4 = mp.Vector3(design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), mp.Vector3(round(0.32,2),round(0.32,2),0) 


                # FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함

                FourierFields_1_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ex,yee_grid=True)

                FourierFields_2_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ex,yee_grid=True)

                FourierFields_3_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ex,yee_grid=True)

                FourierFields_4_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Ex,yee_grid=True)

                FourierFields_1_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)

                FourierFields_2_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)

                FourierFields_3_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)

                FourierFields_4_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_4,size=monitor_size_4),mp.Ey,yee_grid=True)

                ob_list = [FourierFields_1_x, FourierFields_1_y, FourierFields_2_x,FourierFields_2_y, FourierFields_3_x, FourierFields_3_y, FourierFields_4_x, FourierFields_4_y]


                fred = []
                fgreen = []
                fblue = []
                # J : Objective function
                # FourierFields가 측정한 필드, 모니터의 중심에서 E 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
                # [frequency index, moniter index]
                def J(fields_1_x, fields_1_y, fields_2_x, fields_2_y, fields_3_x, fields_3_y, fields_4_x, fields_4_y):

                    red = npa.sum(npa.abs(fields_1_x[21:30,:]) ** 2)  + npa.sum(npa.abs(fields_1_y[21:30,:]) ** 2)
                    green = npa.sum(npa.abs(fields_2_x[11:20,:]) ** 2) + npa.sum(npa.abs(fields_2_y[11:20,:]) ** 2) + npa.sum(npa.abs(fields_4_x[11:20,:]) ** 2) + npa.sum(npa.abs(fields_4_y[11:20,:]) ** 2)
                    blue = npa.sum(npa.abs(fields_3_x[0:10,:]) ** 2) + npa.sum(npa.abs(fields_3_y[0:10,:]) ** 2)
                
                    redfactor = 1
                    greenfactor = 1
                    bluefactor = 1

                    fred.append(red/redfactor)
                    fgreen.append(green/greenfactor)
                    fblue.append(blue/bluefactor)

                    return blue/bluefactor + green/greenfactor + red/redfactor

                # optimization 설정

                opt = mpa.OptimizationProblem(
                    simulation=sim,
                    objective_functions=[J],
                    objective_arguments=ob_list,
                    design_regions=[design_region_0],
                    frequencies=frequencies,
                    decay_by=1e-3,
                    maximum_run_time=300
                )

                # 함수 설정

                evaluation_history = []
                cur_iter = [0]
                numevl = 1


            ###############################################################################################################################
            if designplot == 1 :
                # ## 3. Design plot
                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)),
                        # source_parameters={'alpha':0},
                        #     boundary_parameters={'alpha':0},
                        )
                # plt.axis("off")
                plt.savefig("./"+directory_name+"/Design/Lastdesignxz.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                opt.plot2D(False, output_plane = mp.Volume(size = (0, np.inf, np.inf), center = (0,0,0)),
                        # source_parameters={'alpha':0},
                        #     boundary_parameters={'alpha':0},
                        )
                # plt.axis("off")
                plt.savefig("./"+directory_name+"/Design/Lastdesignyz.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, 0), center = (0,0,0.5)))
                plt.savefig("./"+directory_name+"/Design/Lastdesignxy.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, ), center = (0,0,round(Sz / 2 - Lpml - Sourcespace - ar_thktop - design_region_height - fl_thickness + air_thickness/2, 2))))
                plt.savefig("./"+directory_name+"/Design/Lastdesignxy1.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, ), center = (0,0,round(Sz / 2 - Lpml - Sourcespace - ar_thktop - design_region_height - fl_thickness - ar_thkbottom - sd_thickness + dti_thickness/2, 2))))
                plt.savefig("./"+directory_name+"/Design/Lastdesignxy2.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                # ## 4. DFT fields

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

                src = mp.GaussianSource(frequency=frequencies[4], fwidth=fwidth_blue, is_integrated=True)

                if i == 0:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
                elif i == 1:
                    source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                elif i == 2:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                
                
                
                opt.sim.change_sources(source)

                plt.figure()

                tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

                tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[4], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

                Ex_Namei="Ex_b_field"
                Ey_Namei="Ey_b_field"
                Ez_Namei="Ez_b_field"

                Hx_Namei="Hx_b_field"
                Hy_Namei="Hy_b_field"
                Hz_Namei="Hz_b_field"

                pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)) #pt는 transmitted flux region과 동일

                if i == 0:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))
                elif i == 1:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ey,pt,1e-3))
                elif i == 2:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))

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

                src = mp.GaussianSource(frequency=frequencies[13], fwidth=fwidth_green, is_integrated=True)
                if i == 0:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
                elif i == 1:
                    source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                elif i == 2:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                opt.sim.change_sources(source)

                plt.figure()


                tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

                tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[13], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

                Ex_Namei="Ex_g_field"
                Ey_Namei="Ey_g_field"
                Ez_Namei="Ez_g_field"

                Hx_Namei="Hx_g_field"
                Hy_Namei="Hy_g_field"
                Hz_Namei="Hz_g_field"

                if i == 0:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))
                elif i == 1:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ey,pt,1e-3))
                elif i == 2:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))

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

                src = mp.GaussianSource(frequency=frequencies[21], fwidth=fwidth_green, is_integrated=True)
                if i == 0:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
                elif i == 1:
                    source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                elif i == 2:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                opt.sim.change_sources(source)

                plt.figure()

                red_y = opt.sim.add_dft_fields([mp.Ey], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                red_x = opt.sim.add_dft_fields([mp.Ex], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)


                tran_Ex = opt.sim.add_dft_fields([mp.Ex], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ey = opt.sim.add_dft_fields([mp.Ey], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Ez = opt.sim.add_dft_fields([mp.Ez], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)

                tran_Hx = opt.sim.add_dft_fields([mp.Hx], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hy = opt.sim.add_dft_fields([mp.Hy], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)
                tran_Hz = opt.sim.add_dft_fields([mp.Hz], frequencies[21], 0 , 1, center = (0,0,round(-Sz/2 + Lpml + sd_thickness + 1/resolution,2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0), yee_grid=True)


                Ex_Namei="Ex_r_field"
                Ey_Namei="Ey_r_field"
                Ez_Namei="Ez_r_field"

                Hx_Namei="Hx_r_field"
                Hy_Namei="Hy_r_field"
                Hz_Namei="Hz_r_field"

                if i == 0:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))
                elif i == 1:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ey,pt,1e-3))
                elif i == 2:
                    opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ex,pt,1e-3))

                opt.sim.output_dft(tran_Ex,"./"+directory_name+"/DFTfield/"+Ex_Namei)
                opt.sim.output_dft(tran_Ey,"./"+directory_name+"/DFTfield/"+Ey_Namei)
                opt.sim.output_dft(tran_Ez,"./"+directory_name+"/DFTfield/"+Ez_Namei)

                opt.sim.output_dft(tran_Hx,"./"+directory_name+"/DFTfield/"+Hx_Namei)
                opt.sim.output_dft(tran_Hy,"./"+directory_name+"/DFTfield/"+Hy_Namei)
                opt.sim.output_dft(tran_Hz,"./"+directory_name+"/DFTfield/"+Hz_Namei)
            ###############################################################################################################################
            if Opticaleff == 1 :
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
                

                fcen = (1/(0.30 * um_scale) + 1/(0.80 * um_scale))/2
                df = 1 /(0.30 * um_scale) - 1/(0.80 * um_scale)
                nfreq = 300

                src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
                if i == 0:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
                elif i == 1:
                    source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                elif i == 2:
                    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
                opt.sim.change_sources(source)


                # reflection moniter 설정

                refl_fr = mp.FluxRegion(
                    center=mp.Vector3(0, 0, round(Sz / 2 - Lpml - o_grid, 2)), size=mp.Vector3(Sx, Sy, 0)
                )
                refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

                # transmission moiniter 설정

                tran_t = mp.FluxRegion(
                    center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + Monitorgap, 2)), size=mp.Vector3(Sx, Sy, 0)
                )
                tran_total = opt.sim.add_flux(fcen, df, nfreq, tran_t)


                # pt는 transmitted flux region과 동일

                pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + Monitorgap, 2))

                #source가 끝난 후에 50 동안 계속 실행하며 component는 Ey, pt 설계의 끝에서 |Ey|^2의 값이 최대 값으로부터 1/1000 만큼 감쇠할때까지
                #추가적인 50 단위의 시간 실행 -> Fourier-transform 수렴예상

                if i == 0:
                    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))
                elif i == 1:
                    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))
                elif i == 2:
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
                    center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + Monitorgap, 2)), size=mp.Vector3(design_region_width_x, design_region_width_y, 0)
                )

                tran_pixel = opt.sim.add_flux(fcen, df, nfreq, tran_p)

                #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
                #the Fourier transforms of the scattered fields 에서 빼줍니다.

                opt.sim.load_minus_flux_data(refl, straight_refl_data)

                #각각 픽셀의 flux 구하기

                tran_r = mp.FluxRegion(
                    center=mp.Vector3(design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
                )

                tran_gr = mp.FluxRegion(
                    center=mp.Vector3(-design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
                )

                tran_b = mp.FluxRegion(
                    center=mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
                )

                tran_gb = mp.FluxRegion(
                    center=mp.Vector3(design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + Monitorgap), size=mp.Vector3(design_region_width_x/2, design_region_width_y/2, 0)
                )

                tran_red = opt.sim.add_flux(fcen, df, nfreq, tran_r)
                tran_greenr = opt.sim.add_flux(fcen, df, nfreq, tran_gr)
                tran_blue = opt.sim.add_flux(fcen, df, nfreq, tran_b)
                tran_greenb = opt.sim.add_flux(fcen, df, nfreq, tran_gb)

                

                plt.figure(dpi=150)
                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, 0), center=mp.Vector3(0, 0, round(Sz / 2 - Lpml - Sourcegap - Sourcespace -  0.5 * ml_thickness_1, 2))))
                plt.savefig("./"+directory_name+"/design_1.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                plt.figure(dpi=150)
                opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)))
                plt.savefig("./"+directory_name+"/plot.png")
                plt.cla()   # clear the current axes
                plt.clf()   # clear the current figure
                plt.close() # closes the current figure

                if i == 0:
                    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))
                elif i == 1:
                    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, 0))
                elif i == 2:
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
                    Tg = np.add(Tgr, Tgb)

                Tr_mean=round(np.mean(Tr[0:66]),3)
                Tr_max=round(np.max(Tr[0:66]),3)

                Tg_mean=round(np.mean(Tgr[66:160]),3)
                Tg_max=round(np.max(Tgr[66:160]),3)

                Tb_mean=round(np.mean(Tb[160:300]),3)
                Tb_max=round(np.max(Tb[160:300]),3)


                #print(Tr_mean,Tg_mean,Tb_mean,Tr_max,Tg_max,Tb_max)
                if Tgr.all()==Tgb.all():
                    is_same='yes'
                else:
                    is_same='no'

                results = (
                    f"Tb_mean: {Tb_mean} "
                    f"Tg_mean: {Tg_mean} "
                    f"Tr_mean: {Tr_mean}\n"
                    f"Tb_max: {Tb_max} "
                    f"Tg_max: {Tg_max} "
                    f"Tr_max: {Tr_max}\n"
                    f"Tgr==Tgb?: {is_same}\n"
                )
                
                with open("./"+directory_name+"/Opticaleff_results.txt", "w") as file:
                    file.write(results)

                if mp.am_master():
                    plt.figure(dpi=150)
                    plt.plot(wl, Tr, "r", label="R")
                    plt.plot(wl, Tgr, "g", label="G")
                    plt.plot(wl, Tb, "b", label="B")
                    #plt.plot(wl, Tgb, color='limegreen', label="Gr")
                    #plt.plot(wl, Tg, color='darkgreen', label="G")

                    plt.axis([0.40, 0.70, 0, 1])
                    plt.xlabel("wavelength (μm)")
                    plt.ylabel("Efficiency")
                    plt.fill([0.4, 0.4, 0.5, 0.5], [-0.03, 1.03, 1.03, -0.03], color='lightblue', alpha=0.5)
                    plt.fill([0.5, 0.5, 0.6, 0.6], [-0.03, 1.03, 1.03, -0.03], color='lightgreen', alpha=0.5)
                    plt.fill([0.6, 0.6, 0.7, 0.7], [-0.03, 1.03, 1.03, -0.03], color='lightcoral', alpha=0.5)
                    plt.legend(loc="upper right")
                    #plt.show()
                    plt.savefig("./"+directory_name+"/QE.png")
                    plt.cla()   # clear the current axes
                    plt.clf()   # clear the current figure
                    plt.close() # closes the current figure


                # np.savetxt("./"+directory_name+"/R_raw_data.txt" , Tr* 100)
                # np.savetxt("./"+directory_name+"/Gr_raw_data.txt" , Tgr* 100)
                # np.savetxt("./"+directory_name+"/B_raw_data.txt" , Tb* 100)
                # np.savetxt("./"+directory_name+"/Gb_raw_data.txt" , Tgb* 100)
                # np.savetxt("./"+directory_name+"/G_raw_data.txt" , Tg* 100)



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
                    plt.xlabel("wavelength (μm)")
                    plt.ylabel("Transmittance")
                    plt.fill([0.40, 0.40, 0.50, 0.50], [0.0, 1.0, 1.0, 0.0], color='lightblue', alpha=0.5)
                    plt.fill([0.50, 0.50, 0.60, 0.60], [0.0, 1.0, 1.0, 0.0], color='lightgreen', alpha=0.5)
                    plt.fill([0.60, 0.60, 0.70, 0.70], [0.0, 1.0, 1.0, 0.0], color='lightcoral', alpha=0.5)
                    plt.legend(loc="upper right")
                    #plt.show()
                    plt.savefig("./"+directory_name+"/T_R.png")
                    plt.cla()   # clear the current axes
                    plt.clf()   # clear the current figure
                    plt.close() # closes the current figure
            ###############################################################################################################################