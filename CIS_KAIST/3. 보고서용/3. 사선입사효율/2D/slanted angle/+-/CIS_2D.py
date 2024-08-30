#!/usr/bin/env python
# coding: utf-8

# In[112]:

#First FoM: 1916.29499624842
#Current FoM: 20641.385989114748

#Elapsed run time = 4605.2399 s



# 4_core_Elapsed run time = 1408.4338 s


# 시뮬레이션 라이브러리 불러오기 


import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import pickle
import sys



# 물질 설정 및 Refractive Index 설정
mp.verbosity(1)
um_scale = 2
seed = 240  # 난수 발생 시드(seed)를 240으로 설정 (재현성을 위해 난수 시드를 고정)
np.random.seed(seed)  # numpy의 난수 생성기의 시드를 설정하여 난수의 재현성 보장

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
SiPD = mp.Medium(epsilon=5)


# cell 환경 설정
resolution = 25 # 해상도
design_region_width = 3 # 디자인 영역 너비
design_region_height = 4 # 디자인 영역 높이
pml_size = 1 # PML 영역 크기

pml_layers = [mp.PML(thickness = pml_size, direction = mp.Y)]

rot_angle = np.radians(6)

rot_angle_n = np.radians(-6)

k_point_0 = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), 0)

k_point = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), rot_angle)

k_point_n = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), rot_angle_n)




# 시뮬레이션 공간 설정
Sx = design_region_width 
Sy = 2 * pml_size + design_region_height + 2
cell_size = mp.Vector3(Sx, Sy)




# 파장, 주파수 설정
wavelengths = np.array([0.43*um_scale, 0.45*um_scale, 0.47*um_scale, 0.53*um_scale, 0.55*um_scale, 0.57*um_scale, 0.62*um_scale, 0.65*um_scale, 0.68*um_scale])
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# Fabrication Constraints 설정

minimum_length = 0.05  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)



# Source 설정


width = 0.1
fcen_red0 = frequencies[6]
fwidth_red0 = width * fcen_red0
fcen_red1 = frequencies[7]
fwidth_red1 = width * fcen_red1
fcen_red2 = frequencies[8]
fwidth_red2 = width * fcen_red2

fcen_green0 = frequencies[3]
fwidth_green0 = width * fcen_green0
fcen_green1 = frequencies[4]
fwidth_green1 = width * fcen_green1
fcen_green2 = frequencies[5]
fwidth_green2 = width * fcen_green2

fcen_blue0 = frequencies[0]
fwidth_blue0 = width * fcen_blue0
fcen_blue1 = frequencies[1]
fwidth_blue1 = width * fcen_blue1
fcen_blue2 = frequencies[2]
fwidth_blue2 = width * fcen_blue2


source_center = [0, Sy / 2 - pml_size - 0.5, 0] # Source 위치
source_size = mp.Vector3(Sx , 0, 0)

src_0 = mp.GaussianSource(frequency=fcen_red2, fwidth=fwidth_red2, is_integrated=True)
src_1 = mp.GaussianSource(frequency=fcen_red1, fwidth=fwidth_red1, is_integrated=True)
src_2 = mp.GaussianSource(frequency=fcen_red0, fwidth=fwidth_red0, is_integrated=True)
src_3 = mp.GaussianSource(frequency=fcen_green2, fwidth=fwidth_green2, is_integrated=True)
src_4 = mp.GaussianSource(frequency=fcen_green1, fwidth=fwidth_green1, is_integrated=True)
src_5 = mp.GaussianSource(frequency=fcen_green0, fwidth=fwidth_green0, is_integrated=True)
src_6 = mp.GaussianSource(frequency=fcen_blue2, fwidth=fwidth_blue2, is_integrated=True)
src_7 = mp.GaussianSource(frequency=fcen_blue1, fwidth=fwidth_blue1, is_integrated=True)
src_8 = mp.GaussianSource(frequency=fcen_blue0, fwidth=fwidth_blue0, is_integrated=True)

source0 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),
		 ]

          
source1 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
		 ]
		 
source2 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle_n == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),
		 ]




# 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
Nx = int(round(design_region_resolution * design_region_width)) + 1
Ny = int(round(design_region_resolution * design_region_height)) + 1


# 설계 영역과 물질을 바탕으로 설계 영역 설정
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)



# filter.py conic_filter 함수와 simple_2d_filter 함수를 사용
def mapping(x, eta, beta):
    # filter
    x = x.flatten()
    
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        design_region_resolution,
    )

    # projection
    # 출력값 0 ~ 1으로 제한
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return projected_field.flatten()


# design region과 동일한 size의 Block 생성
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(0, -3, 0), size=mp.Vector3(Sx, 2, 0), material=SiPD
    ),
    # DTI가 있을 경우 사용
    # mp.Block(
    #     center=mp.Vector3(-1, -3, 0), size=mp.Vector3(0.5, 2, 0), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(0, -3, 0), size=mp.Vector3(0.5, 2, 0), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(1, -3, 0), size=mp.Vector3(0.5, 2, 0), material=SiPD
    # )
]



# Meep simulation 세팅
sim0 = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source0,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point_0 # bloch boundary
)

sim1 = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source1,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point,
)

sim2 = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source2,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point_n,
)


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-1, -2 - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -2 - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(1, -2 - 0.5/resolution), mp.Vector3(0.01,0)


# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_1 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_2 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_3 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_4 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_5 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_6 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_7 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_8 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
ob_list_0 = [FourierFields_0, FourierFields_1, FourierFields_2,FourierFields_3, FourierFields_4, FourierFields_5 ,FourierFields_6, FourierFields_7, FourierFields_8]

FourierFields_01 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_11 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_21 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_31 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_41 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_51 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_61 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_71 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_81 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
ob_list_1 = [FourierFields_01, FourierFields_11, FourierFields_21,FourierFields_31, FourierFields_41, FourierFields_51 ,FourierFields_61, FourierFields_71, FourierFields_81]

FourierFields_02 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_12 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_22 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_32 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_42 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_52 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_62 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_72 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_82 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
ob_list_2 = [FourierFields_02, FourierFields_12, FourierFields_22,FourierFields_32, FourierFields_42, FourierFields_52 ,FourierFields_62, FourierFields_72, FourierFields_82]


# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
# [frequency index, moniter index]
def J_0(fields_0, fields_1, fields_2,fields_3, fields_4, fields_5,fields_6, fields_7, fields_8):
    return npa.abs(fields_8[0,1]) ** 2 + npa.abs(fields_7[1,1]) ** 2 + npa.abs(fields_6[2,1]) ** 2 + npa.abs(fields_5[3,1]) ** 2 + npa.abs(fields_4[4,1]) ** 2 + npa.abs(fields_3[5,1]) ** 2 + npa.abs(fields_2[6,1]) ** 2 + npa.abs(fields_1[7,1]) ** 2 + npa.abs(fields_0[8,1]) ** 2



# 최적화 설정
opt0 = mpa.OptimizationProblem(
    simulation=sim0,
    objective_functions=[J_0],
    objective_arguments=ob_list_0,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1, # 모니터에 남아있는 필드 값의 비율
)
opt1 = mpa.OptimizationProblem(
    simulation=sim1,
    objective_functions=[J_0],
    objective_arguments=ob_list_1,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1, # 모니터에 남아있는 필드 값의 비율
)

opt2 = mpa.OptimizationProblem(
    simulation=sim2,
    objective_functions=[J_0],
    objective_arguments=ob_list_2,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1, # 모니터에 남아있는 필드 값의 비율
)

evaluation_history = []
evaluation_history_0 = []
evaluation_history_1 = []
evaluation_history_2 = []

cur_iter = [0]

def f(v, gradient, beta):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0_0, dJ_du_0 = opt0([mapping(v, eta_i, beta)])  # compute objective and gradient
    f0_1, dJ_du_1 = opt1([mapping(v, eta_i, beta)])  # compute objective and gradient
    f0_2, dJ_du_2 = opt2([mapping(v, eta_i, beta)])  # compute objective and gradient
    
    dJ_du = dJ_du_0 + dJ_du_1 + dJ_du_2
    
    f0 = f0_0 + f0_1 + f0_2

    
    # Adjoint gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, np.sum(dJ_du, axis=1)
        )  # backprop
        
        
    evaluation_history_0.append(np.real(f0_0))
    evaluation_history_1.append(np.real(f0_1))
    evaluation_history_2.append(np.real(f0_2))
    evaluation_history.append(np.real(f0))

    cur_iter[0] = cur_iter[0] + 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    

    return np.real(f0)



algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny  # number of parameters

# Initial guess - 초기 시작값 랜덤
x = np.random.rand((n))

# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 10
update_factor = 20  # number of iterations between beta updates
ftol = 1e-5 

for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration



# FoM plot

plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("FoMresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

plt.figure()

plt.plot(evaluation_history_0, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("FoMresult_0.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

plt.figure()

plt.plot(evaluation_history_1, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("FoMresult_1.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


plt.figure()

plt.plot(evaluation_history_2, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("FoMresult_2.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

# Last design plot

plt.imshow(npa.rot90(design_variables.weights.reshape(Nx, Ny)), cmap='binary')
plt.colorbar()

np.savetxt("lastdesign.txt", design_variables.weights)
plt.savefig("lastdesign.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure



# focal point 에 잘 모였는지 확인


plt.figure(figsize=(20, 8))

intensities_R = np.abs(opt0.get_objective_arguments()[0][:,0]) ** 2
plt.subplot(1,3,1)
plt.plot(wavelengths/um_scale, intensities_R, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_G = np.abs(opt0.get_objective_arguments()[3][:,1]) ** 2
plt.subplot(1,3,2)
plt.plot(wavelengths/um_scale, intensities_G, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_B = np.abs(opt0.get_objective_arguments()[6][:,1]) ** 2
plt.subplot(1,3,3)
plt.plot(wavelengths/um_scale, intensities_B, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()
plt.savefig("FinalEz.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure
