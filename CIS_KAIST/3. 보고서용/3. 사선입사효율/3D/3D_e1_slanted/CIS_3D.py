#!/usr/bin/env python
# coding: utf-8

# In[1]:
#First FoM: 38.62378215463889
#Current FoM: 1020.9999958808755

#Elapsed run time = 13354.1881 s

# 시뮬레이션 라이브러리 불러오기16103.1462 s

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


# In[2]:


# 물질 설정 및 Refractive Index 설정
mp.verbosity(1)
um_scale = 2

seed = 240  # 난수 발생 시드(seed)를 240으로 설정 (재현성을 위해 난수 시드를 고정)
np.random.seed(seed)  # numpy의 난수 생성기의 시드를 설정하여 난수의 재현성 보장

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
SiPD = mp.Medium(epsilon=5)

# In[3]:


resolution = 25 # 해상도
design_region_width = 2 # 디자인 영역 x
design_region_width_y = 2 # 디자인 영역 y
design_region_height = 4 # 디자인 영역 높이 z
pml_size = 1.0 # PML 영역 크기


# In[4]:


# 시뮬레이션 공간 설정
Sx = design_region_width
Sy = design_region_width_y
Sz = 2 * pml_size + design_region_height + 2
cell_size = mp.Vector3(Sx, Sy, Sz)


# In[5]:


# 파장, 주파수 설정
wavelengths = np.array([0.43*um_scale, 0.45*um_scale, 0.47*um_scale, 0.53*um_scale, 0.55*um_scale, 0.57*um_scale, 0.62*um_scale, 0.65*um_scale, 0.68*um_scale])
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# In[6]:


minimum_length = 0.05  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)


# In[7]:


pml_layers = [mp.PML(thickness = pml_size, direction = mp.Z)]

rot_angle = np.radians(6)

rot_angle_n = np.radians(-6)

k_point_0 = mp.Vector3(z = -1).rotate(mp.Vector3(y=1), 0)

k_point = mp.Vector3(z = -1).rotate(mp.Vector3(y=1), rot_angle)

k_point_n = mp.Vector3(z = -1).rotate(mp.Vector3(y=1), rot_angle_n)


# In[8]:


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


# In[9]:


source_center = [0, 0, Sz / 2 - pml_size - 0.5] # Source 위치
source_size = mp.Vector3(Sx, Sy, 0)

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
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
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
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
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
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_1,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_2,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_3,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_4,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_5,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_6,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_7,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
	mp.EigenModeSource(
		    src=src_8,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle_n == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.ODD_Y,
		    eig_match_freq=True,
		),
		 ]


# 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
Nx = int(round(design_region_resolution * design_region_width)) + 1
Ny = int(round(design_region_resolution * design_region_width_y)) + 1
Nz = int(round(design_region_resolution * design_region_height)) + 1

# 설계 영역과 물질을 바탕으로 설계 영역 설정
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(design_region_width, design_region_width_y, design_region_height),
    ),
)


# In[11]:


# filter.py conic_filter 함수와 simple_2d_filter 함수를 사용
def mapping(x, eta, beta):
    # filter
    x=((x.reshape(Nx*Ny,Nz)).transpose()).flatten()
    
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        (Nx*Nz-1)/design_region_resolution,
        (Ny-1)/design_region_resolution,
        design_region_resolution,
    )

    # projection
    # 출력값 -1 ~ 1으로 제한
    
    x=((filtered_field.reshape(Nz,Nx*Ny)).transpose()).flatten()
    
    projected_field = mpa.tanh_projection(x, beta, eta)

    # interpolate to actual materials
    return projected_field.flatten()


# In[12]:


# design region과 동일한 size의 Block 생성
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(0, 0, -3), size=mp.Vector3(Sx, Sy, 2), material=SiPD
    ),
    # mp.Block(
    #     center=mp.Vector3(-Sx/4, Sy/4, -4), size=mp.Vector3(0.5, 0.5, 4), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(-Sx/4, -Sy/4, -4), size=mp.Vector3(0.5, 0.5, 4), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(Sx/4, -Sy/4, -4), size=mp.Vector3(0.5, 0.5, 4), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(Sx/4, Sy/4, -4), size=mp.Vector3(0.5, 0.5, 4), material=SiPD
    # )
]


# In[13]:


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


# In[14]:


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-Sx/4, Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(-Sx/4, -Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(Sx/4, -Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_3, monitor_size_3 = mp.Vector3(Sx/4, Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 

# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_1 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_2 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_3 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_4 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_5 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_6 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_7 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_8 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_9 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_10 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_11 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
ob_list_0 = [FourierFields_0, FourierFields_1, FourierFields_2,FourierFields_3, FourierFields_4, FourierFields_5 ,FourierFields_6, FourierFields_7, FourierFields_8,FourierFields_9, FourierFields_10, FourierFields_11]

FourierFields_01 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_11 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_21 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_31 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_41 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_51 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_61 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_71 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_81 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_91 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_101 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_111 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
ob_list_1 = [FourierFields_01, FourierFields_11, FourierFields_21,FourierFields_31, FourierFields_41, FourierFields_51 ,FourierFields_61, FourierFields_71, FourierFields_81,FourierFields_91, FourierFields_101, FourierFields_111]

FourierFields_02 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_12 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_22 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_32 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_42 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_52 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_62 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_72 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_82 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_92 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_102 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_112 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
ob_list_2 = [FourierFields_02, FourierFields_12, FourierFields_22,FourierFields_32, FourierFields_42, FourierFields_52 ,FourierFields_62, FourierFields_72, FourierFields_82,FourierFields_92, FourierFields_102, FourierFields_112]


# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
def J_0(fields_0, fields_1, fields_2,fields_3, fields_4, fields_5,fields_6, fields_7, fields_8,fields_9, fields_10, fields_11):
    return npa.mean(npa.abs(fields_0[6,:]) ** 2) + npa.mean(npa.abs(fields_1[7,:]) ** 2) + npa.mean(npa.abs(fields_2[8,:]) ** 2) + npa.mean(npa.abs(fields_3[3,:]) ** 2) + npa.mean(npa.abs(fields_4[4,:]) ** 2) + npa.mean(npa.abs(fields_5[5,:]) ** 2) + npa.mean(npa.abs(fields_6[0,:]) ** 2) + npa.mean(npa.abs(fields_7[1,:]) ** 2) + npa.mean(npa.abs(fields_8[2,:]) ** 2) + npa.mean(npa.abs(fields_9[3,:]) ** 2) + npa.mean(npa.abs(fields_10[4,:]) ** 2) + npa.mean(npa.abs(fields_11[5,:]) ** 2)




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
numevl = 1

def f(v, gradient, beta):
    global numevl
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


    np.savetxt("structure_0"+str(numevl) +".txt", design_variables.weights)

    
    numevl += 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))


    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)


# In[19]:


algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny * Nz  # number of parameters

# Initial guess - 초기 시작값 0.5
x = np.random.rand((n))

# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny * Nz))
ub = np.ones((Nx * Ny * Nz))

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
    # solver.set_max_objective(lambda a, g: f1(a, g, cur_beta), lambda a, g: f2(a, g, cur_beta), lambda a, g: f3(a, g, cur_beta)) # Set the objective function to be maximized
    solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration


#np.save("evaluation", evaluation_history)
np.savetxt("evaluation.txt", evaluation_history)

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



#np.save("lastdesign", design_variables.weights)
np.savetxt("lastdesign.txt", design_variables.weights)



# In[22]:



intensities_0 = np.abs(opt0.get_objective_arguments()[0][:,1]) ** 2
plt.figure(figsize=(10, 8))
plt.plot(wavelengths/um_scale, intensities_0, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
plt.savefig("x_0_Ez_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


intensities_1 = np.abs(opt0.get_objective_arguments()[4][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_1, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
plt.savefig("x_01_Ez_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_2 = np.abs(opt0.get_objective_arguments()[8][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_2, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
plt.savefig("x_02_Ez_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


intensities_3 = np.abs(opt0.get_objective_arguments()[9][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_3, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
plt.savefig("x_03_Ez_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

