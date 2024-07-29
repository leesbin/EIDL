# Design CIS 3D color router
# Resoultion 20 
# RGGB pattern
# First FoM: -22.775187623735164
# Current FoM: 264.1704627386973

# Elapsed run time = 57921.2683 s





# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import os
import math

mp.verbosity(1)

seed = 240  # 난수 발생 시드(seed)를 240으로 설정 (재현성을 위해 난수 시드를 고정)
np.random.seed(seed)  # numpy의 난수 생성기의 시드를 설정하여 난수의 재현성 보장

design_dir = "./CIS_design/"

# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

# scaling & refractive index
um_scale = 1/0.675 # 1A = 675nm

Air = mp.Medium(index=1.0)
SiN = mp.Medium(index=2.066715)
SiO2 = mp.Medium(index=1.463285)
HfO2 = mp.Medium(index=1.91365)
SiPD = mp.Medium(epsilon=4.676095)

# 해상도 및 사이즈 설정
resolution = 20 # 1 pixel = 27nm
ar_thk = 0.081 * um_scale # AR thickness
fl_size = 0.513 * um_scale # focal layer size
ml_size = 0.32 # 0.216 * um_scale # multi layer size
el_size = 0.04 # 0.027 * um_scale # etch layer size

dti_thk = ar_thk * 2 # DTI thickness 
sp_size = 0.621 * um_scale # subpixel size
# sp_size = 2 * um_scale # SiPD size

Lpml = 0.4 # PML 영역 크기
pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]
Sourcespace = 0.4

# 설계 공간
design_region_width_x = sp_size * 4 + 0.027 * um_scale * 2 # 디자인 영역 x
design_region_width_y = sp_size * 4 + 0.027 * um_scale * 2 # 디자인 영역 y
design_region_height = ml_size * 5 + el_size * 4 # 디자인 영역 높이 z

# 전체 공간
Sx = design_region_width_x * 2 * 4
Sy = design_region_width_y * 2 * 4
Sz = (Lpml + ar_thk + fl_size + design_region_height + Sourcespace + Lpml +0.04) * 2 * 2
cell_size = mp.Vector3(Sx, Sy, Sz)

# 파장, 주파수 설정
wavelengths = np.linspace(0.425*um_scale, 0.655*um_scale, 24) 
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies

# Fabrication Constraints 설정

minimum_length = 0.027 * um_scale # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)


# source 설정
width = 0.4

fcen_red = 1/(0.625*um_scale)
fwidth_red = fcen_red * width

fcen_green = 1/(0.545*um_scale)
fwidth_green = fcen_green * width

fcen_blue = 1/(0.455*um_scale)
fwidth_blue = fcen_blue * width

src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)

src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=fwidth_green, is_integrated=True)

src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=fwidth_blue, is_integrated=True)

source_center = [0, 0, Sz / 2 - Lpml - Sourcespace / 2 ] # Source 위치
source_size = mp.Vector3(Sx, Sy, 0)

source = [mp.Source(src_0, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_0, component=mp.Ey, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_1, component=mp.Ey, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_2, component=mp.Ey, size=source_size, center=source_center,)]


# 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
Nx = int(round(design_region_resolution * design_region_width_x)/2) 
Ny = int(round(design_region_resolution * design_region_width_y)/2) 
Nz = int(round(design_region_resolution * design_region_height))

# 설계 영역과 물질을 바탕으로 설계 영역 설정
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height/2),
        size=mp.Vector3(design_region_width_x, design_region_width_y, design_region_height),
    ),
)

# 대각선대칭

def mapping(x, eta, beta):
    x_copy = (x.reshape(Nx * Ny, Nz)).transpose()

    # projection
    z = 0
    x2 = []

    while z < Nz:
        
        number = z
        x1 = x_copy[int(number)]
        
        # filter
        filtered_field = mpa.conic_filter(
            x1,
            filter_radius,
            1.84,
            1.84,
            design_region_resolution,
        )
        
        z_slice = ((filtered_field.reshape(Nx, Ny)) + filtered_field.reshape(Nx, Ny).transpose()) / 2
        x2 = npa.concatenate((x2,z_slice.flatten()),axis=0) 
        z = z + 1

    x2 = ((x2.reshape(Nz,Nx*Ny)).transpose()).flatten()
    
    x2 = (x2.reshape(Nx * Ny, Nz)).transpose()
        
    # Compute the mean for the pairs of columns
    mean_columns_1layer = npa.mean(x2[:8, :], axis=0)  # Mean for columns 
    mean_columns_2layer = npa.mean(x2[9:17, :], axis=0)  # Mean for columns 
    mean_columns_3layer = npa.mean(x2[18:26, :], axis=0)  # Mean for columns 
    mean_columns_4layer = npa.mean(x2[27:35, :], axis=0)  # Mean for columns 
    mean_columns_5layer = npa.mean(x2[36:44, :], axis=0)  # Mean for columns 
    
    # Create new arrays with mean values
    new_x2_1layer = npa.tile(mean_columns_1layer, 8).reshape(8, -1).flatten()
    new_x2_1layer_e = npa.zeros(Nx * Ny).flatten()
    new_x2_2layer = npa.tile(mean_columns_2layer, 8).reshape(8, -1).flatten()
    new_x2_2layer_e = npa.zeros(Nx * Ny).flatten()
    new_x2_3layer = npa.tile(mean_columns_3layer, 8).reshape(8, -1).flatten()
    new_x2_3layer_e = npa.zeros(Nx * Ny).flatten()
    new_x2_4layer = npa.tile(mean_columns_4layer, 8).reshape(8, -1).flatten()
    new_x2_4layer_e = npa.zeros(Nx * Ny).flatten()
    new_x2_5layer = npa.tile(mean_columns_5layer, 8).reshape(8, -1).flatten()
    
    # Concatenate the arrays to get the final result
    x2 = npa.concatenate([new_x2_1layer, new_x2_1layer_e,new_x2_2layer, new_x2_2layer_e,new_x2_3layer, new_x2_3layer_e,new_x2_4layer, new_x2_4layer_e,new_x2_5layer],axis=0)
    x2 = ((x2.reshape(Nz,Nx*Ny)).transpose()).flatten()
    x2 = mpa.tanh_projection(x2, beta, eta).flatten()
    x = x2

    return x


# design region과 동일한 size의 Block 생성
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),

    # Focal Layer
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height - fl_size/2), size=mp.Vector3(Sx, Sy, fl_size), material=SiO2
    ),

    #AR coating
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height - fl_size - ar_thk/2), size=mp.Vector3(Sx, Sy, ar_thk), material=SiO2
    ),
    
    #Si
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height - fl_size - ar_thk - Lpml/2), size=mp.Vector3(Sx, Sy, Lpml), material=SiO2
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
    k_point = mp.Vector3(0,0,0)
)

###############################################################################################################################
# ## 2. Optimization Environment

# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml + 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_3, monitor_size_3 = mp.Vector3(design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml + 0.5/resolution), mp.Vector3(0.01,0.01,0) 

# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함

FourierFields_0_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ex,yee_grid=True)

FourierFields_1_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ex,yee_grid=True)

FourierFields_2_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ex,yee_grid=True)

FourierFields_3_x = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ex,yee_grid=True)

FourierFields_0_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)

FourierFields_1_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)

FourierFields_2_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)

FourierFields_3_y = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)

ob_list = [FourierFields_0_x, FourierFields_0_y, FourierFields_1_x,FourierFields_1_y, FourierFields_2_x, FourierFields_2_y, FourierFields_3_x, FourierFields_3_y]


fred = []
fgreen = []
fblue = []
# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 E 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
# [frequency index, moniter index]
def J(fields_0_x, fields_0_y, fields_1_x, fields_1_y, fields_2_x, fields_2_y, fields_3_x, fields_3_y):
    red = npa.sum(npa.abs(fields_3_x[17:23,1]) ** 2)  + npa.sum(npa.abs(fields_3_y[17:23,1]) ** 2)
    green = npa.sum(npa.abs(fields_0_x[9:15,1]) ** 2) + npa.sum(npa.abs(fields_0_y[9:15,1]) ** 2) + npa.sum(npa.abs(fields_2_x[9:15,1]) ** 2) + npa.sum(npa.abs(fields_2_y[9:15,1]) ** 2)
    blue = npa.sum(npa.abs(fields_1_x[0:6,1]) ** 2) + npa.sum(npa.abs(fields_1_y[0:6,1]) ** 2)
    redfactor = 0.5
    greenfactor = 0.7
    bluefactor = 0.6
    
    red_c = npa.sum(npa.abs(fields_3_x[0:6,1]) ** 2)  + npa.sum(npa.abs(fields_3_y[0:6,1]) ** 2) + npa.sum(npa.abs(fields_3_x[9:15,1]) ** 2)  + npa.sum(npa.abs(fields_3_y[9:15,1]) ** 2)
    green_c = npa.sum(npa.abs(fields_0_x[0:6,1]) ** 2) + npa.sum(npa.abs(fields_0_y[0:6,1]) ** 2) + npa.sum(npa.abs(fields_2_x[0:6,1]) ** 2) + npa.sum(npa.abs(fields_2_y[0:6,1]) ** 2) + npa.sum(npa.abs(fields_0_x[17:23,1]) ** 2) + npa.sum(npa.abs(fields_0_y[17:23,1]) ** 2) + npa.sum(npa.abs(fields_2_x[17:23,1]) ** 2) + npa.sum(npa.abs(fields_2_y[17:23,1]) ** 2)
    blue_c = npa.sum(npa.abs(fields_1_x[9:15,1]) ** 2) + npa.sum(npa.abs(fields_1_y[9:15,1]) ** 2) + npa.sum(npa.abs(fields_1_x[17:23,1]) ** 2) + npa.sum(npa.abs(fields_1_y[17:23,1]) ** 2)
    
    fred.append(red/redfactor)
    fgreen.append(green/greenfactor)
    fblue.append(blue/bluefactor)
    OE = blue/bluefactor + green/greenfactor + red/redfactor
    CT = blue_c/bluefactor + green_c/greenfactor + red_c/redfactor
    return OE - CT

# optimization 설정

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-2,
)

# 함수 설정

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
            v, eta_i, beta, np.sum(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))

    np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights)
    
    numevl += 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))

    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)

###############################################################################################################################

#
print('forward_run')
opt.forward_run()
opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)),
           # source_parameters={'alpha':0},
           #     boundary_parameters={'alpha':0},
          )
# plt.axis("off")
plt.savefig(design_dir+"forward_run.png")
