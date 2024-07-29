# Design CIS 2D color router
# Resoultion 25 

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

# normalzation ############################################################################################################
norm = 1
###############################################################################################################################

seed = 25  # 난수 발생 시드(seed)를 240으로 설정 (재현성을 위해 난수 시드를 고정)
np.random.seed(seed)  # numpy의 난수 생성기의 시드를 설정하여 난수의 재현성 보장

design_dir = "./Plane_focus_EH_gray_norm/"

# 디렉터리가 없으면 생성
try:
    if not os.path.exists(design_dir):
        os.makedirs(design_dir)
        print("디렉토리 생성 성공:", design_dir)
    else:
        print("디렉토리 이미 존재합니다:", design_dir)
except OSError as e:
    print("디렉토리 생성 실패:", design_dir, "| 에러:", e)

# scaling & refractive index
um_scale = 2

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
TiO2 = mp.Medium(epsilon=7)
SiPD = mp.Medium(epsilon=5)

# 설계 공간
design_region_width = 9 # 디자인 영역 너비
design_region_height = 4 # 디자인 영역 높이

# 해상도 및 사이즈 설정
resolution = 50
gapop = 0 ####################################################################################################
air_gap = 0
dti = 0.4
subpixelsize = design_region_width/3 - dti
if gapop == 1:
    air_gap = dti/2
PDsize = 2
Lpml = 1 # PML 영역 크기
pml_layers = [mp.PML(thickness = Lpml, direction = mp.Y)]
Sourcespace = 2



# 전체 공간
Sx = design_region_width
Sy = PDsize + design_region_height + Sourcespace + Lpml
cell_size = mp.Vector3(Sx, Sy)

# 파장, 주파수 설정
wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# Fabrication Constraints 설정

minimum_length = 0.02  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)



# Source 설정

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

source_center = mp.Vector3(0, Sy/ 2 - Lpml - Sourcespace / 2, 0) # Source 위치
source_size = mp.Vector3(Sx, 0, 0)

source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center,),]




# 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
Nx = int(round(design_region_resolution * design_region_width)) + 1
Ny = int(round(design_region_resolution * design_region_height)) + 1


# 설계 영역과 물질을 바탕으로 설계 영역 설정
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, - Sy /2 + PDsize + design_region_height / 2, 0),
        size=mp.Vector3(design_region_width-air_gap*2, design_region_height, 0),
    ),
)



# filter.py conic_filter 함수와 simple_2d_filter 함수를 사용
def mapping(x, eta, beta):
    # filter
    x = x.flatten()
    
    # filtered_field = mpa.conic_filter(
    #     x,
    #     filter_radius,
    #     design_region_width,
    #     design_region_height,
    #     design_region_resolution,
    # )

    # # projection
    # # 출력값 0 ~ 1으로 제한
    # projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return x.flatten()


# design region과 동일한 size의 Block 생성
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(Sx, PDsize, 0), material=SiO2
    ),
    # DTI가 있을 경우 사용
    mp.Block(
        center=mp.Vector3(-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    )
]


if norm == 1:
# normalization flux calculation

    sim = mp.Simulation(
        cell_size=cell_size, 
        boundary_layers=pml_layers,
        sources=source,
        default_material=Air, # 빈공간
        resolution=resolution,
        k_point = mp.Vector3(0,0,0)
    )

    tran_t = mp.FluxRegion(
        center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0)
    )

    b_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[3], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[4], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[5], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[6], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    b_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    b_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[7], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[13], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[14], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[15], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[16], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    g_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    g_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[17], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_0_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_0_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_0_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_0_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_0_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[20], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_1_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_1_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_1_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_1_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_1_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[24], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_2_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_2_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_2_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_2_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_2_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[25], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_3_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_3_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_3_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_3_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_3_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_3_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[26], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_4_tran_Ex = sim.add_dft_fields([mp.Ex], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_4_tran_Ey = sim.add_dft_fields([mp.Ey], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_4_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)

    r_4_tran_Hx = sim.add_dft_fields([mp.Hx], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_4_tran_Hy = sim.add_dft_fields([mp.Hy], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    r_4_tran_Hz = sim.add_dft_fields([mp.Hz], frequencies[27], 0 , 1, center=mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), size=mp.Vector3(Sx, 0, 0), yee_grid=True)


    #################################################################################################################################################################################

    sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-9))

    b_0_dft_array_Ex = sim.get_dft_array(b_0_tran_Ex, mp.Ex, 0)
    b_0_dft_array_Ey = sim.get_dft_array(b_0_tran_Ey, mp.Ey, 0)
    b_0_dft_array_Ez = sim.get_dft_array(b_0_tran_Ez, mp.Ez, 0)

    b_0_dft_array_Hx = sim.get_dft_array(b_0_tran_Hx, mp.Hx, 0)
    b_0_dft_array_Hy = sim.get_dft_array(b_0_tran_Hy, mp.Hy, 0)
    b_0_dft_array_Hz = sim.get_dft_array(b_0_tran_Hz, mp.Hz, 0)

    b_0_dft_total = npa.sum(b_0_dft_array_Ez*np.conj(b_0_dft_array_Hx)) 

    b_1_dft_array_Ex = sim.get_dft_array(b_1_tran_Ex, mp.Ex, 0)
    b_1_dft_array_Ey = sim.get_dft_array(b_1_tran_Ey, mp.Ey, 0)
    b_1_dft_array_Ez = sim.get_dft_array(b_1_tran_Ez, mp.Ez, 0)

    b_1_dft_array_Hx = sim.get_dft_array(b_1_tran_Hx, mp.Hx, 0)
    b_1_dft_array_Hy = sim.get_dft_array(b_1_tran_Hy, mp.Hy, 0)
    b_1_dft_array_Hz = sim.get_dft_array(b_1_tran_Hz, mp.Hz, 0)

    b_1_dft_total = npa.sum(b_1_dft_array_Ez*np.conj(b_1_dft_array_Hx)) 

    b_2_dft_array_Ex = sim.get_dft_array(b_2_tran_Ex, mp.Ex, 0)
    b_2_dft_array_Ey = sim.get_dft_array(b_2_tran_Ey, mp.Ey, 0)
    b_2_dft_array_Ez = sim.get_dft_array(b_2_tran_Ez, mp.Ez, 0)

    b_2_dft_array_Hx = sim.get_dft_array(b_2_tran_Hx, mp.Hx, 0)
    b_2_dft_array_Hy = sim.get_dft_array(b_2_tran_Hy, mp.Hy, 0)
    b_2_dft_array_Hz = sim.get_dft_array(b_2_tran_Hz, mp.Hz, 0)

    b_2_dft_total = npa.sum(b_2_dft_array_Ez*np.conj(b_2_dft_array_Hx)) 

    b_3_dft_array_Ex = sim.get_dft_array(b_3_tran_Ex, mp.Ex, 0)
    b_3_dft_array_Ey = sim.get_dft_array(b_3_tran_Ey, mp.Ey, 0)
    b_3_dft_array_Ez = sim.get_dft_array(b_3_tran_Ez, mp.Ez, 0)

    b_3_dft_array_Hx = sim.get_dft_array(b_3_tran_Hx, mp.Hx, 0)
    b_3_dft_array_Hy = sim.get_dft_array(b_3_tran_Hy, mp.Hy, 0)
    b_3_dft_array_Hz = sim.get_dft_array(b_3_tran_Hz, mp.Hz, 0)

    b_3_dft_total = npa.sum(b_3_dft_array_Ez*np.conj(b_3_dft_array_Hx)) 

    b_4_dft_array_Ex = sim.get_dft_array(b_4_tran_Ex, mp.Ex, 0)
    b_4_dft_array_Ey = sim.get_dft_array(b_4_tran_Ey, mp.Ey, 0)
    b_4_dft_array_Ez = sim.get_dft_array(b_4_tran_Ez, mp.Ez, 0)

    b_4_dft_array_Hx = sim.get_dft_array(b_4_tran_Hx, mp.Hx, 0)
    b_4_dft_array_Hy = sim.get_dft_array(b_4_tran_Hy, mp.Hy, 0)
    b_4_dft_array_Hz = sim.get_dft_array(b_4_tran_Hz, mp.Hz, 0)

    b_4_dft_total = npa.sum(b_4_dft_array_Ez*np.conj(b_4_dft_array_Hx))

    g_0_dft_array_Ex = sim.get_dft_array(g_0_tran_Ex, mp.Ex, 0)
    g_0_dft_array_Ey = sim.get_dft_array(g_0_tran_Ey, mp.Ey, 0)
    g_0_dft_array_Ez = sim.get_dft_array(g_0_tran_Ez, mp.Ez, 0)

    g_0_dft_array_Hx = sim.get_dft_array(g_0_tran_Hx, mp.Hx, 0)
    g_0_dft_array_Hy = sim.get_dft_array(g_0_tran_Hy, mp.Hy, 0)
    g_0_dft_array_Hz = sim.get_dft_array(g_0_tran_Hz, mp.Hz, 0)

    g_0_dft_total = npa.sum(g_0_dft_array_Ez*np.conj(g_0_dft_array_Hx))

    g_1_dft_array_Ex = sim.get_dft_array(g_1_tran_Ex, mp.Ex, 0)
    g_1_dft_array_Ey = sim.get_dft_array(g_1_tran_Ey, mp.Ey, 0)
    g_1_dft_array_Ez = sim.get_dft_array(g_1_tran_Ez, mp.Ez, 0)

    g_1_dft_array_Hx = sim.get_dft_array(g_1_tran_Hx, mp.Hx, 0)
    g_1_dft_array_Hy = sim.get_dft_array(g_1_tran_Hy, mp.Hy, 0)
    g_1_dft_array_Hz = sim.get_dft_array(g_1_tran_Hz, mp.Hz, 0)

    g_1_dft_total = npa.sum(g_1_dft_array_Ez*np.conj(g_1_dft_array_Hx))

    g_2_dft_array_Ex = sim.get_dft_array(g_2_tran_Ex, mp.Ex, 0)
    g_2_dft_array_Ey = sim.get_dft_array(g_2_tran_Ey, mp.Ey, 0)
    g_2_dft_array_Ez = sim.get_dft_array(g_2_tran_Ez, mp.Ez, 0)

    g_2_dft_array_Hx = sim.get_dft_array(g_2_tran_Hx, mp.Hx, 0)
    g_2_dft_array_Hy = sim.get_dft_array(g_2_tran_Hy, mp.Hy, 0)
    g_2_dft_array_Hz = sim.get_dft_array(g_2_tran_Hz, mp.Hz, 0)

    g_2_dft_total = npa.sum(g_2_dft_array_Ez*np.conj(g_2_dft_array_Hx))

    g_3_dft_array_Ex = sim.get_dft_array(g_3_tran_Ex, mp.Ex, 0)
    g_3_dft_array_Ey = sim.get_dft_array(g_3_tran_Ey, mp.Ey, 0)
    g_3_dft_array_Ez = sim.get_dft_array(g_3_tran_Ez, mp.Ez, 0)

    g_3_dft_array_Hx = sim.get_dft_array(g_3_tran_Hx, mp.Hx, 0)
    g_3_dft_array_Hy = sim.get_dft_array(g_3_tran_Hy, mp.Hy, 0)
    g_3_dft_array_Hz = sim.get_dft_array(g_3_tran_Hz, mp.Hz, 0)

    g_3_dft_total = npa.sum(g_3_dft_array_Ez*np.conj(g_3_dft_array_Hx))

    g_4_dft_array_Ex = sim.get_dft_array(g_4_tran_Ex, mp.Ex, 0)
    g_4_dft_array_Ey = sim.get_dft_array(g_4_tran_Ey, mp.Ey, 0)
    g_4_dft_array_Ez = sim.get_dft_array(g_4_tran_Ez, mp.Ez, 0)

    g_4_dft_array_Hx = sim.get_dft_array(g_4_tran_Hx, mp.Hx, 0)
    g_4_dft_array_Hy = sim.get_dft_array(g_4_tran_Hy, mp.Hy, 0)
    g_4_dft_array_Hz = sim.get_dft_array(g_4_tran_Hz, mp.Hz, 0)

    g_4_dft_total = npa.sum(g_4_dft_array_Ez*np.conj(g_4_dft_array_Hx))

    r_0_dft_array_Ex = sim.get_dft_array(r_0_tran_Ex, mp.Ex, 0)
    r_0_dft_array_Ey = sim.get_dft_array(r_0_tran_Ey, mp.Ey, 0)
    r_0_dft_array_Ez = sim.get_dft_array(r_0_tran_Ez, mp.Ez, 0)

    r_0_dft_array_Hx = sim.get_dft_array(r_0_tran_Hx, mp.Hx, 0)
    r_0_dft_array_Hy = sim.get_dft_array(r_0_tran_Hy, mp.Hy, 0)
    r_0_dft_array_Hz = sim.get_dft_array(r_0_tran_Hz, mp.Hz, 0)

    r_0_dft_total = npa.sum(r_0_dft_array_Ez*np.conj(r_0_dft_array_Hx))

    r_1_dft_array_Ex = sim.get_dft_array(r_1_tran_Ex, mp.Ex, 0)
    r_1_dft_array_Ey = sim.get_dft_array(r_1_tran_Ey, mp.Ey, 0)
    r_1_dft_array_Ez = sim.get_dft_array(r_1_tran_Ez, mp.Ez, 0)

    r_1_dft_array_Hx = sim.get_dft_array(r_1_tran_Hx, mp.Hx, 0)
    r_1_dft_array_Hy = sim.get_dft_array(r_1_tran_Hy, mp.Hy, 0)
    r_1_dft_array_Hz = sim.get_dft_array(r_1_tran_Hz, mp.Hz, 0)

    r_1_dft_total = npa.sum(r_1_dft_array_Ez*np.conj(r_1_dft_array_Hx))

    r_2_dft_array_Ex = sim.get_dft_array(r_2_tran_Ex, mp.Ex, 0)
    r_2_dft_array_Ey = sim.get_dft_array(r_2_tran_Ey, mp.Ey, 0)
    r_2_dft_array_Ez = sim.get_dft_array(r_2_tran_Ez, mp.Ez, 0)

    r_2_dft_array_Hx = sim.get_dft_array(r_2_tran_Hx, mp.Hx, 0)
    r_2_dft_array_Hy = sim.get_dft_array(r_2_tran_Hy, mp.Hy, 0)
    r_2_dft_array_Hz = sim.get_dft_array(r_2_tran_Hz, mp.Hz, 0)

    r_2_dft_total = npa.sum(r_2_dft_array_Ez*np.conj(r_2_dft_array_Hx))

    r_3_dft_array_Ex = sim.get_dft_array(r_3_tran_Ex, mp.Ex, 0)
    r_3_dft_array_Ey = sim.get_dft_array(r_3_tran_Ey, mp.Ey, 0)
    r_3_dft_array_Ez = sim.get_dft_array(r_3_tran_Ez, mp.Ez, 0)

    r_3_dft_array_Hx = sim.get_dft_array(r_3_tran_Hx, mp.Hx, 0)
    r_3_dft_array_Hy = sim.get_dft_array(r_3_tran_Hy, mp.Hy, 0)
    r_3_dft_array_Hz = sim.get_dft_array(r_3_tran_Hz, mp.Hz, 0)

    r_3_dft_total = npa.sum(r_3_dft_array_Ez*np.conj(r_3_dft_array_Hx))

    r_4_dft_array_Ex = sim.get_dft_array(r_4_tran_Ex, mp.Ex, 0)
    r_4_dft_array_Ey = sim.get_dft_array(r_4_tran_Ey, mp.Ey, 0)
    r_4_dft_array_Ez = sim.get_dft_array(r_4_tran_Ez, mp.Ez, 0)

    r_4_dft_array_Hx = sim.get_dft_array(r_4_tran_Hx, mp.Hx, 0)
    r_4_dft_array_Hy = sim.get_dft_array(r_4_tran_Hy, mp.Hy, 0)
    r_4_dft_array_Hz = sim.get_dft_array(r_4_tran_Hz, mp.Hz, 0)

    r_4_dft_total = npa.sum(r_4_dft_array_Ez*np.conj(r_4_dft_array_Hx))

# Meep simulation 세팅

sim.reset_meep()

sim = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = mp.Vector3(0,0,0) # bloch boundary
)

# print(b_0_dft_total, b_1_dft_total, b_2_dft_total, b_3_dft_total, b_4_dft_total, g_0_dft_total, g_1_dft_total, g_2_dft_total, g_3_dft_total, g_4_dft_total, r_0_dft_total, r_1_dft_total, r_2_dft_total, r_3_dft_total, r_4_dft_total)

###############################################################################################################################
# ## 2. Optimization Environment


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize/2,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize/2,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize/2,0)


# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2= mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)

FourierFields_0_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Hx,yee_grid=True)

FourierFields_1_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Hx,yee_grid=True)

FourierFields_2_h = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Hx,yee_grid=True)

ob_list = [FourierFields_0, FourierFields_1, FourierFields_2, FourierFields_0_h, FourierFields_1_h, FourierFields_2_h]



fred = []
fgreen = []
fblue = []
# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취함
# [frequency index, moniter index]
def J(fields_0, fields_1, fields_2, fields_0_h, fields_1_h, fields_2_h):
    red_0 = npa.sum(fields_0[23,:]*(npa.real(fields_0_h[23,:])-npa.imag(fields_0_h[23,:])*1j)) 
    red_1 = npa.sum(fields_0[24,:]*(npa.real(fields_0_h[24,:])-npa.imag(fields_0_h[24,:])*1j)) 
    red_2 = npa.sum(fields_0[25,:]*(npa.real(fields_0_h[25,:])-npa.imag(fields_0_h[25,:])*1j))
    red_3 = npa.sum(fields_0[26,:]*(npa.real(fields_0_h[26,:])-npa.imag(fields_0_h[26,:])*1j)) 
    red_4 = npa.sum(fields_0[27,:]*(npa.real(fields_0_h[27,:])-npa.imag(fields_0_h[27,:])*1j)) 

    green_0 = npa.sum(fields_1[13,:]*(npa.real(fields_1_h[13,:])-npa.imag(fields_1_h[13,:])*1j)) 
    green_1 = npa.sum(fields_1[14,:]*(npa.real(fields_1_h[14,:])-npa.imag(fields_1_h[14,:])*1j)) 
    green_2 = npa.sum(fields_1[15,:]*(npa.real(fields_1_h[15,:])-npa.imag(fields_1_h[15,:])*1j)) 
    green_3 = npa.sum(fields_1[16,:]*(npa.real(fields_1_h[16,:])-npa.imag(fields_1_h[16,:])*1j)) 
    green_4 = npa.sum(fields_1[17,:]*(npa.real(fields_1_h[17,:])-npa.imag(fields_1_h[17,:])*1j)) 

    blue_0 = npa.sum(fields_2[3,:]*(npa.real(fields_2_h[3,:])-npa.imag(fields_2_h[3,:])*1j)) 
    blue_1 = npa.sum(fields_2[4,:]*(npa.real(fields_2_h[4,:])-npa.imag(fields_2_h[4,:])*1j))
    blue_2 = npa.sum(fields_2[5,:]*(npa.real(fields_2_h[5,:])-npa.imag(fields_2_h[5,:])*1j))
    blue_3 = npa.sum(fields_2[6,:]*(npa.real(fields_2_h[6,:])-npa.imag(fields_2_h[6,:])*1j)) 
    blue_4 = npa.sum(fields_2[7,:]*(npa.real(fields_2_h[7,:])-npa.imag(fields_2_h[7,:])*1j)) 

    red = npa.real(red_0)/npa.real(r_0_dft_total) + npa.real(red_1)/npa.real(r_1_dft_total) + npa.real(red_2)/npa.real(r_2_dft_total)+ npa.real(red_3)/npa.real(r_3_dft_total) + npa.real(red_4)/npa.real(r_4_dft_total)
    green = npa.real(green_0)/npa.real(g_0_dft_total) + npa.real(green_1)/npa.real(g_1_dft_total) + npa.real(green_2)/npa.real(g_2_dft_total) + npa.real(green_3)/npa.real(g_3_dft_total) + npa.real(green_4)/npa.real(g_4_dft_total)
    blue = npa.real(blue_0)/npa.real(b_0_dft_total) + npa.real(blue_1)/npa.real(b_1_dft_total)  + npa.real(blue_2)/npa.real(b_2_dft_total)  + npa.real(blue_3)/npa.real(b_3_dft_total)  + npa.real(blue_4)/npa.real(b_4_dft_total) 

    fred.append(red)
    fgreen.append(green)
    fblue.append(blue)

    print("red: ", red, "green: ", green, "blue: ", blue)

    return blue + green + red


# optimization 설정
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1, # 모니터에 남아있는 필드 값의 비율
)


evaluation_history = []
cur_iter = [0]
numevl = 1
EPS_Step = 0.02
FoM_old = 0
FoM_oold = 0
FoM_cur = 0


def f(v, cur_beta):
    global numevl
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])  # compute objective and gradient
    # f0, dJ_du = opt()
    
    # Adjoint gradient
    if v.size > 0:
        gradient = v*0
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, npa.sum(dJ_du, axis=1)
        )  # backprop

    print(dJ_du,gradient)

    evaluation_history.append(np.real(f0))

    np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights)

    numevl += 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    
    dv = (EPS_Step)*((gradient)/(npa.clip(npa.max(npa.abs(gradient)), 1e-9, npa.inf)))
    #This is only for speacial case: Y mirror symmetry & EVEN number of designregion pixel
    Updated_v = v + dv
    Updated_v = npa.clip(Updated_v, 0, 1)
    cur_iter[0] = cur_iter[0] + 1
    X = v

    return np.real(f0), Updated_v, X

###############################################################################################################################

# ## 3. Algorithm select


n = Nx * Ny  # number of parameters

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

###############################################################################################################################


# ## 4. Save result

#np.save("evaluation", evaluation_history)
np.savetxt(design_dir+"evaluation.txt", evaluation_history)

# FoM plot

plt.figure()

plt.plot(evaluation_history, "k-")
plt.grid(False)
plt.tick_params(axis='x', direction='in', pad = 5)
plt.tick_params(axis='y', direction='in', pad = 10)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"FoMresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


def extract_elements(lst):
    # 결과를 저장할 리스트를 생성합니다.
    result = []

    # 리스트의 길이만큼 반복하면서 4의 배수 인덱스의 요소를 추출합니다.
    for i in range(0, len(lst), 7):
        result.append(lst[i])

    return result

# RGB FoM plot

fred = extract_elements(fred)
fgreen = extract_elements(fgreen)
fblue = extract_elements(fblue)

np.savetxt(design_dir+"fred.txt", fred)
np.savetxt(design_dir+"fgreen.txt", fgreen)
np.savetxt(design_dir+"fblue.txt", fblue)

plt.figure()

plt.plot(fred, "r-")
plt.plot(fgreen, "g-")
plt.plot(fblue, "b-")
plt.grid(False)
plt.tick_params(axis='x', direction='in', pad = 5)
plt.tick_params(axis='y', direction='in', pad = 10)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"FoMresultr.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure





# Last design plot

plt.imshow(npa.rot90(design_variables.weights.reshape(Nx, Ny)), cmap='binary')
plt.colorbar()

np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
plt.savefig(design_dir+"lastdesign.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure



# focal point 에 잘 모였는지 확인


plt.figure(figsize=(20, 8))

intensities_R = np.abs(opt.get_objective_arguments()[0][:]) ** 2
plt.subplot(1,3,1)
plt.plot(wavelengths/um_scale, intensities_R, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_G = np.abs(opt.get_objective_arguments()[1][:]) ** 2
plt.subplot(1,3,2)
plt.plot(wavelengths/um_scale, intensities_G, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_B = np.abs(opt.get_objective_arguments()[2][:]) ** 2
plt.subplot(1,3,3)
plt.plot(wavelengths/um_scale, intensities_B, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()
plt.savefig(design_dir+"FinalEz.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure
