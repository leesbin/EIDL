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

seed = 25  # 난수 발생 시드(seed)를 240으로 설정 (재현성을 위해 난수 시드를 고정)
np.random.seed(seed)  # numpy의 난수 생성기의 시드를 설정하여 난수의 재현성 보장

design_dir = "./CIS_design/"

# 디렉터리가 없으면 생성
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

# scaling & refractive index
um_scale = 2

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
TiO2 = mp.Medium(epsilon=7)
SiPD = mp.Medium(epsilon=5)

# 설계 공간
design_region_width = 3 # 디자인 영역 너비
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
    # x = x.flatten()
    
    # filtered_field = mpa.conic_filter(
    #     x,
    #     filter_radius,
    #     design_region_width,
    #     design_region_height,
    #     design_region_resolution,
    # )

    # projection
    # 출력값 0 ~ 1으로 제한
    # projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return x


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



# Meep simulation 세팅
sim = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = mp.Vector3(0,0,0) # bloch boundary
)

###############################################################################################################################
# ## 2. Optimization Environment


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0)


# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2= mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)

ob_list = [FourierFields_0, FourierFields_1, FourierFields_2,]



fred = []
fgreen = []
fblue = []
# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취함
# [frequency index, moniter index]
def J_0(fields_0, fields_1, fields_2):
    red = npa.sum(npa.abs(fields_0[21:30,1]) ** 2)
    green = npa.sum(npa.abs(fields_1[11:20,1]) ** 2) 
    blue = npa.sum(npa.abs(fields_2[1:10,1]) ** 2) 
    
    redfactor = 2.9
    greenfactor = 2.1
    bluefactor = 1.1
    
    fred.append(red/redfactor)
    fgreen.append(green/greenfactor)
    fblue.append(blue/bluefactor)
    return blue/bluefactor + green/greenfactor + red/redfactor


# optimization 설정
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J_0],
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
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )  # backprop

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


algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

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
    for i in range(0, len(lst), 4):
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

intensities_R = np.abs(opt.get_objective_arguments()[0][:,0]) ** 2
plt.subplot(1,3,1)
plt.plot(wavelengths/um_scale, intensities_R, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_G = np.abs(opt.get_objective_arguments()[1][:,1]) ** 2
plt.subplot(1,3,2)
plt.plot(wavelengths/um_scale, intensities_G, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_B = np.abs(opt.get_objective_arguments()[2][:,1]) ** 2
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
