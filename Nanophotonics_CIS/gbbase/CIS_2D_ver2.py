# Design CIS 2D color router
# Resoultion 25 

# ## 1. Simulation Environment

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
import math
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
Sourcespace = 1



# 전체 공간
Sx = design_region_width * 5
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

beam_x0 = mp.Vector3(0, Sourcespace/2)  # beam focus (relative to source center)
rot_angle = 0  # CCW rotation angle about z axis (0: +y axis)
beam_kdir = mp.Vector3(0, -1, 0).rotate(
    mp.Vector3(0, 0, 1), math.radians(rot_angle)
)  # beam propagation direction
beam_w0 = 1.5  # beam waist radius
beam_E0 = mp.Vector3(0, 0, 1)

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

source = [
    mp.GaussianBeamSource(
        src=src_0,
        center=source_center,
        size=source_size,
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    ),
    mp.GaussianBeamSource(
        src=src_1,
        center=source_center,
        size=source_size,
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    ),
    mp.GaussianBeamSource(
        src=src_2,
        center=source_center,
        size=source_size,
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
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
        center=mp.Vector3(0, - Sy /2 + PDsize + design_region_height / 2, 0),
        size=mp.Vector3(design_region_width-air_gap*2, design_region_height, 0),
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
    # design region
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(Sx/5, 0, 0), size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(2*Sx/5, 0, 0), size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(-Sx/5, 0, 0), size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(-2*Sx/5, 0, 0), size=design_region.size, material=design_variables
    ),

    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(Sx, PDsize, 0), material=SiO2
    ),

    # DTI가 있을 경우 사용
    mp.Block(
        center=mp.Vector3(-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),

    mp.Block(
        center=mp.Vector3(Sx/5-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(Sx/5, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(Sx/5+design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),

    mp.Block(
        center=mp.Vector3(2*Sx/5-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(2*Sx/5, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(2*Sx/5+design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),

    mp.Block(
        center=mp.Vector3(-Sx/5-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(-Sx/5, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(-Sx/5+design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),

    mp.Block(
        center=mp.Vector3(-2*Sx/5-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(-2*Sx/5, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(-2*Sx/5+design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, 2, 0), material=SiPD
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
    
    redfactor = 4.5
    greenfactor = 2.9
    bluefactor = 1.7
    
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

    cur_iter[0] = cur_iter[0] + 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    

    return np.real(f0)

###############################################################################################################################

# ## 3. Algorithm select


algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny  # number of parameters

# Initial guess - 초기 시작값 랜덤
x = np.random.uniform(0.45, 0.55, n)

# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 7
update_factor = 20  # number of iterations between beta updates
ftol = 1e-5
for iters in range(num_betas):
    print("current beta: ", cur_beta)
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    if cur_beta >=128:
        solver.set_max_objective(lambda a, g: f(a, g, mp.inf))
        solver.set_maxeval(1) # Set the maximum number of function evaluations
    else:
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
   # solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration

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
