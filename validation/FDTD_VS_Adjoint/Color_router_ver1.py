# Design CIS 3D color router
# Resoultion 25 
# RGGB pattern
# Elapsed run time = 33956.9432 s


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
SiN = mp.Medium(index=2.1)
SiO2 = mp.Medium(index=1.4)
HfO2 = mp.Medium(index=1.9)
# SiPD = mp.Medium(index=3.5)

# 해상도 및 사이즈 설정
resolution = 25 # 1 pixel = 27nm
ar_thk = 0.081 * um_scale # AR thickness
fl_size = 0.513 * um_scale # focal layer size
ml_size = 0.216 * um_scale # multi layer size
el_size = 0.027 * um_scale # etch layer size

dti_thk = ar_thk * 2 # DTI thickness 
sp_size = 0.621 * um_scale # subpixel size
# sp_size = 2 * um_scale # SiPD size

Lpml = 0.5 # PML 영역 크기
pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]
Sourcespace = 0.5

# 설계 공간
design_region_width_x = sp_size * 2 # 디자인 영역 x
design_region_width_y = sp_size * 2 # 디자인 영역 y
design_region_height = ml_size * 5 + el_size * 4 # 디자인 영역 높이 z

# 전체 공간
Sx = design_region_width_x
Sy = design_region_width_y
Sz = Lpml + ar_thk + fl_size + design_region_height + Sourcespace + Lpml
cell_size = mp.Vector3(Sx, Sy, Sz)

# 파장, 주파수 설정
wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies

# Fabrication Constraints 설정

minimum_length = 0.08  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
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

source_center = [0, 0, Sz / 2 - Lpml - Sourcespace / 2 ] # Source 위치
source_size = mp.Vector3(Sx, Sy, 0)

source = [mp.Source(src_0, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_0, component=mp.Ey, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_1, component=mp.Ey, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ex, size=source_size, center=source_center,),mp.Source(src_2, component=mp.Ey, size=source_size, center=source_center,)]


# 설계 영역의 픽셀 - 해상도와 디자인 영역에 따라 결정
Nx = int(round(design_region_resolution * design_region_width_x)/2)
Ny = int(round(design_region_resolution * design_region_width_y)/2) 
Nz = 5

# 설계 영역과 물질을 바탕으로 설계 영역 설정
design_variables_1 = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region_1 = mpa.DesignRegion(
    design_variables_1,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - ml_size/2),
        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_size),
    ),
)
design_variables_2 = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region_2 = mpa.DesignRegion(
    design_variables_2,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - ml_size/2 - (ml_size + el_size)),
        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_size),
    ),
)
design_variables_3 = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region_3 = mpa.DesignRegion(
    design_variables_3,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - ml_size/2 - (ml_size + el_size)*2),
        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_size),
    ),
)
design_variables_4 = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region_4 = mpa.DesignRegion(
    design_variables_4,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - ml_size/2 - (ml_size + el_size)*3),
        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_size),
    ),
)
design_variables_5 = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, SiN, grid_type="U_MEAN",do_averaging=False)
design_region_5 = mpa.DesignRegion(
    design_variables_5,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - ml_size/2 - (ml_size + el_size)*4),
        size=mp.Vector3(design_region_width_x, design_region_width_y, ml_size),
    ),
)




# 대각선대칭

def mapping(x, eta, beta):

    x_copy = (x.reshape(Nx * Ny, 5)).transpose()

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
            0.88,
            0.88,
            design_region_resolution,
        )
        
        z_slice = ((filtered_field.reshape(Nx, Ny)) + filtered_field.reshape(Nx, Ny).transpose()) / 2
        z_slice = mpa.tanh_projection(z_slice, beta, eta).flatten()
        x2 = npa.concatenate((x2,z_slice.flatten()),axis=0) 
        z = z + 1

    x2 = ((x2.reshape(Nz,Nx*Ny)).transpose()).flatten()
    x3 = (x2.reshape(Nx * Ny, 5)).transpose()
    x_0 = x3[0]
    x_1 = x3[1]
    x_2 = x3[2]
    x_3 = x3[3]
    x_4 = x3[4]


    return x_0, x_1, x_2, x_3, x_4


geometry = [
    # Design region
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
    mp.Block(
        center=design_region_5.center, size=design_region_5.size, material=design_variables_5
    ),
    
    # Etch Layer
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - el_size/2 - ml_size), size=mp.Vector3(Sx, Sy, el_size), material=SiO2
    ),
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - el_size/2 - ml_size * 2 - el_size), size=mp.Vector3(Sx, Sy, el_size), material=SiO2
    ),
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - el_size/2 - ml_size * 3 - el_size * 2), size=mp.Vector3(Sx, Sy, el_size), material=SiO2
    ),
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - el_size/2 - ml_size * 4 - el_size * 3), size=mp.Vector3(Sx, Sy, el_size), material=SiO2
    ),
    
    # Focal Layer
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height - fl_size/2), size=mp.Vector3(Sx, Sy, fl_size), material=SiO2
    ),

    #AR coating
    mp.Block(
        center=mp.Vector3(0, 0, Sz / 2 - Lpml - Sourcespace - design_region_height - fl_size - ar_thk/2), size=mp.Vector3(Sx, Sy, ar_thk), material=HfO2
    ),

    # # DTI가 있을 경우 사용
    # mp.Block(
    #     center=mp.Vector3(Sx/4, Sy/4, -Sz/2 + PDsize/2), size=mp.Vector3(subpixelsize, subpixelsize,  PDsize), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(-Sx/4, -Sy/4, -Sz/2 + PDsize/2), size=mp.Vector3(subpixelsize, subpixelsize,  PDsize), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(Sx/4, -Sy/4, -Sz/2 + PDsize/2), size=mp.Vector3(subpixelsize, subpixelsize,  PDsize), material=SiPD
    # ),
    # mp.Block(
    #     center=mp.Vector3(-Sx/4, Sy/4, -Sz/2 + PDsize/2), size=mp.Vector3(subpixelsize, subpixelsize,  PDsize), material=SiPD
    # )
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
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml - 0.5/resolution), mp.Vector3(0.04,0.04,0.04) 
monitor_position_1, monitor_size_1 = mp.Vector3(-design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml - 0.5/resolution), mp.Vector3(0.04,0.04,0.04) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width_x/4, -design_region_width_y/4, -Sz/2 + Lpml - 0.5/resolution), mp.Vector3(0.04,0.04,0.04) 
monitor_position_3, monitor_size_3 = mp.Vector3(design_region_width_x/4, design_region_width_y/4, -Sz/2 + Lpml - 0.5/resolution), mp.Vector3(0.04,0.04,0.04) 

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
    red = npa.sum(npa.abs(fields_1_x[21:30,1]) ** 2)  + npa.sum(npa.abs(fields_1_y[21:30,1]) ** 2)
    green = npa.sum(npa.abs(fields_0_x[11:20,1]) ** 2) + npa.sum(npa.abs(fields_0_y[11:20,1]) ** 2) + npa.sum(npa.abs(fields_2_x[11:20,1]) ** 2) + npa.sum(npa.abs(fields_2_y[11:20,1]) ** 2)
    blue = npa.sum(npa.abs(fields_3_x[1:10,1]) ** 2) + npa.sum(npa.abs(fields_3_y[1:10,1]) ** 2)
    redfactor = 1.1
    greenfactor = 0.7
    bluefactor = 0.5
    
    fred.append(red/redfactor)
    fgreen.append(green/greenfactor)
    fblue.append(blue/bluefactor)
    return blue/bluefactor + green/greenfactor + red/redfactor

# optimization 설정

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region_1,design_region_2,design_region_3,design_region_4,design_region_5],
    frequencies=frequencies,
    decay_by=1e-3,
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
            v, eta_i, beta, np.mean(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))

    np.savetxt(design_dir+"structure1_0"+str(numevl) +".txt", design_variables_1.weights)
    np.savetxt(design_dir+"structure2_0"+str(numevl) +".txt", design_variables_2.weights)
    np.savetxt(design_dir+"structure3_0"+str(numevl) +".txt", design_variables_3.weights)
    np.savetxt(design_dir+"structure4_0"+str(numevl) +".txt", design_variables_4.weights)
    np.savetxt(design_dir+"structure5_0"+str(numevl) +".txt", design_variables_5.weights)
    
    numevl += 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))

    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)

###############################################################################################################################

# ## 3. Algorithm select

algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny * Nz  # number of parameters

# Initial guess 
x = np.random.uniform(0.45, 0.55, n )



# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny * Nz))
ub = np.ones((Nx * Ny * Nz))

f0, dJ_du = opt()

# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 8
update_factor = 30  # number of iterations between beta updates
for iters in range(num_betas):
    print("current beta: ", cur_beta)
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    if cur_beta >=256:
        solver.set_max_objective(lambda a, g: f(a, g, mp.inf))
        solver.set_maxeval(1) # Set the maximum number of function evaluations
    else:
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration

###############################################################################################################################

# ## 4. Save result

#np.save("evaluation", evaluation_history)
np.savetxt(design_dir+"evaluation.txt", evaluation_history)

def extract_elements(lst):
    # 결과를 저장할 리스트를 생성합니다.
    result = []

    # 리스트의 길이만큼 반복하면서 4의 배수 인덱스의 요소를 추출합니다.
    for i in range(0, len(lst), 9):
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
plt.savefig(design_dir+"FoMrgbresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure
# FoM plot

plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"FoMresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

np.savetxt(design_dir+"lastdesign_1.txt", design_variables_1.weights)
np.savetxt(design_dir+"lastdesign_2.txt", design_variables_2.weights)
np.savetxt(design_dir+"lastdesign_3.txt", design_variables_3.weights)
np.savetxt(design_dir+"lastdesign_4.txt", design_variables_4.weights)
np.savetxt(design_dir+"lastdesign_5.txt", design_variables_5.weights)

intensities_0 = np.abs(opt.get_objective_arguments()[2][:,1]) ** 2
plt.figure(figsize=(10, 8))
plt.plot(wavelengths/um_scale, intensities_0, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ex|^2 intensity (a.u.)")
plt.savefig(design_dir+"r_Ex_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_0_1 = np.abs(opt.get_objective_arguments()[3][:,1]) ** 2
plt.figure(figsize=(10, 8))
plt.plot(wavelengths/um_scale, intensities_0_1, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ey|^2 intensity (a.u.)")
plt.savefig(design_dir+"r_Ey_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


intensities_1 = np.abs(opt.get_objective_arguments()[0][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_1, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ex|^2 intensity (a.u.)")
plt.savefig(design_dir+"g0_Ex_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_1_1 = np.abs(opt.get_objective_arguments()[1][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_1_1, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ey|^2 intensity (a.u.)")
plt.savefig(design_dir+"g0_Ey_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_2 = np.abs(opt.get_objective_arguments()[4][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_2, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ex|^2 intensity (a.u.)")
plt.savefig(design_dir+"g1_Ex_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_2_1 = np.abs(opt.get_objective_arguments()[5][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_2_1, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ey|^2 intensity (a.u.)")
plt.savefig(design_dir+"g1_Ey_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_3 = np.abs(opt.get_objective_arguments()[6][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_3, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ex|^2 intensity (a.u.)")
plt.savefig(design_dir+"b_Ex_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

intensities_3_1 = np.abs(opt.get_objective_arguments()[7][:,1]) ** 2
plt.figure(figsize=(10, 8))
# plt.figure()
plt.plot(wavelengths/um_scale, intensities_3_1, "-o")
plt.grid(True)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ey|^2 intensity (a.u.)")
plt.savefig(design_dir+"b_Ey_intensity.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure
