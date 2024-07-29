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
wavelengths_400 = np.linspace(0.44*um_scale, 0.46*um_scale, 3)
wavelengths_500 = np.linspace(0.54*um_scale, 0.56*um_scale, 3) 
wavelengths_600 = np.linspace(0.64*um_scale, 0.66*um_scale, 3) 
frequencies_400 = 1/wavelengths_400
frequencies_500 = 1/wavelengths_500
frequencies_600 = 1/wavelengths_600
nf = 3  # number of frequencies


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

source0 = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center, )]
source1 = [mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center, )]
source2 = [mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center, )]




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
sim0 = mp.Simulation(
    cell_size=cell_size, # cell_size = mp.Vector3(Sx, Sy)
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source0,
    default_material=Air, # 빈공간
    resolution=resolution,
    # extra_materials=[SiN],
    k_point = mp.Vector3(0,0,0)
)
sim1 = mp.Simulation(
    cell_size=cell_size, # cell_size = mp.Vector3(Sx, Sy)
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source1,
    default_material=Air, # 빈공간
    resolution=resolution,
    # extra_materials=[SiN],
    k_point = mp.Vector3(0,0,0)
)
sim2 = mp.Simulation(
    cell_size=cell_size, # cell_size = mp.Vector3(Sx, Sy)
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source2,
    default_material=Air, # 빈공간
    resolution=resolution,
    # extra_materials=[SiN],
    k_point = mp.Vector3(0,0,0)
)

###############################################################################################################################
# ## 2. Optimization Environment


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0)


# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim0,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2= mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)

ob_list0 = [FourierFields_0]
ob_list1 = [FourierFields_1]
ob_list2 = [FourierFields_2]



fred = []
fgreen = []
fblue = []
# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취함
# [frequency index, moniter index]

def J_0(fields):
    return - npa.abs(fields[:,1]) ** 2
def J_1(fields):
    return - npa.abs(fields[:,1]) ** 2
def J_2(fields):
    return - npa.abs(fields[:,1]) ** 2


# optimization 설정

opt0 = mpa.OptimizationProblem(
    simulation=sim0,
    objective_functions=[J_0],
    objective_arguments=ob_list0,
    design_regions=[design_region],
    frequencies=frequencies_600,
    decay_by=1e-5, # 모니터에 남아있는 필드 값의 비율
)
opt1 = mpa.OptimizationProblem(
    simulation=sim1,
    objective_functions=[J_1],
    objective_arguments=ob_list1,
    design_regions=[design_region],
    frequencies=frequencies_500,
    decay_by=1e-5, # 모니터에 남아있는 필드 값의 비율
)
opt2 = mpa.OptimizationProblem(
    simulation=sim2,
    objective_functions=[J_2],
    objective_arguments=ob_list2,
    design_regions=[design_region],
    frequencies=frequencies_400,
    decay_by=1e-5, # 모니터에 남아있는 필드 값의 비율
)


evaluation_history = []
cur_iter = [0]
numevl = 1

def f(x, gradient):
    t = x[0]
    v = x[1:]
    if gradient.size > 0:
        gradient[0] = 1
        gradient[1:] = 0
    return t

def c(result, x, gradient, cur_beta): 

    global numevl
    t = x[0]
    v = x[1:]

    f0_1, dJ_du1 = opt0([mapping(v, eta_i, cur_beta)])
    f0_2, dJ_du2 = opt1([mapping(v, eta_i, cur_beta)])
    f0_3, dJ_du3 = opt2([mapping(v, eta_i, cur_beta)])

    f0_1 = f0_1.flatten()
    f0_2 = f0_2.flatten()
    f0_3 = f0_3.flatten()
    print(f0_1, f0_2, f0_3)
    print(dJ_du1, dJ_du2, dJ_du3)

    f0_merged = np.hstack((f0_1, f0_2, f0_3))
    f0_merged_str = '[' + ','.join(str(ff) for ff in f0_merged) + ']'
    nfrq = 3

    my_grad = np.zeros((Nx * Ny, 3*nfrq))
    my_grad[:,:nfrq] = dJ_du1
    my_grad[:,nfrq:2*nfrq] = dJ_du2
    my_grad[:,2*nfrq:] = dJ_du3

    for k in range(3 * nfrq):
        my_grad[:,k] = tensor_jacobian_product(mapping,0)(
            v,
            eta_i,
            cur_beta,
            my_grad[:,k],
        )

    if gradient.size > 0:
        gradient[:,0] = -1
        gradient[:, 1:] = my_grad.T

    result[:] = np.real(f0_merged) - t

    evaluation_history.append([np.max(np.real(f0_1)), np.max(np.real(f0_2)), np.max(np.real(f0_3))])

    numevl += 1
    cur_iter[0] = cur_iter[0] + 1

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

# insert dummy parameter bounds and variable
x = np.insert(x, 0, -1)  # our initial guess for the worst error
lb = np.insert(lb, 0, -np.inf)
ub = np.insert(ub, 0, 0)


# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 15
update_factor = 20  # number of iterations between beta updates

tol_epi = np.array([1e-4] * 3 * 3)
ftol = 1e-5
for iters in range(num_betas):
    print("current beta: ", cur_beta)
    solver = nlopt.opt(algorithm, n+1)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    solver.set_min_objective(f)
    solver.add_inequality_mconstraint(
        lambda rr, xx, gg: c(
            rr,
            xx,
            gg,
            cur_beta,
        ),
        tol_epi
    )
    solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
        
    # solver.set_param("dual_ftol_rel", 1e-7)
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


# Last design plot

plt.imshow(npa.rot90(design_variables.weights.reshape(Nx, Ny)), cmap='binary')
plt.colorbar()

np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
plt.savefig(design_dir+"lastdesign.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

evaluation_history = np.array(evaluation_history)
evaluation_history = evaluation_history.reshape(numevl-1,3)

plt.figure()
plt.plot(- evaluation_history[:,0], "r-", label="R")
plt.plot(- evaluation_history[:,1], "g-", label="G") 
plt.plot(- evaluation_history[:,2], "b-", label="B")               

plt.xlabel("Iterations")
plt.ylabel("FOM")
plt.savefig("result.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

