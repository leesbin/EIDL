# CIS 설계 강건성 최적화 코드
# 이 코드는 MEEP와 adjoint 최적화 기법을 이용하여 CIS 설계의 강건성을 최적화합니다.

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import nlopt
from matplotlib import pyplot as plt

# 시뮬레이션 환경 설정
mp.verbosity(1)
um_scale = 2  # 1A = 0.5 um
seed = 240  # 재현성을 위한 난수 시드 고정
np.random.seed(seed)

# 물질 정의
Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
SiPD = mp.Medium(epsilon=9)

# 시뮬레이션 해상도 및 공간 설정
resolution = 25  # 해상도 설정
design_region_width = 3  # 디자인 영역 너비 (1.5 um)
design_region_height = 4  # 디자인 영역 높이 (2 um)
pml_size = 1  # PML 영역 크기 (0.5 um)
pml_layers = [mp.PML(thickness=pml_size, direction=mp.Y)]  # PML 설정

# 시뮬레이션 공간 크기 설정
Sx = design_region_width  # x축 크기 (1.5 um)
Sy = 2 * pml_size + design_region_height + 2  # y축 크기 (3 um)
cell_size = mp.Vector3(Sx, Sy)  # 시뮬레이션 셀 크기 설정

# 파장 및 주파수 설정
wavelengths = np.array([0.43 * um_scale, 0.45 * um_scale, 0.47 * um_scale, 
                        0.53 * um_scale, 0.55 * um_scale, 0.57 * um_scale, 
                        0.62 * um_scale, 0.65 * um_scale, 0.68 * um_scale])
frequencies = 1 / wavelengths
nf = len(frequencies)  # frequency 개수

# 제작 공정 제약 설정
minimum_length = 0.08  # 최소 feature 크기 (40 nm)
eta_i = 0.5  # intermediate design 필드 임계값
eta_e = 0.55  # erosion design 필드 임계값
eta_d = 1 - eta_e  # dilation design 필드 임계값
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, 0.75)  # 필터 반지름
design_region_resolution = int(resolution)  # 디자인 영역 해상도

# 소스 설정
width = 2
frequency = 1 / (0.550 * um_scale)
fwidth = frequency * width
src = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)

source_center = [0, Sy / 2 - pml_size - 0.5, 0]  # 소스 위치
source_size = mp.Vector3(Sx, 0, 0)  # 소스 크기

src_0 = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)  # 소스 설정
k_point_0 = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), 0)  # k-point 설정 (입사각 0도)

# EigenMode 소스 설정
source0 = [mp.EigenModeSource(src=src_0,
                              center=source_center,
                              size=source_size,
                              direction=mp.AUTOMATIC,
                              eig_kpoint=k_point_0,
                              eig_band=1,
                              eig_parity=mp.EVEN_Y + mp.ODD_Z,
                              eig_match_freq=True)]

# 설계 영역 설정
Nx = int(round(design_region_resolution * design_region_width)) + 1
Ny = int(round(design_region_resolution * design_region_height)) + 1
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables, 
    volume=mp.Volume(center=mp.Vector3(0, 0, 0), 
    size=mp.Vector3(design_region_width, design_region_height, 0)
    )
)

# 필터 및 프로젝션 함수 정의
def mapping(x, eta, beta):
    # 필터 및 프로젝션 수행
    x = x.flatten()
    filtered_field = mpa.conic_filter(
        x, 
        filter_radius, 
        design_region_width, 
        design_region_height, 
        design_region_resolution
        )
    
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    return projected_field.flatten()

# 시뮬레이션 설정 및 블록 정의
geometry = [
    mp.Block(
        center=design_region.center, 
        size=design_region.size, 
        material=design_variables
        ),
    mp.Block(
        center=mp.Vector3(0, -3, 0), 
        size=mp.Vector3(Sx, 2, 0), 
        material=SiPD
        )
]

# Normalization flux 계산을 위한 시뮬레이션 설정
sim = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers, 
    sources=source0, 
    default_material=Air, 
    resolution=resolution, 
    k_point=k_point_0
)

# 각 주파수에서 필드 추가 및 계산
# DFT fields 계산
tran_Ez = [sim.add_dft_fields([mp.Ez], frequencies[i], 0, 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True) for i in range(9)]
sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-9))

# DFT array 계산
dft_array_Ez = [np.sum(np.abs(sim.get_dft_array(tran_Ez[i], mp.Ez, 0)) ** 2) for i in range(9)]

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
    sources=source0,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point_0, # bloch boundary
)

sim2 = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source0,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point_0, # bloch boundary
)

# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-1, -2 - 0.5/resolution), mp.Vector3(0.5,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -2 - 0.5/resolution), mp.Vector3(0.5,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(1, -2 - 0.5/resolution), mp.Vector3(0.5,0)

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

FourierFields_0_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_1_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_2_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_3_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_4_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_5_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_6_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_7_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_8_1 = mpa.FourierFields(sim1,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
ob_list_1 = [FourierFields_0_1, FourierFields_1_1, FourierFields_2_1,FourierFields_3_1, FourierFields_4_1, FourierFields_5_1 ,FourierFields_6_1, FourierFields_7_1, FourierFields_8_1]

FourierFields_0_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_1_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_2_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)
FourierFields_3_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_4_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_5_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)
FourierFields_6_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_7_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
FourierFields_8_2 = mpa.FourierFields(sim2,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)
ob_list_2 = [FourierFields_0_2, FourierFields_1_2, FourierFields_2_2,FourierFields_3_2, FourierFields_4_2, FourierFields_5_2 ,FourierFields_6_2, FourierFields_7_2, FourierFields_8_2]

# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
# [frequency index, moniter index]
def J_0(fields_0, fields_1, fields_2, fields_3, fields_4, fields_5, fields_6, fields_7, fields_8):
    blue = (npa.sum(npa.abs(fields_8[0, :]) ** 2)) / dft_array_Ez[0] + (npa.sum(npa.abs(fields_7[1, :]) ** 2)) / dft_array_Ez[1] + (npa.sum(npa.abs(fields_6[2, :]) ** 2)) / dft_array_Ez[2]
    green = (npa.sum(npa.abs(fields_5[3, :]) ** 2)) / dft_array_Ez[3] + (npa.sum(npa.abs(fields_4[4, :]) ** 2)) / dft_array_Ez[4] + (npa.sum(npa.abs(fields_3[5, :]) ** 2)) / dft_array_Ez[5]
    red = (npa.sum(npa.abs(fields_2[6, :]) ** 2)) / dft_array_Ez[6] + (npa.sum(npa.abs(fields_1[7, :]) ** 2)) / dft_array_Ez[7] + (npa.sum(npa.abs(fields_0[8, :]) ** 2)) / dft_array_Ez[8]

    return 9 - (blue + green + red)

# 최적화 설정
opt0 = mpa.OptimizationProblem(
    simulation=sim0,
    objective_functions=[J_0],
    objective_arguments=ob_list_0,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-3, # 모니터에 남아있는 필드 값의 비율
)
opt1 = mpa.OptimizationProblem(
    simulation=sim1,
    objective_functions=[J_0],
    objective_arguments=ob_list_1,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-3, # 모니터에 남아있는 필드 값의 비율
)
opt2 = mpa.OptimizationProblem(
	simulation=sim2,
	objective_functions=[J_0],
	objective_arguments=ob_list_2,
	design_regions=[design_region],
	frequencies=frequencies,
	decay_by=1e-3, # 모니터에 남아있는 필드 값의 비율
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


def c(result, x, gradient, beta):
    
    print("Current iteration: {}".format(cur_iter[0] + 1))

    global numevl
    t=x[0]
    v=x[1:]

    np.savetxt("v.txt", v) # save the design variables
    
    f0_0, dJ_du0 = opt0([mapping(v, eta_i, beta)])  # compute objective and gradient blueprint
    f0_1, dJ_du1 = opt1([mapping(v, eta_d, beta)])  # compute objective and gradient dilation
    f0_2, dJ_du2 = opt2([mapping(v, eta_e, beta)])  # compute objective and gradient erosion

    f0_0 = f0_0.flatten()
    f0_1 = f0_1.flatten()
    f0_2 = f0_2.flatten()

    f0_merged = np.hstack([f0_0, f0_1, f0_2])
    f0_merged_str = '[' + ','.join(str(ff) for ff in f0_merged) + ']'
    nf = 3
    
    dJ_du_0 = np.sum(dJ_du0, axis=1)
    dJ_du_1 = np.sum(dJ_du1, axis=1)
    dJ_du_2 = np.sum(dJ_du2, axis=1)

    my_grad = np.zeros((Nx * Ny, nf))
    my_grad[:,0] = dJ_du_0
    my_grad[:,1] = dJ_du_1
    my_grad[:,2] = dJ_du_2

    for k in range(3):
        my_grad[:,k] = tensor_jacobian_product(mapping,0)(
            v,
            eta_i,
            cur_beta,
            my_grad[:,k],
        )
        
    if gradient.size > 0:
        gradient[:, 0] = -1
        gradient[:, 1:] = my_grad.T

    result[:] = np.real(f0_merged) - t

    evaluation_history.append([np.max(np.real(f0_0)), np.max(np.real(f0_1)), np.max(np.real(f0_2))])

    numevl += 1
    cur_iter[0] = cur_iter[0] + 1

    print("First FoM: {}".format(evaluation_history[0]))
    print("Current f0_0: {}, f0_1: {}, f0_2: {}".format(np.real(f0_0), np.real(f0_1), np.real(f0_2)))

algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny  # number of parameters

# Initial guess - 초기 시작값 랜덤
x = np.random.rand((n))

# lower and upper bounds (상한 : 1, 하한 : 0)
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# insert dummy parameter bounds and variable
x = np.insert(x, 0, -1)  # our initial guess for the worst error
lb = np.insert(lb, 0, -np.inf) # lower bound for the error
ub = np.insert(ub, 0, 0) # upper bound for the error

# Optimization parameter
cur_beta = 2 # initial beta value
beta_scale = 2 # scaling factor for beta updates
num_betas = 20 # number of beta updates
tol_epi = np.array([1e-4] * 3) # tolerances for the epsilon constraints
update_factor = 30  # number of iterations between beta updates
ftol = 1e-5  # tolerance for the objective function

# Optimization
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
