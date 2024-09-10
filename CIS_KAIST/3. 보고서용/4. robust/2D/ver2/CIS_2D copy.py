# Description: This code is for the optimization of the color router

# ## 1. Simulation Environment


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
SiPD = mp.Medium(epsilon=9)


# cell 환경 설정
resolution = 25 # 해상도
design_region_width = 3 # 디자인 영역 너비
design_region_height = 4 # 디자인 영역 높이
pml_size = 1 # PML 영역 크기

pml_layers = [mp.PML(thickness = pml_size, direction = mp.Y)]

rot_angle = np.radians(6)

k_point_n = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), -rot_angle)

k_point_0 = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), 0)

k_point_p = mp.Vector3(y = -1).rotate(mp.Vector3(z=1), rot_angle)




# 시뮬레이션 공간 설정
Sx = design_region_width 
Sy = 2 * pml_size + design_region_height + 2
cell_size = mp.Vector3(Sx, Sy)




# 파장, 주파수 설정
wavelengths = np.array([0.43*um_scale, 0.45*um_scale, 0.47*um_scale, 0.53*um_scale, 0.55*um_scale, 0.57*um_scale, 0.62*um_scale, 0.65*um_scale, 0.68*um_scale])
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# Fabrication Constraints 설정

minimum_length = 0.08  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.75  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, 0.75)
design_region_resolution = int(resolution)



# source 설정
width = 2
frequency = 1/(0.550*um_scale)
fwidth = frequency * width
src = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)

source_center = [0, Sy / 2 - pml_size - 0.5, 0] # Source 위치
source_size = mp.Vector3(Sx , 0, 0)

src_0 = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)

source0 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC,
		    eig_kpoint=k_point_0,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z,
		    eig_match_freq=True,
		),]

          
source1 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_n,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),]

source2 = [mp.EigenModeSource(
		    src=src_0,
		    center=source_center,
		    size=source_size,
		    direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
		    eig_kpoint=k_point_p,
		    eig_band=1,
		    eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
		    eig_match_freq=True,
		),]


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

if True : ##### normalization flux calculation #################################################################################################################
	sim = mp.Simulation(
		cell_size=cell_size, 
		boundary_layers=pml_layers,
		sources=source0,
		default_material=Air, # 빈공간
		resolution=resolution,
		k_point = k_point_0
	)

	b_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[0], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
	b_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[1], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True) 
	b_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[2], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
     
	g_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[3], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
	g_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[4], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
	g_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[5], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
    
	r_0_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[6], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
	r_1_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[7], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
	r_2_tran_Ez = sim.add_dft_fields([mp.Ez], frequencies[8], 0 , 1, center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0), yee_grid=True)
     
	sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-9))
    
	b_0_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(b_0_tran_Ez, mp.Ez, 0)) ** 2)
	b_1_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(b_1_tran_Ez, mp.Ez, 0)) ** 2)
	b_2_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(b_2_tran_Ez, mp.Ez, 0)) ** 2)
     
	g_0_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(g_0_tran_Ez, mp.Ez, 0)) ** 2)
	g_1_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(g_1_tran_Ez, mp.Ez, 0)) ** 2)
	g_2_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(g_2_tran_Ez, mp.Ez, 0)) ** 2)

	r_0_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(r_0_tran_Ez, mp.Ez, 0)) ** 2)
	r_1_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(r_1_tran_Ez, mp.Ez, 0)) ** 2)
	r_2_dft_array_Ez = np.sum(np.abs(sim.get_dft_array(r_2_tran_Ez, mp.Ez, 0)) ** 2)
     
	


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
    k_point = k_point_0,
)

sim2 = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source0,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = k_point_0,
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
    blue = (npa.sum(npa.abs(fields_8[0, :]) ** 2)) / b_0_dft_array_Ez + (npa.sum(npa.abs(fields_7[1, :]) ** 2)) / b_1_dft_array_Ez + (npa.sum(npa.abs(fields_6[2, :]) ** 2)) / b_2_dft_array_Ez
    green = (npa.sum(npa.abs(fields_5[3, :]) ** 2)) / g_0_dft_array_Ez + (npa.sum(npa.abs(fields_4[4, :]) ** 2)) / g_1_dft_array_Ez + (npa.sum(npa.abs(fields_3[5, :]) ** 2)) / g_2_dft_array_Ez
    red = (npa.sum(npa.abs(fields_2[6, :]) ** 2)) / r_0_dft_array_Ez + (npa.sum(npa.abs(fields_1[7, :]) ** 2)) / r_1_dft_array_Ez + (npa.sum(npa.abs(fields_0[8, :]) ** 2)) / r_2_dft_array_Ez
 
    # print(red)
    # print(green)
    # print(blue)

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
opt0.plot2D(True)
# plt.axis("off")
plt.savefig("Lastdesignxy.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure 

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
    
    f0_0, dJ_du0 = opt0([mapping(v, eta_i, beta)])  # compute objective and gradient
    f0_1, dJ_du1 = opt1([mapping(v, eta_d, beta)])  # compute objective and gradient
    f0_2, dJ_du2 = opt2([mapping(v, eta_e, beta)])  # compute objective and gradient

    f0_0 = f0_0.flatten()
    f0_1 = f0_1.flatten()
    f0_2 = f0_2.flatten()
    # print(f0_0, f0_1, f0_2)
    # print(dJ_du0, dJ_du1, dJ_du2)
    
    f0_merged = np.hstack([f0_0, f0_1, f0_2])
    f0_merged_str = '[' + ','.join(str(ff) for ff in f0_merged) + ']'
    nf = 3
    
    dJ_du_0 = np.sum(dJ_du0, axis=1)
    dJ_du_1 = np.sum(dJ_du1, axis=1)
    dJ_du_2 = np.sum(dJ_du2, axis=1)
    # print(dJ_du_0.shape)

    my_grad = np.zeros((Nx * Ny, nf))
    my_grad[:,0] = dJ_du_0
    my_grad[:,1] = dJ_du_1
    my_grad[:,2] = dJ_du_2
    # print(my_grad.shape)

    for k in range(3):
        my_grad[:,k] = tensor_jacobian_product(mapping,0)(
            v,
            eta_i,
            cur_beta,
            my_grad[:,k],
        )

    # print(gradient.shape)
        
    if gradient.size > 0:
        gradient[:, 0] = -1
        gradient[:, 1:] = my_grad.T
        
    # print(result.shape)

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
lb = np.insert(lb, 0, -np.inf)
ub = np.insert(ub, 0, 0)

# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 20
tol_epi = np.array([1e-4] * 3)
update_factor = 30  # number of iterations between beta updates
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

plt.figure(figsize=(20, 8))

intensities_R = np.abs(opt1.get_objective_arguments()[0][:,0]) ** 2
plt.subplot(1,3,1)
plt.plot(wavelengths/um_scale, intensities_R, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_G = np.abs(opt1.get_objective_arguments()[3][:,1]) ** 2
plt.subplot(1,3,2)
plt.plot(wavelengths/um_scale, intensities_G, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()

intensities_B = np.abs(opt1.get_objective_arguments()[6][:,1]) ** 2
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
