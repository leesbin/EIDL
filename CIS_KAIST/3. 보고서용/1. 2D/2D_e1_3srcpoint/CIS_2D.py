#!/usr/bin/env python
# coding: utf-8

# In[112]:
# First FoM: 4.998263621736106
# Current FoM: 77.24726233563729

# Elapsed run time = 220.7563 s





# 시뮬레이션 라이브러리 불러오기 4_core_1468.4815 s


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
# from meep.materials import SiN


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
design_region_width = 3 # 디자인 영역 너비
design_region_height = 4 # 디자인 영역 높이
pml_size = 1 # PML 영역 크기


# In[4]:


# 시뮬레이션 공간 설정
Sx = design_region_width 
Sy = 2 * pml_size + design_region_height + 2
cell_size = mp.Vector3(Sx, Sy)


# In[5]:


# 파장, 주파수 설정
wavelengths = np.array([0.45*um_scale, 0.55*um_scale, 0.65*um_scale])
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


pml_layers = [mp.PML(thickness = pml_size, direction = mp.Y)]


# In[8]:


width = 0.1
fcen_red = frequencies[2]
fwidth_red = width * fcen_red


fcen_green = frequencies[1]
fwidth_green = width * fcen_green


fcen_blue = frequencies[0]
fwidth_blue = width * fcen_blue



# In[9]:


source_center = [0, Sy / 2 - pml_size - 0.5, 0] # Source 위치
source_size = mp.Vector3(Sx , 0, 0)

src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=1 /(0.60 * um_scale) - 1/(0.70 * um_scale), is_integrated=True)
src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=1 /(0.50 * um_scale) - 1/(0.60 * um_scale), is_integrated=True)
src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=1 /(0.40 * um_scale) - 1/(0.50 * um_scale), is_integrated=True)
source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center, amplitude = 1),
          mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.397612003),
          mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center, amplitude = 2.090177677)]


# In[10]:


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


# In[11]:


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
    # 출력값 -1 ~ 1으로 제한
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return projected_field.flatten()


# In[12]:


# design region과 동일한 size의 Block 생성
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(0, -3, 0), size=mp.Vector3(Sx, 2, 0), material=SiPD
    ),
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


# In[13]:


# Meep simulation 세팅
sim = mp.Simulation(
    cell_size=cell_size, # cell_size = mp.Vector3(Sx, Sy)
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # 빈공간
    resolution=resolution,
    extra_materials=[SiN],
    k_point = mp.Vector3(0,0,0)
)


# In[14]:


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-1, -2 - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -2 - 0.5/resolution), mp.Vector3(0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(1, -2 - 0.5/resolution), mp.Vector3(0.01,0)


# In[15]:


# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)

ob_list = [FourierFields_0, FourierFields_1, FourierFields_2]


# In[16]:


# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
def J_0(fields_0, fields_1, fields_2):
    return npa.abs(fields_2[0,1]) ** 2 + npa.abs(fields_1[1,1]) ** 2 + npa.abs(fields_0[2,1]) ** 2
    # return npa.mean(npa.abs(fields[0,:]) ** 2)
# def J_1(fields):
#     return npa.mean(npa.abs(fields[1,:]) ** 2) 
# def J_2(fields):
#     return npa.mean(npa.abs(fields[2,:]) ** 2)


# In[17]:


opt = mpa.OptimizationProblem(
    simulation=sim,
    # objective_functions=[J_0, J_1, J_2],
    objective_functions=[J_0],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1,
)
plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.plot2D(False)


# In[18]:


evaluation_history = []
cur_iter = [0]

def f(v, gradient, beta):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([mapping(v, eta_i, beta)])  # compute objective and gradient
    # f0, dJ_du = opt()
    
    # Adjoint gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, np.sum(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))

    # plt.figure()
    # ax = plt.gca()
    # opt.plot2D(
    #     False,
    #     ax=ax,
    #     plot_sources_flag=False,
    #     plot_monitors_flag=False,
    #    plot_boundaries_flag=False,
    # )
    # circ = Circle((2, 2), minimum_length / 2)
    # ax.add_patch(circ)
    # ax.axis("off")
    # plt.show()

    cur_iter[0] = cur_iter[0] + 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    

    return np.real(f0)


# In[19]:


algorithm = nlopt.LD_MMA # 어떤 알고리즘으로 최적화를 할 것인가?
# MMA : 점근선 이동

n = Nx * Ny  # number of parameters

# Initial guess - 초기 시작값 0.5
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
    # solver.set_max_objective(lambda a, g: f1(a, g, cur_beta), lambda a, g: f2(a, g, cur_beta), lambda a, g: f3(a, g, cur_beta)) # Set the objective function to be maximized
    solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration


# In[20]:


plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FOM")
plt.savefig("FoMresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


# In[21]:


plt.imshow(npa.rot90(design_variables.weights.reshape(Nx, Ny)), cmap='binary')
plt.colorbar()

np.savetxt("lastdesign.txt", design_variables.weights)
plt.savefig("lastdesign.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


# In[114]:


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
plt.savefig("FinalEz.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure

# In[115]:


# # z plane에서의 Ez field plot

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.ContinuousSource(frequency=frequencies[0], fwidth=fwidth_blue0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, )]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[0]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[1], fwidth=fwidth_blue1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[1]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[2], fwidth=fwidth_blue2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[2]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# plt.savefig("EzBlue.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[43]:



# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.GaussianSource(frequency=frequencies[0], fwidth=fwidth_blue0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.760188053)]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# tran_total0 = opt.sim.add_dft_fields([mp.Ez], frequencies[0], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[0]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일


# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux0 = np.abs(sim.get_dft_array(tran_total0, mp.Ez, 0))**2


# plt.imshow(npa.rot90(total_flux0.reshape(Sx* resolution+1, Sy* resolution+1)),cmap="Blues")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[1], fwidth=fwidth_blue1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.685985199)]
# opt.sim.change_sources(source)

# tran_total1 = opt.sim.add_dft_fields([mp.Ez], frequencies[1], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)


# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[1]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일


# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux1 = np.abs(sim.get_dft_array(tran_total1, mp.Ez, 0))**2

# plt.imshow(npa.rot90(total_flux1.reshape(Sx* resolution+1, Sy* resolution+1)),cmap="Blues")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[2], fwidth=fwidth_blue2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.586447998)]
# opt.sim.change_sources(source)

# tran_total2 = opt.sim.add_dft_fields([mp.Ez], frequencies[2], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)

# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[2]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일


# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux2 = np.abs(sim.get_dft_array(tran_total2, mp.Ez, 0))**2

# plt.imshow(npa.rot90(total_flux2.reshape(Sx* resolution+1, Sy* resolution+1)),cmap="Blues")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# plt.savefig("IntensityBlue.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[25]:


# print(total_flux0)


# # In[116]:


# # z plane에서의 Ez field plot

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.ContinuousSource(frequency=frequencies[3], fwidth=fwidth_green0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[3]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[4], fwidth=fwidth_green1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[4]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[5], fwidth=fwidth_green2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[5]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# plt.savefig("EzGreen.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[44]:


# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.GaussianSource(frequency=frequencies[3], fwidth=fwidth_green0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.405765992)]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# tran_total3 = opt.sim.add_dft_fields([mp.Ez], frequencies[3], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[3]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux3 = np.abs(sim.get_dft_array(tran_total3, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux3.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Greens")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[4], fwidth=fwidth_green1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.329122323)]
# opt.sim.change_sources(source)

# tran_total4 = opt.sim.add_dft_fields([mp.Ez], frequencies[4], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[4]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux4 = np.abs(sim.get_dft_array(tran_total4, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux4.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Greens")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[5], fwidth=fwidth_green2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.264200342)]
# opt.sim.change_sources(source)

# tran_total5 = opt.sim.add_dft_fields([mp.Ez], frequencies[5], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[5]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux5 = np.abs(sim.get_dft_array(tran_total5, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux4.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Greens")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# plt.savefig("IntensityGreen.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[117]:


# # z plane에서의 Ez field plot

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     # extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.ContinuousSource(frequency=frequencies[6], fwidth=fwidth_red0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[6]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     # extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[7], fwidth=fwidth_red1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[7]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     # extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.ContinuousSource(frequency=frequencies[8], fwidth=fwidth_red2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[8]/um_scale))
# opt.sim.run(until=300)
# plt.fill([Sx/6, Sx/6, Sx/2, Sx/2], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='blue', alpha=0.5)
# plt.fill([-Sx/6, -Sx/6, Sx/6, Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='green', alpha=0.5)
# plt.fill([-Sx/2, -Sx/2, -Sx/6, -Sx/6], [-Sy/2, -Sy/2+pml_size, -Sy/2+pml_size, -Sy/2], color='red', alpha=0.5)
# opt.sim.plot2D(fields=mp.Ez)
# plt.savefig("EzRed.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[45]:


# # z plane에서의 Ez field plot

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )
# src = mp.GaussianSource(frequency=frequencies[6], fwidth=fwidth_red0, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.123943214)]
# opt.sim.change_sources(source)

# plt.figure(figsize=(16, 10))

# tran_total6 = opt.sim.add_dft_fields([mp.Ez], frequencies[6], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,1)
# plt.title('wavelength = '+str(wavelengths[6]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux6 = np.abs(sim.get_dft_array(tran_total6, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux6.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Reds")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[7], fwidth=fwidth_red1, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center, amplitude = 1.056608025)]
# opt.sim.change_sources(source)

# tran_total7 = opt.sim.add_dft_fields([mp.Ez], frequencies[7], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,2)
# plt.title('wavelength = '+str(wavelengths[7]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux7 = np.abs(sim.get_dft_array(tran_total7, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux7.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Reds")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# opt.sim.reset_meep()

# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# src = mp.GaussianSource(frequency=frequencies[8], fwidth=fwidth_red2, is_integrated=True)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)

# tran_total8 = opt.sim.add_dft_fields([mp.Ez], frequencies[8], 0 , 1, center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), yee_grid=True)
# plt.subplot(1,3,3)
# plt.title('wavelength = '+str(wavelengths[8]/um_scale))
# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))
# total_flux8 = np.abs(sim.get_dft_array(tran_total8, mp.Ez, 0)) ** 2

# plt.imshow(npa.rot90(total_flux8.reshape(Sx* resolution+1, Sy* resolution+1)), cmap="Reds")
# plt.fill([100, 100, 150, 150], [600, 550, 550, 600], color='blue', alpha=0.5)
# plt.fill([50, 50, 100, 100], [600, 550, 550, 600], color='green', alpha=0.5)
# plt.fill([0, 0, 50, 50], [600, 550, 550, 600], color='red', alpha=0.5)
# plt.colorbar()
# plt.savefig("IntensityRed.png")
# plt.cla()   # clear the current axes
# plt.clf()   # clear the current figure
# plt.close() # closes the current figure

# # In[101]:


# opt.sim.reset_meep()
# geometry_1 = [
#     mp.Block(
#         center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, 0), material=Air
#     )
# ]


# # In[102]:


# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry_1,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# fcen = (1/(0.40 * um_scale) + 1/(0.70 * um_scale))/2
# df = 1 /(0.40 * um_scale) - 1/(0.70 * um_scale)
# nfreq = 300

# src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
# # print((frequencies[2]+frequencies[0])/2)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)


# # In[103]:


# tran_t = mp.FluxRegion(
#     center=mp.Vector3(0, -Sy/2 + pml_size + 0.5 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0)
# )
# tran_total = opt.sim.add_flux(fcen, df, nfreq, tran_t)

# refl_fr = mp.FluxRegion(
#     center=mp.Vector3(0, Sy/2 - pml_size - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0)
# ) 
# refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution, 0) #pt는 transmitted flux region과 동일


# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))

# straight_refl_data = sim.get_flux_data(refl)
# total_flux = mp.get_fluxes(tran_total)
# flux_freqs = mp.get_flux_freqs(tran_total)


# # In[104]:


# opt.sim.reset_meep()


# # In[105]:


# opt.sim = mp.Simulation(
#     cell_size=mp.Vector3(Sx, Sy),
#     boundary_layers=pml_layers,
#     geometry=geometry,
#     sources=source,
#     default_material=Air,
#     resolution=resolution,
#     extra_materials=[SiN],
#     k_point = mp.Vector3(0,0,0)
# )

# fcen = (1/(0.40 * um_scale) + 1/(0.70 * um_scale))/2
# df = 1 /(0.40 * um_scale) - 1/(0.70 * um_scale)

# src = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True)
# # print((frequencies[2]+frequencies[0])/2)
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# opt.sim.change_sources(source)


# # In[106]:


# # 반사된 flux 구하기

# refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

# # 투과된 flux 구하기

# tran_fr = mp.FluxRegion(
#     center=mp.Vector3(0, -Sy/2 + pml_size + 0.5 - 0.5/resolution, 0), size=mp.Vector3(Sx, 0, 0)
# )
# tran = opt.sim.add_flux(fcen, df, nfreq, tran_fr)

# #반사된 필드와 입사되는 필드를 구분하기 위해서 The Fourier-transformed incident fields을
# #the Fourier transforms of the scattered fields 에서 빼줍니다.
# sim.load_minus_flux_data(refl, straight_refl_data)


# # In[107]:


# tran_r = mp.FluxRegion(
#     center=mp.Vector3(-1, -2 - 0.5/resolution, 0), size=mp.Vector3(1, 0, 0)
# )
# tran_g = mp.FluxRegion(
#     center=mp.Vector3(0, -2 - 0.5/resolution, 0), size=mp.Vector3(1, 0, 0)
# )
# tran_b = mp.FluxRegion(
#     center=mp.Vector3(1, -2 - 0.5/resolution, 0), size=mp.Vector3(1, 0, 0)
# )
# nfreq = 300
# tran_red = opt.sim.add_flux(fcen, df, nfreq, tran_r)
# tran_green = opt.sim.add_flux(fcen, df, nfreq, tran_g)
# tran_blue = opt.sim.add_flux(fcen, df, nfreq, tran_b)


# # In[108]:


# pt = mp.Vector3(1, -Sy/2 + pml_size + 0.5/resolution , 0) #pt는 transmitted flux region과 동일

# opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,pt,1e-3))

# red_flux = mp.get_fluxes(tran_red)
# green_flux = mp.get_fluxes(tran_green)
# blue_flux = mp.get_fluxes(tran_blue)

# refl_flux = mp.get_fluxes(refl)
# tran_flux = mp.get_fluxes(tran)


# # In[109]:


# wl = []
# Rs = []
# Ts = []
# for i in range(nfreq):
#     wl = np.append(wl, 1 / (flux_freqs[i] * um_scale))
#     Rs = np.append(Rs, -refl_flux[i] / total_flux[i])
#     Ts = np.append(Ts, tran_flux[i] / total_flux[i])

# if mp.am_master():
#     plt.figure(dpi=150)
#     plt.plot(wl, Rs, "b", label="reflectance")
#     plt.plot(wl, Ts, "r", label="transmittance")
#     plt.plot(wl, 1 - Rs - Ts, "g", label="loss")
#     #plt.axis([5.0, 10.0, 0, 1])
#     #plt.axis([0.38, 0.78, 0, 1])
#     plt.xlabel("wavelength (μm)")
#     plt.ylabel("Transmittance")
#     plt.fill([0.40, 0.40, 0.50, 0.50], [0.0, 1.0, 1.0, 0.0], color='lightblue', alpha=0.5)
#     plt.fill([0.50, 0.50, 0.60, 0.60], [0.0, 1.0, 1.0, 0.0], color='lightgreen', alpha=0.5)
#     plt.fill([0.60, 0.60, 0.70, 0.70], [0.0, 1.0, 1.0, 0.0], color='lightcoral', alpha=0.5)
#     plt.legend(loc="upper right")
#     # plt.show()
#     plt.savefig("TransRefl.png")
#     plt.cla()   # clear the current axes
#     plt.clf()   # clear the current figure
#     plt.close() # closes the current figure


# # In[110]:


# wl = []
# Tr = []
# Tg = []
# Tb = []
# Tg0 = []
# for i in range(nfreq):
#     wl = np.append(wl, 1 / (flux_freqs[i] * um_scale))
#     Tr = np.append(Tr, red_flux[i] / tran_flux[i] )
#     Tg = np.append(Tg, green_flux[i] / tran_flux[i] )
#     Tb = np.append(Tb, blue_flux[i] / tran_flux[i] )


# if mp.am_master():
#     plt.figure(dpi=150)
#     plt.plot(wl, Tr, "r", label="Redpixel")
#     plt.plot(wl, Tg, "g", label="Greenpixel1")
#     plt.plot(wl, Tb, "b", label="Bluepixel")

    
#     plt.axis([0.40, 0.70, 0, 1])
#     plt.xlabel("wavelength (μm)")
#     plt.ylabel("Transmittance")
#     plt.fill([0.40, 0.40, 0.50, 0.50], [0.0, 1.0, 1.0, 0.0], color='lightblue', alpha=0.5)
#     plt.fill([0.50, 0.50, 0.60, 0.60], [0.0, 1.0, 1.0, 0.0], color='lightgreen', alpha=0.5)
#     plt.fill([0.60, 0.60, 0.70, 0.70], [0.0, 1.0, 1.0, 0.0], color='lightcoral', alpha=0.5)
#     plt.legend(loc="upper right")
#     #plt.show()
#     plt.savefig("OpricalEffeciency.png")
#     plt.cla()   # clear the current axes
#     plt.clf()   # clear the current figure
#     plt.close() # closes the current figure

# # In[111]:


# wl = []
# Tr = []
# Tg = []
# Tb = []
# for i in range(nfreq):
#     wl = np.append(wl, 1 / (flux_freqs[i] * um_scale))
#     Tr = np.append(Tr, red_flux[i] / total_flux[i])
#     Tg = np.append(Tg, green_flux[i] / total_flux[i])
#     Tb = np.append(Tb, blue_flux[i] / total_flux[i])

# if mp.am_master():
#     plt.figure(dpi=150)
#     plt.plot(wl, Tr, "r", label="Redpixel")
#     plt.plot(wl, Tg, "g", label="Greenpixel1")
#     plt.plot(wl, Tb, "b", label="Bluepixel")
    
#     plt.axis([0.40, 0.70, 0, 1])
#     plt.xlabel("wavelength (μm)")
#     plt.ylabel("Transmittance")
#     plt.fill([0.40, 0.40, 0.50, 0.50], [0.0, 1.0, 1.0, 0.0], color='lightblue', alpha=0.5)
#     plt.fill([0.50, 0.50, 0.60, 0.60], [0.0, 1.0, 1.0, 0.0], color='lightgreen', alpha=0.5)
#     plt.fill([0.60, 0.60, 0.70, 0.70], [0.0, 1.0, 1.0, 0.0], color='lightcoral', alpha=0.5)
#     plt.legend(loc="upper right")
#     #plt.show()
#     plt.savefig("Transmission.png")
#     plt.cla()   # clear the current axes
#     plt.clf()   # clear the current figure
#     plt.close() # closes the current figure








