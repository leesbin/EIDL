#!/usr/bin/env python
# coding: utf-8

# In[1]:
#First FoM: 9.41166973008889
#Current FoM: 275.12673310494995

#Elapsed run time = 13376.2922 s


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


src_0 = mp.GaussianSource(frequency=fcen_red1, fwidth=1 /(0.60 * um_scale) - 1/(0.70 * um_scale), is_integrated=True)

src_1 = mp.GaussianSource(frequency=fcen_green1, fwidth=1 /(0.50 * um_scale) - 1/(0.60 * um_scale), is_integrated=True)

src_2 = mp.GaussianSource(frequency=fcen_blue1, fwidth=1 /(0.40 * um_scale) - 1/(0.50 * um_scale), is_integrated=True)


source = [mp.Source(src_0, component=mp.Ey, size=source_size, center=source_center, amplitude = 1),
          mp.Source(src_1, component=mp.Ey, size=source_size, center=source_center, amplitude = 1.397612003),
          mp.Source(src_2, component=mp.Ey, size=source_size, center=source_center, amplitude = 2.090177677),]


# In[10]:


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
sim = mp.Simulation(
    cell_size=cell_size, # cell_size = mp.Vector3(Sx, Sy)
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # 빈공간
    resolution=resolution,
    k_point = mp.Vector3(0,0,0)
)


# In[14]:


# 모니터 위치와 크기 설정 (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-Sx/4, Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(-Sx/4, -Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(Sx/4, -Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 
monitor_position_3, monitor_size_3 = mp.Vector3(Sx/4, Sy/4, -2 - 0.5/resolution), mp.Vector3(0.01,0.01,0) 

# FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_2 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ey,yee_grid=True)
FourierFields_3 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_4 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_5 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ey,yee_grid=True)
FourierFields_6 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_7 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_8 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ey,yee_grid=True)
FourierFields_9 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_10 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
FourierFields_11 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_3,size=monitor_size_3),mp.Ey,yee_grid=True)
ob_list = [FourierFields_0, FourierFields_1, FourierFields_2,FourierFields_3, FourierFields_4, FourierFields_5 ,FourierFields_6, FourierFields_7, FourierFields_8,FourierFields_9, FourierFields_10, FourierFields_11]

# In[16]:


# J : Objective function
# FourierFields가 측정한 필드, 모니터의 중심에서 Ez 구성요소의 절댓값을 제곱한 값을 취한 후 평균을 계산하여 평균 강도를 계산
def J_0(fields_0, fields_1, fields_2,fields_3, fields_4, fields_5,fields_6, fields_7, fields_8,fields_9, fields_10, fields_11):
    return npa.mean(npa.abs(fields_0[6,:]) ** 2) + npa.mean(npa.abs(fields_1[7,:]) ** 2) + npa.mean(npa.abs(fields_2[8,:]) ** 2) + npa.mean(npa.abs(fields_3[3,:]) ** 2) + npa.mean(npa.abs(fields_4[4,:]) ** 2) + npa.mean(npa.abs(fields_5[5,:]) ** 2) + npa.mean(npa.abs(fields_6[0,:]) ** 2) + npa.mean(npa.abs(fields_7[1,:]) ** 2) + npa.mean(npa.abs(fields_8[2,:]) ** 2) + npa.mean(npa.abs(fields_9[3,:]) ** 2) + npa.mean(npa.abs(fields_10[4,:]) ** 2) + npa.mean(npa.abs(fields_11[5,:]) ** 2)




# In[17]:


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J_0],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-1,
)



# In[18]:


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

    plt.figure()
    ax = plt.gca()
    opt.plot2D(True, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)))
    circ = Circle((2, 2), minimum_length / 2)
    ax.add_patch(circ)
    ax.axis("off")
    np.savetxt("structure_0"+str(numevl) +".txt", design_variables.weights)
    #np.save("structure_0"+str(numevl), design_variables.weights)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    

    plt.figure()
    ax = plt.gca()
    opt.plot2D(True, output_plane = mp.Volume(size = (np.inf, np.inf, 0), center = (0,0,2)))
    circ = Circle((2, 2), minimum_length / 2)
    ax.add_patch(circ)
    ax.axis("off")
    #np.savetxt("structure_00"+str(numevl) +".txt", design_variables.weights)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
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


plt.figure()

plt.plot(evaluation_history, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig("result.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure



#np.save("lastdesign", design_variables.weights)
np.savetxt("lastdesign.txt", design_variables.weights)



# In[22]:



intensities_0 = np.abs(opt.get_objective_arguments()[0][:,1]) ** 2
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


intensities_1 = np.abs(opt.get_objective_arguments()[4][:,1]) ** 2
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

intensities_2 = np.abs(opt.get_objective_arguments()[8][:,1]) ** 2
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


intensities_3 = np.abs(opt.get_objective_arguments()[9][:,1]) ** 2
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

