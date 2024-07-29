# GDS generator
# need numpy array

import numpy as np
import gdspy
from autograd import numpy as npa



################################################### design parameter #########################
resolution = 50 # 해상도
Sx= 1.0# 높이: 위아래 pml 0부터 LN, 그아래는 SiO2
Sy= 3.0# 너비: 좌우 pml, 폭 3um                         
Sz= 3.0# 길이 (진행방향): 3 
LNwg_h= 0.5 # LN waveguide

# 설계 공간
design_region_x = LNwg_h
design_region_y = Sy
design_region_z = 2
Nx = int(resolution * design_region_x) + 1
Ny = int(resolution * design_region_y) + 1
Nz = int(resolution * design_region_z) + 1
##############################################################################################



# Numpy 배열 생성 2D 배열을 만듭니다
structure_weight = np.loadtxt('lastdesign.txt')
structure_weight =structure_weight.reshape(Nx, Ny*Nz)[Nx-1]
data = npa.rot90(structure_weight.reshape(Ny,Nz))


# GDS 파일 생성
cell = gdspy.Cell('TOP')

# 직사각형의 가로와 세로 크기
width = 1
height = 1

# Numpy 배열을 기반으로 GDS의 경로(Path) 생성
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] == 1:
            # 각각의 1에 대해 직사각형을 만듭니다.
            rectangle = gdspy.Rectangle((j * width, -i * height), ((j + 1) * width, -(i + 1) * height))
            cell.add(rectangle)

# GDS 파일 저장
gdspy.write_gds('output.gds')

# Optionally, save an image of the cell as SVG.
cell.write_svg('outpu.svg')

# Display all cells using the internal viewer.
gdspy.LayoutViewer()