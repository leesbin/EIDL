import meep as mp
import numpy as np
import time

mp.verbosity(1)

# 해상도 및 사이즈 설정
resolution = 20 
Lpml = 0.4 # PML 영역 크기
pml_layers = [mp.PML(thickness=Lpml, direction=mp.Z)]

# 전체 공간
Sx = 20
Sy = 20
Sz = 50
cell_size = mp.Vector3(Sx, Sy, Sz)

# source 설정
width = 0.4
fcen_red = 1 / (0.625)
fwidth_red = fcen_red * width

src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)
source_center = [0, 0, Sz / 2 - Lpml - 1]  # Source 위치
source_size = mp.Vector3(Sx, Sy, 0)
source = [
    mp.Source(src_0, component=mp.Ex, size=source_size, center=source_center),
    mp.Source(src_0, component=mp.Ey, size=source_size, center=source_center)
]

# 시뮬레이션 세팅
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=source,
    resolution=resolution,
    k_point=mp.Vector3(0, 0, 0)
)

# 실행 시간 측정 및 반복
num_runs = 5
times = []

for i in range(num_runs):
    start_time = time.time()
    sim.reset_meep()  # 시뮬레이션 초기화
    sim.run(until=100)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    print(f"Run {i+1}/{num_runs}: Elapsed time = {elapsed_time:.2f} s")

# 평균 실행 시간 계산
average_time = np.mean(times)
std_dev_time = np.std(times)

print(f"\nAverage run time over {num_runs} runs = {average_time:.2f} s")
print(f"Standard deviation of run time = {std_dev_time:.2f} s")

# 결과를 파일에 저장
with open("performance_results.txt", "w") as f:
    f.write(f"Average run time over {num_runs} runs = {average_time:.2f} s\n")
    f.write(f"Standard deviation of run time = {std_dev_time:.2f} s\n")
    f.write(f"Individual run times: {times}\n")
