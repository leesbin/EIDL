close all
clear all
clc
%% 실행 횟수
run=1;  
%% 폴 갯수(1~3개가 이상적 클수록 Lorentz 연산량이 증가함) 
N_pole = 4;
%% 차원 수 = 폴 갯수 * 3(gamma_n, sigma_n, omega_n)
dim=3 * N_pole ;
%% 입자수 , 최대 반복횟수
N = 40;                        % Size of the swarm " no of objects "
Max_Iteration  = 600;              % Maximum number of "iterations"
%% 소재 데이터 선택 
FileName = ['Si_meep_f_eps1_eps2.txt'];
%% 사용할 benchmark_function ID 설정 (바운더리 설정 및 Lorentz_unit 실행)
% *이때 details.m 파일에서 바운더리 설정(차원수도 맞춰주어야함) 
Benchmark_Function_ID=28; %Benchmark function ID  27 for non_negative QCRF, 25 for normal
%% gbest 행렬 생성
gBest_Matrix=zeros(run,dim);
gBestScore_Matrix=zeros(run,1);
for i= 1:run
%% PSOGSA_fast 실행(PSO 알고리즘) >> gbest행렬 및 최적값 받아오기
[gBestScore,gBest,GlobalBestCost,best_fit_position]= PSOGSA_fast(Benchmark_Function_ID, N, Max_Iteration,FileName);
gBest_Matrix(i,:)=gBest;
gBestScore_Matrix(i,:)=gBestScore;
end
%% 1열을 최적화된 Fitness값, 나머지는 그에 맞는 gbest들로 구성된 (run횟수 X 차원수+1(각 차원의 gamma들))행렬 생성
Post_processing = zeros(run,dim+1);
Post_processing(:,1) = gBestScore_Matrix(:,1);
for i = 1 : run
   for j = 1: dim
      Post_processing(i,j+1) = gBest_Matrix(i,j);
   end
end
%% 1열기준 오름차순 정렬: 제일 낮은 Fitness값(오차)을 가진 행으로 구성된 1행 크기의 행렬 생성 
Post_processing=sortrows(Post_processing,1);
Gamma = zeros(1,dim);
%% 위 행렬의 gamma(gbest값)들을 모두 Gamma(최적의 결과)에 저장
for i = 1 : dim
    Gamma(i) = Post_processing(1,i+1);
end
%% 그래프 그리기 및 오차 계산
MEEP_unit_Lorentz_plot(Gamma,FileName);