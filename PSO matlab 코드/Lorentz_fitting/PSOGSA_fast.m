%PSOGSA source code v2.0, Generated by SeyedAli Mirjalili, 2011. 
%Adopted from: S. Mirjalili, S.Z. Mohd Hashim, A New Hybrid PSOGSA 
%Algorithm for Function Optimization, in IEEE International Conference 
%on Computer and Information Application?ICCIA 2010), China, 2010, pp. 374-377.


                              %-------------------------------------------%
                              %         Evaluate the population           %           
                              %-------------------------------------------%                                 %--%
 %%% Variables %%%

%current_position:  Position of particles
%velocity:          Velocity
%force:             The gravitational force between the particles ?? 중력??
%acceleration:      Acceleration
%mass:              Mass /질량
%dim:               Dimension of test functions
%n:                 Number of particles
%G0                 Gravitational constant
%low:               The lower bound of the search space
%up:                The higher bound of the search space


function [gBestScore,gBest,GlobalBestCost,best_fit_position]=PSOGSA_fast(Benchmark_Function_ID,n,iteration,FileName)

[low,up,dim]=benchmark_functions_details(Benchmark_Function_ID,FileName);%define the boundary and dimension of the benchmark function

current_fitness =zeros(n,1);
gBest=zeros(1,dim);
gBestScore=inf;

for i=1:n
        pBestScore(i)=inf;
end
        pBest=zeros(n,dim);
%% Initialize
G0=1;                                          % gravitational constant
current_position = rand(n,dim).*(up-low)+low;  %initial positions in the problem's boundary
%particle_track=zeros(iteration+1,dim);
velocity = .3*randn(n,dim) ;
acceleration=zeros(n,dim);
mass(n)=0;
force=zeros(n,dim);
%% 인지율과 사회적비율
C1=2; %C1 in Equation (9)
C2=2; %C2 in Equation (9)

%% main loop
iter = 0 ;                  % Iterations?counter
while  ( iter < iteration )
    % 반복문 내부에서 현재 위치(current_position) 업데이트 전에 제약 조건을 적용
    for i = 1:n
        % 현재 위치를 저장
        currentPosition = current_position(i,:);

        % 제약 조건 확인 및 조정
        % 예를 들어, x1과 x4가 -4.5에서 4.5 사이에 있도록 제약을 둔다면:
        if currentPosition(1) < -4
            currentPosition(1) = -4;
        elseif currentPosition(1) > 4
            currentPosition(1) = 4;
        end

        if currentPosition(4) < -4
            currentPosition(4) = -4;
        elseif currentPosition(4) > 4
            currentPosition(4) = 4;
        end

        if currentPosition(7) < -4
            currentPosition(7) = -4;
        elseif currentPosition(7) > 4
            currentPosition(7) = 4;
        end

        % 업데이트된 위치를 다시 current_position에 할당
        current_position(i,:) = currentPosition;
    end
% particle_track(iter+1,:) = current_position(3,:);  % tracking  Num 3 particle
G=G0*exp(-23*iter/iteration); %Equation (4)
iter = iter + 1;
iter;
force=zeros(n,dim);
mass(n)=0;
acceleration=zeros(n,dim);

for i = 1:n
    fitness=0;
    %///Bound the search Space///
     %Tp=current_position(i,:)>up;Tm=current_position(i,:)<low;current_position(i,:)=(current_position(i,:).*(~(Tp+Tm)))+up.*Tp+low.*Tm;                     
    %////////////////////////////
%    
                                 %-------------------------------------------%
                                 %         Evaluate the population           %           
                                 %-------------------------------------------%      
                                 % Lorentz model로 적합성 평가 이때 평가 기준값은 오차율(최소로)
    fitness=benchmark_functions(current_position(i,:),Benchmark_Function_ID,dim,FileName);
    current_fitness(i)=fitness;   
        
    if(pBestScore(i)>fitness)
        pBestScore(i)=fitness;
        pBest(i,:)=current_fitness(i,:);
%         if(iter<10)
%            pBest 
%         end
    end
    if(gBestScore>fitness)
    gBestScore=fitness;
    gBest=current_position(i,:);

    % % x1과 x4 위치만을 따로 처리하여 값 제한
    % if abs(gBest(1)) > 4.5
    %     gBest(1) = sign(gBest(1)) * 4.5;
    % end
    % 
    % if abs(gBest(4)) > 4.5
    %     gBest(4) = sign(gBest(4)) * 4.5;
    % end
    % 
    % if abs(gBest(7)) > 4.5
    %     gBest(7) = sign(gBest(7)) * 4.5;
    % end
end
    
end
%particle_track(iter+1,:) = current_position(3,:);  % tracking  Num 3 particle
best=min(current_fitness);
worst=max(current_fitness);

        GlobalBestCost(iter)=gBestScore;
        GlobalBestCost(iter);
        best;

    for pp=1:n
        if current_fitness(pp)==best
            break;
        end
        
    end
    
    bestIndex=pp;
            
    for pp=1:dim
        best_fit_position(iter,1)=best;
        best_fit_position(iter,pp+1)=current_position(bestIndex,pp);   
    end


                                               %-------------------%
                                               %   Calculate Mass  %
                                               %-------------------%
    for i=1:n
    mass(i)=(current_fitness(i)-0.99*worst)/(best-worst);    
end

for i=1:n
    mass(i)=mass(i)*5/sum(mass);    
    
end

                                               %-------------------%
                                               %  Force    update  %
                                               %-------------------%

for i=1:n
    for j=1:dim
        for k=1:n
            if(current_position(k,j)~=current_position(i,j))
                % Equation (3)
                force(i,j)=force(i,j)+ rand()*G*mass(k)*mass(i)*(current_position(k,j)-current_position(i,j))/abs(current_position(k,j)-current_position(i,j));
            end
        end
    end
end
                                               %------------------------------------%
                                               %  Accelations $ Velocities  UPDATE  %
                                               %------------------------------------%

for i=1:n
       for j=1:dim
            if(mass(i)~=0)
                %Equation (6)
                acceleration(i,j)=force(i,j)/mass(i);
            end
       end
end   

for i=1:n
        for j=1:dim
            %Equation(9)
            velocity(i,j)=rand()*velocity(i,j)+C1*rand()*acceleration(i,j) + C2*rand()*(gBest(j)-current_position(i,j));
        end
end
                                               %--------------------------%
                                               %   positions   UPDATE     %
                                               %--------------------------%
                                                        
%Equation (10) 
current_position = current_position + velocity; 

end
current_position
end