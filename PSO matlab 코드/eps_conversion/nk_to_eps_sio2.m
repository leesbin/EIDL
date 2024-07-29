% Convert  Wavelength_N_K  to Freq_eps1_eps2
% Author: Haejun Chung
% Date: 2013
%
% References
%%%%%%%%%%%%%%%%%%%%% Unit conversion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j=complex(0,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numcomp=1; % 1 for e^jwt, 2 fore^-iwt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% Measurement Tissue Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=1.2566370614*10^(-6);
eps0=8.854187*10^(-12);
c=1/sqrt(mu*eps0);

%%%%%%%%%%%%%%%% conversion 할 경험적 데이터(n, k) 불러오기 %%%%%%%%%%%%%%%%%

data = xlsread('SiO2.csv');
meep_a = 10e-6; % MEEP_unit=10 um
material_n.lambda=data(:,1)*10^-6;
material_n.n=data(:,2); % n 파장 영역 (um)

data = xlsread('SiO2.csv');
material_k.lambda=data(:,1)*10^-6;
material_k.k=data(:,3); % k 파장 영역 (um)
%%%%%%%%%%%%%%%%%%%%%%%%%% 유전율 데이터 계산 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

material.lambda= linspace(8e-6,13e-6,100); % 8 ~ 13um 영역 100 등분
material.freq=c./material.lambda; % 주파수 영역으로 변환
material.n=csapi(material_n.lambda,material_n.n,material.lambda);% n 배치
material.kappa=csapi(material_k.lambda,material_k.k,material.lambda);% k 배치

material.eps1=material.n.^(2)-material.kappa.^(2); % Re(eps)
material.eps2=2*material.n.*material.kappa; % Im(eps)

for i = 1: length(material.lambda)% 파장영역(8 ~ 13 um)에 계산된 유전율 데이터 배치
material.eps1=material.n.^(2)-material.kappa.^(2); % Re(eps)
material.eps2=2*material.n.*material.kappa; % Im(eps)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% 데이터 작성 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp100=fopen('SiO2_lambda_eps1_eps2.txt','w');% 파장영역 유전율 데이터
fp101=fopen('SiO2_meep_a_eps1_eps2.txt','w');% MEEP_파장영역 유전율 데이터
fp102=fopen('SiO2_meep_f_eps1_eps2.txt','w');% MEEP_주파수영역 유전율 데이터

%%%%%%%%%%%%%%%%%%%%%%%%%%% Unit conversion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1 : length(material.lambda)   
    fprintf(fp100,'%e\t%e\t%e\t\n',material.lambda(i),material.eps1(i),material.eps2(i));
    fprintf(fp101,'%e\t%e\t%e\t\n',material.lambda(i)/meep_a,material.eps1(i),material.eps2(i));
    fprintf(fp102,'%e\t%e\t%e\t\n',1/(material.lambda(i)/meep_a),material.eps1(i),material.eps2(i));
end
fclose(fp100);
fclose(fp101);
fclose(fp102);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 작성 완료 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% 각 영역의 유전율 데이터를 그림으로 표시 %%%%%%%%%%%%%%%%%
figure
plot(material.lambda./1e-6,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./1e-6,material.eps2, 'b-','linewidth',2);
xlabel('Wavelength [\mum]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_eps12','-dpng'); % 파장영역 유전율 데이터 곡선

figure
plot(material.lambda./meep_a,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./meep_a,material.eps2, 'b-','linewidth',2);
xlabel('Wavelength [c/a]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_over_a_eps12','-dpng'); % MEEP_파장영역 유전율 데이터 곡선

figure
plot(1./(material.lambda./meep_a),material.eps1, 'r-','linewidth',2);
hold on
plot(1./(material.lambda./meep_a),material.eps2, 'b-','linewidth',2);
xlabel('Frequency [a/c]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('freq_eps12','-dpng'); %MEEP_주파수영역 유전율 데이터 곡선


