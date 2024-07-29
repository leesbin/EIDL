%%%%%%%%%%%%%%%%%%%% Unit conversion code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j=complex(0,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numcomp=1; % 1 for e^jwt, 2 fore^-iwt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% Measurement Tissue Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=1.2566370614*10^(-6);
eps0=8.854187*10^(-12);
c=1/sqrt(mu*eps0);
FileName = ['Si.txt'];% conversion 할 분산물질 데이터(n, k)파일.
data = load(FileName);

%%%%%%%%%%%%%% v2 = xlsread('vh15deg.csv');
material_n.lambda=data(:,1)*10^-9; % Wave length
material_n.n=data(:,2); % n data

FileName = ['Si.txt'];% conversion 할 분산물질 데이터(n, k)파일.
data = load(FileName);

%%%%%%%%%%%%%% 
material_k.lambda=data(:,1)*10^-9; % Wave length
material_k.k=data(:,3); % k data

material.lambda= linspace(400e-9,700e-9,100);
material.freq=c./material.lambda;
material.n=csapi(material_n.lambda,material_n.n,material.lambda);
material.kappa=csapi(material_k.lambda,material_k.k,material.lambda);

material.eps1=material.n.^(2)-material.kappa.^(2); % re(eps)
material.eps2=2*material.n.*material.kappa; % im(eps)

for i = 1: length(material.lambda)

material.eps1=material.n.^(2)-material.kappa.^(2); % re(eps)
material.eps2=2*material.n.*material.kappa; % im(eps)
material.omega=2*pi*material.freq; % frequency

end
%%%%%%%%%%%%%%%%%%%%%%%%%% 데이터 작성 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fp100=fopen('CdSe_lambda_eps1_eps2.txt','w'); % 파장 기반 유전율 데이터
%fp101=fopen('CdSe_meep_a_eps1_eps2.txt','w'); % Meep 파장 기반 유전율 데이터
fp102=fopen('Si_meep_f_eps1_eps2.txt','w'); % Meep 주파수 기반 유전율 데이터


for i = 1 : length(material.lambda)

   % j=length(material.lamda)-i+1;
    % fprintf(fp100,'%e\t%e\t%e\t\n',material.freq(i),material.eps1(i),material.eps2(i));
    % fprintf(fp101,'%e\t%e\t%e\t\n',material.lambda(i)/500e-9,material.eps1(i),material.eps2(i));
    fprintf(fp102,'%e\t%e\t%e\t\n',1/(material.lambda(i)/1e-6),material.eps1(i),material.eps2(i));
end
% fclose(fp100);
% fclose(fp101);
fclose(fp102);
%%%%%%%%%%%%%%%%%%%%%%% 데이터 작성 완료 %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% 각 데이터를 figure 로 표시 %%%%%%%%%%%%%%%%%%
figure
%plot(fx./1e12,Absorption_1, 'r','linewidth',2);
plot(material.lambda./1e-6,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./1e-6,material.eps2, 'b-','linewidth',2);
xlabel('Wavelength [nm]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_eps12','-dpng');% 파장 기반 유전율 데이터

figure
plot(material.lambda./1e-6,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./1e-6,material.eps2, 'b-','linewidth',2);
xlabel('Wavelength [c/a]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_over_a_eps12','-dpng');% Meep 파장 기반 유전율 데이터


figure
plot(1./(material.lambda./1e-6),material.eps1, 'r-','linewidth',2);
hold on
plot(1./(material.lambda./1e-6),material.eps2, 'b-','linewidth',2);
xlabel('Frequency [a/c]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('freq_eps12','-dpng');% Meep 주파수 기반 유전율 데이터

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%