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

%%%%%%%%%%%%%%%% conversion �� ������ ������(n, k) �ҷ����� %%%%%%%%%%%%%%%%%

data = xlsread('SiO2.csv');
meep_a = 10e-6; % MEEP_unit=10 um
material_n.lambda=data(:,1)*10^-6;
material_n.n=data(:,2); % n ���� ���� (um)

data = xlsread('SiO2.csv');
material_k.lambda=data(:,1)*10^-6;
material_k.k=data(:,3); % k ���� ���� (um)
%%%%%%%%%%%%%%%%%%%%%%%%%% ������ ������ ��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

material.lambda= linspace(8e-6,13e-6,100); % 8 ~ 13um ���� 100 ���
material.freq=c./material.lambda; % ���ļ� �������� ��ȯ
material.n=csapi(material_n.lambda,material_n.n,material.lambda);% n ��ġ
material.kappa=csapi(material_k.lambda,material_k.k,material.lambda);% k ��ġ

material.eps1=material.n.^(2)-material.kappa.^(2); % Re(eps)
material.eps2=2*material.n.*material.kappa; % Im(eps)

for i = 1: length(material.lambda)% ���念��(8 ~ 13 um)�� ���� ������ ������ ��ġ
material.eps1=material.n.^(2)-material.kappa.^(2); % Re(eps)
material.eps2=2*material.n.*material.kappa; % Im(eps)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% ������ �ۼ� ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp100=fopen('SiO2_lambda_eps1_eps2.txt','w');% ���念�� ������ ������
fp101=fopen('SiO2_meep_a_eps1_eps2.txt','w');% MEEP_���念�� ������ ������
fp102=fopen('SiO2_meep_f_eps1_eps2.txt','w');% MEEP_���ļ����� ������ ������

%%%%%%%%%%%%%%%%%%%%%%%%%%% Unit conversion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1 : length(material.lambda)   
    fprintf(fp100,'%e\t%e\t%e\t\n',material.lambda(i),material.eps1(i),material.eps2(i));
    fprintf(fp101,'%e\t%e\t%e\t\n',material.lambda(i)/meep_a,material.eps1(i),material.eps2(i));
    fprintf(fp102,'%e\t%e\t%e\t\n',1/(material.lambda(i)/meep_a),material.eps1(i),material.eps2(i));
end
fclose(fp100);
fclose(fp101);
fclose(fp102);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �ۼ� �Ϸ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% �� ������ ������ �����͸� �׸����� ǥ�� %%%%%%%%%%%%%%%%%
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
print('lambda_eps12','-dpng'); % ���念�� ������ ������ �

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
print('lambda_over_a_eps12','-dpng'); % MEEP_���念�� ������ ������ �

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
print('freq_eps12','-dpng'); %MEEP_���ļ����� ������ ������ �


