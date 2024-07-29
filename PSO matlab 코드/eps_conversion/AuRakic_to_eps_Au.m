% Convert  Wavelength_N_K  to Freq_eps1_eps2
% Author: Haejun Chung
% Date: 2013
%
% References


j=complex(0,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numcomp=1; % 1 for e^jwt, 2 fore^-iwt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% Measurement Tissue Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%

const=MATLAB_CONSTANTS;
lambda=linspace(400e-9,600e-9,100);  %% input wavelength
omega=const.c./lambda.*2.*pi;
eps=Au_Palik_EpsInterp(omega);  %% Au epsilon dividied by water epsilon

figure
%plot(fx./1e12,Absorption_1, 'r','linewidth',2);
plot(lambda./1e-9,real(eps), 'r-','linewidth',2);
hold on
plot(lambda./1e-9,imag(eps), 'b-','linewidth',2);
%plot(c_0./material.freq./1e-9,total_absorption, 'ko','linewidth',2, 'MarkerSize',6);
%plot(lam_x./1e-9,smooth(Absorption_asi), 'b-','linewidth',2);
%plot(lam_x./1e-9,smooth(Absorption_csi), 'k-','linewidth',2);
xlabel('Wavelength [nm]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_eps12','-dpng');


% 
% %FileName = ['a_si.txt'];
% mu=1.2566370614*10^(-6);
% eps0=8.854187*10^(-12);
% c=1/sqrt(mu*eps0);
% FileName = ['Au_flimetric.txt'];
% data = load(FileName);
% 
% %%%%%%%%%%%%%% 
% material_n.lambda=data(:,1)*10^-9;
% material_n.n=data(:,2); % 
% 
% FileName = ['Au_flimetric.txt'];
% data = load(FileName);
% 
% %%%%%%%%%%%%%% 
% material_k.lambda=data(:,1)*10^-9;
% material_k.k=data(:,3); % 
% 
%material.lamda= linspace(300e-9,1100e-9,100);
material.lambda= lambda;
material.freq=const.c./material.lambda;

% material.eps1=material.n.^(2)-material.kappa.^(2); % 
% material.eps2=2*material.n.*material.kappa; % im(eps)
%material.omega=2*pi*material.freq; % frequency

for i = 1: length(material.lambda)
%material.n(i)=
material.eps1=real(eps); % 
material.eps2=imag(eps);
material.omega=2*pi*material.freq; % frequency

%material.a(i)= 4*pi*material.k(i)/(material.lamda(i))/100;
end

% col 1  =  wavelength (nm)
% col 2  =  Index of refraction (n)
% col 3  =  extinction coefficient (k)
% (n_bar = n + i¥ê) 
%
%  Conversion between refractive index and dielectric constant is done by:
%   epsilon_1= n^2 - kappa^2
%   epsilon_2 = 2n * kappa
% where \tilde\epsilon=\epsilon_1+i\epsilon_2= (n+i\kappa)^2.
%fp100=fopen('silicon_substrate_1_freq_2_eps1_3_eps2.txt','w');
fp100=fopen('Au_eps1_eps2.txt','w');
fp101=fopen('Au_a_eps1_eps2.txt','w');
fp102=fopen('Au_f_eps1_eps2.txt','w');






% eps=re(eps)-j*im(eps)
%material.freq=299792458./(data(:,1).*(10)^(-9)); % frequency
 
%%%%%%%%%%%%%%
% epsinf=0;   %%%%%%%%%% Not imporant for quadratic rational function (but important for Debye model) 
% hh=size(data(:,1));
% epsinfm=epsinf*ones(hh(:,1),hh(:,2));
% %%%%%%%%%%%%%% 
% material.lamda=data(:,1);
% material.n=data(:,2); % 
% material.kappa=data(:,3); % 


% material.eps1=material.n.^(2)-material.kappa.^(2); % 
% material.eps2=2*material.n.*material.kappa; % im(eps)
% material.omega=2*pi*material.freq; % frequency
%   fprintf(fp100,'layer # \n');
%     fprintf(fp100,'%d \n', length(material.a));
% % col1 = micro meter, col2= absorption
% j=0;
for i = 1 : length(material.lambda)

   % j=length(material.lamda)-i+1;
    fprintf(fp100,'%e\t%e\t%e\t\n',material.freq(i),material.eps1(i),material.eps2(i));
    fprintf(fp101,'%e\t%e\t%e\t\n',material.lambda(i)/500e-9,material.eps1(i),material.eps2(i));
    fprintf(fp102,'%e\t%e\t%e\t\n',1/(material.lambda(i)/500e-9),material.eps1(i),material.eps2(i));
end
fclose(fp100);
fclose(fp101);
fclose(fp102);

figure
%plot(fx./1e12,Absorption_1, 'r','linewidth',2);
plot(material.lambda./1e-9,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./1e-9,material.eps2, 'b-','linewidth',2);
%plot(c_0./material.freq./1e-9,total_absorption, 'ko','linewidth',2, 'MarkerSize',6);
%plot(lam_x./1e-9,smooth(Absorption_asi), 'b-','linewidth',2);
%plot(lam_x./1e-9,smooth(Absorption_csi), 'k-','linewidth',2);
xlabel('Wavelength [nm]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_eps12','-dpng');

figure
%plot(fx./1e12,Absorption_1, 'r','linewidth',2);
plot(material.lambda./500e-9,material.eps1, 'r-','linewidth',2);
hold on
plot(material.lambda./500e-9,material.eps2, 'b-','linewidth',2);
%plot(c_0./material.freq./1e-9,total_absorption, 'ko','linewidth',2, 'MarkerSize',6);
%plot(lam_x./1e-9,smooth(Absorption_asi), 'b-','linewidth',2);
%plot(lam_x./1e-9,smooth(Absorption_csi), 'k-','linewidth',2);
xlabel('Wavelength [a/c]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('lambda_over_a_eps12','-dpng');


figure
%plot(fx./1e12,Absorption_1, 'r','linewidth',2);
plot(1./(material.lambda./500e-9),material.eps1, 'r-','linewidth',2);
hold on
plot(1./(material.lambda./500e-9),material.eps2, 'b-','linewidth',2);
%plot(c_0./material.freq./1e-9,total_absorption, 'ko','linewidth',2, 'MarkerSize',6);
%plot(lam_x./1e-9,smooth(Absorption_asi), 'b-','linewidth',2);
%plot(lam_x./1e-9,smooth(Absorption_csi), 'k-','linewidth',2);
xlabel('Frequency [c/a]','fontname', 'arial','fontweight','bold','fontsize', 16)
ylabel('eps','Rotation',90,'fontname', 'arial','fontweight','bold','fontsize', 16)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
legend('eps1','eps2') 
legend boxoff
print('freq_eps12','-dpng');

%set(gca,'XLim',[400 1100])
%set(gca,'YLim',[0 1])
% legend('eps1','eps2','arial','fontsize',16,'fontweight','bold','Location','northeast') 
% legend boxoff
% print -dpng -r300 -adobecset eps12
