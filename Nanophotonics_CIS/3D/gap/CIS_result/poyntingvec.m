
close all
color=load('redblue.mat').c;


clearvars -except seq color
clc;

rs=25;
X=rs*2.8+1;
Y=rs*2.8+1;

blue = 0
green = 0
red = 0


% ///////////////////////////////////////////////////////////////////////////////////////////////////////
            %blue       
            Hx_b= ["Hx_b_field.h5"];                           %Hx
            hinfo_Hx_b= hdf5info("Hx_b_field.h5");
            Hy_b= ["Hy_b_field.h5"];                           %Hy
            hinfo_Hy_b= hdf5info("Hy_b_field.h5");
            Hz_b= ["Hz_b_field.h5"];                            %Hz
            hinfo_Hz_b= hdf5info("Hz_b_field.h5");

            Ex_b= ["Ex_b_field.h5"];                           %Hx
            hinfo_Ex_b= hdf5info("Ex_b_field.h5");
            Ey_b= ["Ey_b_field.h5"];                           %Hy
            hinfo_Ey_b= hdf5info("Ey_b_field.h5");
            Ez_b= ["Ez_b_field.h5"];                            %Hz
            hinfo_Ez_b= hdf5info("Ez_b_field.h5");

            %green       
            Hx_g= ["Hx_g_field.h5"];                           %Hx
            hinfo_Hx_g= hdf5info("Hx_g_field.h5");
            Hy_g= ["Hy_g_field.h5"];                           %Hy
            hinfo_Hy_g= hdf5info("Hy_g_field.h5");
            Hz_g= ["Hz_g_field.h5"];                            %Hz
            hinfo_Hz_g= hdf5info("Hz_g_field.h5");

            Ex_g= ["Ex_g_field.h5"];                           %Hx
            hinfo_Ex_g= hdf5info("Ex_g_field.h5");
            Ey_g= ["Ey_g_field.h5"];                           %Hy
            hinfo_Ey_g= hdf5info("Ey_g_field.h5");
            Ez_g= ["Ez_g_field.h5"];                            %Hz
            hinfo_Ez_g= hdf5info("Ez_g_field.h5");

            %red       
            Hx_r= ["Hx_r_field.h5"];                           %Hx
            hinfo_Hx_r= hdf5info("Hx_r_field.h5");
            Hy_r= ["Hy_r_field.h5"];                           %Hy
            hinfo_Hy_r= hdf5info("Hy_r_field.h5");
            Hz_r= ["Hz_r_field.h5"];                            %Hz
            hinfo_Hz_r= hdf5info("Hz_r_field.h5");

            Ex_r= ["Ex_r_field.h5"];                           %Hx
            hinfo_Ex_r= hdf5info("Ex_r_field.h5");
            Ey_r= ["Ey_r_field.h5"];                           %Hy
            hinfo_Ey_r= hdf5info("Ey_r_field.h5");
            Ez_r= ["Ez_r_field.h5"];                            %Hz
            hinfo_Ez_r= hdf5info("Ez_r_field.h5");

             
            %blue 
            %Hx
            Name_Hxrb=["/hx_0.r"];
            Name_Hxib=["/hx_0.i"];    
            Hx_data_rb = h5read(Hx_b,Name_Hxrb);          %Real
            Hx_data_ib = h5read(Hx_b,Name_Hxib);           %Imagine
            Hx_datab = Hx_data_rb + Hx_data_ib * 1.0i;       %Complex
            % Hy
            Name_Hyrb=["/hy_0.i"];
            Name_Hyib=["/hy_0.i"];    
            Hy_data_rb = h5read(Hy_b,Name_Hyrb);          %Real
            Hy_data_ib = h5read(Hy_b,Name_Hyib);           %Imagine
            Hy_datab = Hy_data_rb + Hy_data_ib * 1.0i;       %Complex
            % Hz
            Name_Hzrb=["/hz_0.r"];
            Name_Hzib=["/hz_0.i"];    
            Hz_data_rb = h5read(Hz_b,Name_Hzrb);          %Real
            Hz_data_ib = h5read(Hz_b,Name_Hzib);           %Imagine
            Hz_datab = Hz_data_rb + Hz_data_ib * 1.0i;       %Complex
            % Ex
            Name_Exrb=["/ex_0.r"];
            Name_Exib=["/ex_0.i"];    
            Ex_data_rb = h5read(Ex_b,Name_Exrb);          %Real
            Ex_data_ib = h5read(Ex_b,Name_Exib);           %Imagine
            Ex_datab = Ex_data_rb + Ex_data_ib * 1.0i;       %Complex
            % Ey
            Name_Eyrb=["/ey_0.r"];
            Name_Eyib=["/ey_0.i"];    
            Ey_data_rb = h5read(Ey_b,Name_Eyrb);          %Real
            Ey_data_ib = h5read(Ey_b,Name_Eyib);           %Imagine
            Ey_datab = Ey_data_rb + Ey_data_ib * 1.0i;       %Complex
            % Ez
            Name_Ezrb=["/ez_0.r"];
            Name_Ezib=["/ez_0.i"];    
            Ez_data_rb = h5read(Ez_b,Name_Ezrb);          %Real
            Ez_data_ib = h5read(Ez_b,Name_Ezib);           %Imagine
            Ez_datab = Ez_data_rb + Ez_data_ib * 1.0i;       %Complex

            %green
            %Hx
            Name_Hxrg=["/hx_0.r"];
            Name_Hxig=["/hx_0.i"];    
            Hx_data_rg = h5read(Hx_g,Name_Hxrg);          %Real
            Hx_data_ig = h5read(Hx_g,Name_Hxig);           %Imagine
            Hx_datag = Hx_data_rg + Hx_data_ig * 1.0i;       %Complex
            % Hy
            Name_Hyrg=["/hy_0.i"];
            Name_Hyig=["/hy_0.i"];    
            Hy_data_rg = h5read(Hy_g,Name_Hyrg);          %Real
            Hy_data_ig = h5read(Hy_g,Name_Hyig);           %Imagine
            Hy_datag = Hy_data_rg + Hy_data_ig * 1.0i;       %Complex
            % Hz
            Name_Hzrg=["/hz_0.r"];
            Name_Hzig=["/hz_0.i"];    
            Hz_data_rg = h5read(Hz_g,Name_Hzrg);          %Real
            Hz_data_ig = h5read(Hz_g,Name_Hzig);           %Imagine
            Hz_datag = Hz_data_rg + Hz_data_ig * 1.0i;       %Complex
            % Ex
            Name_Exrg=["/ex_0.r"];
            Name_Exig=["/ex_0.i"];    
            Ex_data_rg = h5read(Ex_g,Name_Exrg);          %Real
            Ex_data_ig = h5read(Ex_g,Name_Exig);           %Imagine
            Ex_datag = Ex_data_rg + Ex_data_ig * 1.0i;       %Complex
            % Ey
            Name_Eyrg=["/ey_0.r"];
            Name_Eyig=["/ey_0.i"];    
            Ey_data_rg = h5read(Ey_g,Name_Eyrg);          %Real
            Ey_data_ig = h5read(Ey_g,Name_Eyig);           %Imagine
            Ey_datag = Ey_data_rg + Ey_data_ig * 1.0i;       %Complex
            % Ez
            Name_Ezrg=["/ez_0.r"];
            Name_Ezig=["/ez_0.i"];    
            Ez_data_rg = h5read(Ez_g,Name_Ezrg);          %Real
            Ez_data_ig = h5read(Ez_g,Name_Ezig);           %Imagine
            Ez_datag = Ez_data_rg + Ez_data_ig * 1.0i;       %Complex

            %red
            %Hx
            Name_Hxrr=["/hx_0.r"];
            Name_Hxir=["/hx_0.i"];    
            Hx_data_rr = h5read(Hx_r,Name_Hxrr);          %Real
            Hx_data_ir = h5read(Hx_r,Name_Hxir);           %Imagine
            Hx_datar = Hx_data_rr + Hx_data_ir * 1.0i;       %Complex
            % Hy
            Name_Hyrr=["/hy_0.i"];
            Name_Hyir=["/hy_0.i"];    
            Hy_data_rr = h5read(Hy_r,Name_Hyrr);          %Real
            Hy_data_ir = h5read(Hy_r,Name_Hyir);           %Imagine
            Hy_datar = Hy_data_rr + Hy_data_ir * 1.0i;       %Complex
            % Hz
            Name_Hzrr=["/hz_0.r"];
            Name_Hzir=["/hz_0.i"];    
            Hz_data_rr = h5read(Hz_r,Name_Hzrr);          %Real
            Hz_data_ir = h5read(Hz_r,Name_Hzir);           %Imagine
            Hz_datar = Hz_data_rr + Hz_data_ir * 1.0i;       %Complex
            % Ex
            Name_Exrr=["/ex_0.r"];
            Name_Exir=["/ex_0.i"];    
            Ex_data_rr = h5read(Ex_r,Name_Exrr);          %Real
            Ex_data_ir = h5read(Ex_r,Name_Exir);           %Imagine
            Ex_datar = Ex_data_rr + Ex_data_ir * 1.0i;       %Complex
            % Ey
            Name_Eyrr=["/ey_0.r"];
            Name_Eyir=["/ey_0.i"];    
            Ey_data_rr = h5read(Ey_r,Name_Eyrr);          %Real
            Ey_data_ir = h5read(Ey_r,Name_Eyir);           %Imagine
            Ey_datar = Ey_data_rr + Ey_data_ir * 1.0i;       %Complex
            % Ez
            Name_Ezrr=["/ez_0.r"];
            Name_Ezir=["/ez_0.i"];    
            Ez_data_rr = h5read(Ez_r,Name_Ezrr);          %Real
            Ez_data_ir = h5read(Ez_r,Name_Ezir);           %Imagine
            Ez_datar = Ez_data_rr + Ez_data_ir * 1.0i;       %Complex

% ///////////////////////////////////////////////////////////////////////////////////////////////////////

Sx_f=zeros(Y, X); 
Sy_f=zeros(Y, X); 
Sz_f=zeros(Y, X); 
MS_f=zeros(Y, X);% amplitude
PS_f=zeros(Y, X);% phase (basis: phi=0 axis)




% calc poynting vector
for y=1:Y% 열(y)
    for x=1:X% 행(x)
        % E, H (free space)
        E_f=[Ex_datab(y, x), Ey_datab(y, x), Ez_datab(y, x)];
        H_f=[conj(Hx_datab(y,x)), conj(Hy_datab(y,x)), conj(Hz_datab(y,x))];
        P_f=real(cross(E_f, H_f));

        %% magnitude and phase of S vector

        % Free space
        Sx_f(y,x)=P_f(1)*0.5;
        Sy_f(y,x)=P_f(2)*0.5;
        Sz_f(y,x)=P_f(3)*0.5;
        MS_fb(y,x)=real(sqrt(sum(real(E_f .* conj(E_f)))));
        if(Sx_f(y,x)>=0) % - - (-pi/2~ +pi/2)
            PS_f(y,x)=asin(Sy_f(y,x)/MS_fb(y,x));       % +-, ++ (-pi/2~pi/2)  phase of poynting vector
        elseif(Sx_f(y,x)<0)% -+ (pi/2~pi)
            PS_f(y,x)=pi*Sy_f(y,x)/abs(Sy_f(y,x))-asin(Sy_f(y,x)/MS_fb(y,x));
        end

        XXb(1,x)=x;
        YYb(y,1)=y;
        Ub(y,x)=cos(PS_f(y,x));
        Vb(y,x)=sin(PS_f(y,x));


    end
end



% calc poynting vector
for y=1:Y% 열(y)
    for x=1:X% 행(x)
        % E, H (free space)
        E_f=[Ex_datag(y, x), Ey_datag(y, x), Ez_datag(y, x)];
        H_f=[conj(Hx_datag(y,x)), conj(Hy_datag(y,x)), conj(Hz_datag(y,x))];
        P_f=real(cross(E_f, H_f));

        %% magnitude and phase of S vector

        % Free space
        Sx_f(y,x)=P_f(1)*0.5;
        Sy_f(y,x)=P_f(2)*0.5;
        Sz_f(y,x)=P_f(3)*0.5;
        MS_fg(y,x)=real(sqrt(sum(real(E_f .* conj(E_f)))));
        if(Sx_f(y,x)>=0) % - - (-pi/2~ +pi/2)
            PS_f(y,x)=asin(Sy_f(y,x)/MS_fg(y,x));       % +-, ++ (-pi/2~pi/2)  phase of poynting vector
        elseif(Sx_f(y,x)<0)% -+ (pi/2~pi)
            PS_f(y,x)=pi*Sy_f(y,x)/abs(Sy_f(y,x))-asin(Sy_f(y,x)/MS_fg(y,x));
        end

        XXg(1,x)=x;
        YYg(y,1)=y;
        Ug(y,x)=cos(PS_f(y,x));
        Vg(y,x)=sin(PS_f(y,x));


    end
end


% calc poynting vector
for y=1:Y% 열(y)
    for x=1:X% 행(x)
        % E, H (free space)
        E_f=[Ex_datar(y, x), Ey_datar(y, x), Ez_datar(y, x)];
        H_f=[conj(Hx_datar(y,x)), conj(Hy_datar(y,x)), conj(Hz_datar(y,x))];
        P_f=real(cross(E_f, H_f));

        %% magnitude and phase of S vector

        % Free space
        Sx_f(y,x)=P_f(1)*0.5;
        Sy_f(y,x)=P_f(2)*0.5;
        Sz_f(y,x)=P_f(3)*0.5;
        MS_fr(y,x)=real(sqrt(sum(real(E_f .* conj(E_f)))));
        if(Sx_f(y,x)>=0) % - - (-pi/2~ +pi/2)
            PS_f(y,x)=asin(Sy_f(y,x)/MS_fr(y,x));       % +-, ++ (-pi/2~pi/2)  phase of poynting vector
        elseif(Sx_f(y,x)<0)% -+ (pi/2~pi)
            PS_f(y,x)=pi*Sy_f(y,x)/abs(Sy_f(y,x))-asin(Sy_f(y,x)/MS_fr(y,x));
        end

        XXr(1,x)=x;
        YYr(y,1)=y;
        Ur(y,x)=cos(PS_f(y,x));
        Vr(y,x)=sin(PS_f(y,x));


    end
end

%% 2X4 subplot의 각 subplot [(1,1)~(2,4)]에  plot 하고 싶은 값 설정.

%blue color%%%%%%%%%%%%%%%%%%%%%%%%%%
white_color = [1 1 1];  % RGB 값

blue_color = [0 0.4 0.6392];  % RGB 값

black_color = [0.0509 0.3098 0.4588];  % RGB 값

% 그라데이션을 갖는 컬러맵 생성
num_steps = 256;  % 그라데이션 단계 수
r = linspace(white_color(1), blue_color(1), num_steps);
g = linspace(white_color(2), blue_color(2), num_steps);
b = linspace(white_color(3), blue_color(3), num_steps);
r2 = linspace(blue_color(1), black_color(1), num_steps);
g2 = linspace(blue_color(2), black_color(2), num_steps);
b2 = linspace(blue_color(3), black_color(3), num_steps);

% 회색에서 파란색, 그리고 검은색으로 변화하는 커스텀 컬러맵 생성
blue_colormap = [r', g', b'; r2', g2', b2'];

%green color%%%%%%%%%%%%%%%%%%%%%%%%%%
white_color = [1 1 1];  % RGB 값

green_color = [0 0.5 0];  % RGB 값

black_color =  [0.0392 0.3608 0.0392]; % RGB 값

% 그라데이션을 갖는 컬러맵 생성
num_steps = 256;  % 그라데이션 단계 수
r = linspace(white_color(1), green_color(1), num_steps);
g = linspace(white_color(2), green_color(2), num_steps);
b = linspace(white_color(3), green_color(3), num_steps);
r2 = linspace(green_color(1), black_color(1), num_steps);
g2 = linspace(green_color(2), black_color(2), num_steps);
b2 = linspace(green_color(3), black_color(3), num_steps);

green_colormap = [r', g', b'; r2', g2', b2'];

%red color%%%%%%%%%%%%%%%%%%%%%%%%%%
white_color = [1 1 1];  % RGB 값

red_color = [1 0 0]; % RGB 값

black_color = [0.7216 0.0784 0.0784]; % RGB 값

% 그라데이션을 갖는 컬러맵 생성
num_steps = 256;  % 그라데이션 단계 수
r = linspace(white_color(1), red_color(1), num_steps);
g = linspace(white_color(2), red_color(2), num_steps);
b = linspace(white_color(3), red_color(3), num_steps);
r2 = linspace(red_color(1), black_color(1), num_steps);
g2 = linspace(red_color(2), black_color(2), num_steps);
b2 = linspace(red_color(3), black_color(3), num_steps);

% 회색에서 파란색, 그리고 검은색으로 변화하는 커스텀 컬러맵 생성
red_colormap = [r', g', b'; r2', g2', b2'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% normalize MS
Max1=max(MS_fb,[],'all');
MI=[Max1];
Norm_MS=max(MI)*1;

RCP_Rho_Free=MS_fb./Norm_MS; % 1, 1
RCP_Phi_Free=PS_f; % 1, 2

% S mag
figure;
PC = pcolor(RCP_Rho_Free); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 Y]); % Default
set(gca, 'XLim', [1 X]);
pbaspect([X Y 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off
% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'blue.png';
% PNG 파일로 저장
saveas(gcf, filename);

new_height=500
new_width=500

% normalize MS
Max1 = max(MS_fb, [], 'all');
MI = [Max1];
Norm_MS = max(MI) * 1;

RCP_Rho_Free = MS_fb / Norm_MS; % 1, 1
RCP_Phi_Free = PS_f; % 1, 2

% 보간을 위한 새로운 좌표 생성
[X_orig, Y_orig] = meshgrid(1:size(RCP_Rho_Free, 2), 1:size(RCP_Rho_Free, 1));
[X_interp, Y_interp] = meshgrid(linspace(1, size(RCP_Rho_Free, 2), new_width), linspace(1, size(RCP_Rho_Free, 1), new_height));

% 데이터 보간
RCP_Rho_Free_interp = interp2(X_orig, Y_orig, RCP_Rho_Free, X_interp, Y_interp, 'linear');

% S mag
figure;
PC = pcolor(RCP_Rho_Free_interp); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 new_height]); % 수정된 높이 값으로 변경
set(gca, 'XLim', [1 new_width]); % 수정된 너비 값으로 변경
pbaspect([new_width new_height 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off

% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'blue_interp.png';

% PNG 파일로 저장
saveas(gcf, filename);


% normalize MS
Max1=max(MS_fg,[],'all');
MI=[Max1];
Norm_MS=max(MI)*1;

RCP_Rho_Free=MS_fg./Norm_MS; % 1, 1
RCP_Phi_Free=PS_f; % 1, 2

% S mag
figure;
PC = pcolor(RCP_Rho_Free); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 Y]); % Default
set(gca, 'XLim', [1 X]);
pbaspect([X Y 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off
% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'green.png';

% PNG 파일로 저장
saveas(gcf, filename);

new_height=500
new_width=500

% normalize MS
Max1 = max(MS_fg, [], 'all');
MI = [Max1];
Norm_MS = max(MI) * 1;

RCP_Rho_Free = MS_fg / Norm_MS; % 1, 1
RCP_Phi_Free = PS_f; % 1, 2

% 보간을 위한 새로운 좌표 생성
[X_orig, Y_orig] = meshgrid(1:size(RCP_Rho_Free, 2), 1:size(RCP_Rho_Free, 1));
[X_interp, Y_interp] = meshgrid(linspace(1, size(RCP_Rho_Free, 2), new_width), linspace(1, size(RCP_Rho_Free, 1), new_height));

% 데이터 보간
RCP_Rho_Free_interp = interp2(X_orig, Y_orig, RCP_Rho_Free, X_interp, Y_interp, 'linear');

% S mag
figure;
PC = pcolor(RCP_Rho_Free_interp); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 new_height]); % 수정된 높이 값으로 변경
set(gca, 'XLim', [1 new_width]); % 수정된 너비 값으로 변경
pbaspect([new_width new_height 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off

% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'green_interp.png';

% PNG 파일로 저장
saveas(gcf, filename);



% normalize MS
Max1=max(MS_fr,[],'all');
MI=[Max1];
Norm_MS=max(MI)*1;

RCP_Rho_Free=MS_fr./Norm_MS; % 1, 1
RCP_Phi_Free=PS_f; % 1, 2

% S mag
figure;
PC = pcolor(RCP_Rho_Free); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 Y]); % Default
set(gca, 'XLim', [1 X]);
pbaspect([X Y 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off
% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'red.png';

% PNG 파일로 저장
saveas(gcf, filename);



new_height=500
new_width=500

% normalize MS
Max1 = max(MS_fr, [], 'all');
MI = [Max1];
Norm_MS = max(MI) * 1;

RCP_Rho_Free = MS_fr / Norm_MS; % 1, 1
RCP_Phi_Free = PS_f; % 1, 2

% 보간을 위한 새로운 좌표 생성
[X_orig, Y_orig] = meshgrid(1:size(RCP_Rho_Free, 2), 1:size(RCP_Rho_Free, 1));
[X_interp, Y_interp] = meshgrid(linspace(1, size(RCP_Rho_Free, 2), new_width), linspace(1, size(RCP_Rho_Free, 1), new_height));

% 데이터 보간
RCP_Rho_Free_interp = interp2(X_orig, Y_orig, RCP_Rho_Free, X_interp, Y_interp, 'linear');

% S mag
figure;
PC = pcolor(RCP_Rho_Free_interp); % Intensity
PC.EdgeColor = 'none';
hold on
c = colorbar();
colormap(gray);
set(gca, 'fontname', 'arial', 'fontsize', 10);
set(gca, 'linewidth', 1);
set(gca, 'YLim', [1 new_height]); % 수정된 높이 값으로 변경
set(gca, 'XLim', [1 new_width]); % 수정된 너비 값으로 변경
pbaspect([new_width new_height 1]); % Figure aspect ratio
view(2); % = view(0,90) < Azimuth, Elevation
mnum = 1;
title("|E|");
axis off
grid off

% 파일명 지정 (원하는 경로와 파일명으로 수정)
filename = 'red_interp.png';

% PNG 파일로 저장
saveas(gcf, filename);
