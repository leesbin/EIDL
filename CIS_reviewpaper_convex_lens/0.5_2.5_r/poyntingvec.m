
close all
color=load('redblue.mat').c;


clearvars -except seq color
clc;


rs=100;
Lpml=0.25;

d_values = [0.5, 1, 1.5, 2, 2.5];

for d = d_values
    X = rs * (d + 2 * Lpml);
    Y = rs * 10;

    % 빨간색       
    Hx_r = sprintf('%.1fHx_r_field.h5', d);  % d 값을 이용하여 문자열 포맷팅
    hinfo_Hx_r = hdf5info(Hx_r);  % 포맷팅된 파일명을 hdf5info 함수에 전달

    Hy_r = sprintf('%.1fHy_r_field.h5', d);  % d 값을 이용하여 문자열 포맷팅
    hinfo_Hy_r = hdf5info(Hy_r);  % 포맷팅된 파일명을 hdf5info 함수에 전달

    Ez_r = sprintf('%.1fEz_r_field.h5', d);  % d 값을 이용하여 문자열 포맷팅
    hinfo_Ez_r = hdf5info(Ez_r);  % 포맷팅된 파일명을 hdf5info 함수에 전달

                % Hy_r= ["d%.1fHy_r_field.h5",d/2];                           %Hy
                % hinfo_Hy_r= hdf5info("d%.1fHy_r_field.h5",d/2);
                % 
                % Ez_r= ["d%.1fEz_r_field.h5",d/2];                            %Hz
                % hinfo_Ez_r= hdf5info("d%.1fEz_r_field.h5",d/2);
    
    
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
            E_f=[0, 0, Ez_datar(y, x)];
            H_f=[conj(Hx_datar(y,x)), conj(Hy_datar(y,x)), 0];
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
    colormap(hot);
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
   filename = sprintf('red_d%.1f.png', d);
    
    % PNG 파일로 저장
    saveas(gcf, filename);
    fprintf('Simulation completed for d=%.1f\n', d);
end
    

