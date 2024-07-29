fprintf('----------------------Start  Oblique [geo] \n')
State='--------------------------Oblique [geo]---------------------- \n';
cd ..
cd Adjoint_small/

exist Oblique_geo_Data dir
if ans==7
    fprintf('[Oblique_geo_Data] is exist \n')
else
    mkdir Oblique_geo_Data
end
clearvars ans

% ///////////////////////////////////////////////////////////////////////////////////////////////////////
m_range_tot=5;
m_range=5;

theta=[-50, -40, -20, 0, 20, 40, 50];
initial_phi= 0; % unit: radian

Mon_size= load('Postprocess_monitor.txt');
Resolution=80;
Sz=round(Mon_size(2))+1;
Sr=round(Mon_size(1))+1;

for theta_sweep=1:7    
    Oblique_x_geo0= 0;
    Oblique_y_geo0= 0;
    Oblique_z_geo0= 0;

    Oblique_x_geo90= 0;
    Oblique_y_geo90= 0;
    Oblique_z_geo90= 0;

    Input_Ex= 0;
    Input_Ey= 0;
    Input_Ez= 0;
    Input_Hx= 0;
    Input_Hy= 0;
    Input_Hz= 0;

    R_Ex= 0;
    R_Ey= 0;
    R_Ez= 0;
    T_Ex= 0;
    T_Ey= 0;
    T_Ez= 0;
    F_Ex= 0;
    F_Ey= 0;
    F_Ez= 0;

    R_Hx= 0;
    R_Hy= 0;
    R_Hz= 0;
    T_Hx= 0;
    T_Hy= 0;
    T_Hz= 0;
    F_Hx= 0;
    F_Hy= 0;
    F_Hz= 0;

    for m_index=m_range_tot-m_range+1:m_range_tot+m_range+1 % temp=  0: RCP, 2: LCP

        Input_Ex_temp= zeros(Sr*2-1,Sr*2-1);
        Input_Ey_temp= zeros(Sr*2-1,Sr*2-1);
        Input_Ez_temp= zeros(Sr*2-1,Sr*2-1);
        Input_Hx_temp= zeros(Sr*2-1,Sr*2-1);
        Input_Hy_temp= zeros(Sr*2-1,Sr*2-1);
        Input_Hz_temp= zeros(Sr*2-1,Sr*2-1);

        R_Ex_temp= zeros(Sr*2-1,Sr*2-1);
        R_Ey_temp= zeros(Sr*2-1,Sr*2-1);
        R_Ez_temp= zeros(Sr*2-1,Sr*2-1);
        T_Ex_temp= zeros(Sr*2-1,Sr*2-1);
        T_Ey_temp= zeros(Sr*2-1,Sr*2-1);
        T_Ez_temp= zeros(Sr*2-1,Sr*2-1);
        F_Ex_temp= zeros(Sr*2-1,Sr*2-1);
        F_Ey_temp= zeros(Sr*2-1,Sr*2-1);
        F_Ez_temp= zeros(Sr*2-1,Sr*2-1);

        R_Hx_temp= zeros(Sr*2-1,Sr*2-1);
        R_Hy_temp= zeros(Sr*2-1,Sr*2-1);
        R_Hz_temp= zeros(Sr*2-1,Sr*2-1);
        T_Hx_temp= zeros(Sr*2-1,Sr*2-1);
        T_Hy_temp= zeros(Sr*2-1,Sr*2-1);
        T_Hz_temp= zeros(Sr*2-1,Sr*2-1);
        F_Hx_temp= zeros(Sr*2-1,Sr*2-1);
        F_Hy_temp= zeros(Sr*2-1,Sr*2-1);
        F_Hz_temp= zeros(Sr*2-1,Sr*2-1);

        for phi=0:+90:90
            initial_phi=pi*phi/180;
            m_number=m_index- m_range_tot- 1;
            mirror_coefficient= exp(pi*1.0i*m_number);
            initial_coefficient= exp(initial_phi*1.0i*m_number);
    
            C1=sin(initial_phi+pi);
            C2=cos(initial_phi+pi);
            I1=sin(initial_phi);
            I2=cos(initial_phi);
    
            %% Load Silicon space DFT
            for Si_load=0:0
                Er_s= ["Er_geo_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Er
                %hinfo_Er_s= hdf5info("Er_geo_a"+theta_sweep+"_m"+m_index+"_field.h5");
                Ep_s= ["Ep_geo_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Ep
                %hinfo_Ep_s= hdf5info("Ep_geo_a"+theta_sweep+"_m"+m_index+"_field.h5");
                Ez_s= ["Ez_geo_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Ez
                %hinfo_Ez_s= hdf5info("Ez_geo_a"+theta_sweep+"_m"+m_index+"_field.h5");
    
                % E_rho
                Name_Errs=["/er_"+0+".r"];
                Name_Eris=["/er_"+0+".i"];    
                Er_data_rs = h5read(Er_s,Name_Errs);          %Real
                Er_data_is = h5read(Er_s,Name_Eris);           %Imagine
                Er_datas = (Er_data_rs + Er_data_is * 1.0i)*initial_coefficient;       %Complex
                % E_phi
                Name_Eprs=["/ep_"+0+".r"];
                Name_Epis=["/ep_"+0+".i"];    
                Ep_data_rs = h5read(Ep_s,Name_Eprs);          %Real
                Ep_data_is = h5read(Ep_s,Name_Epis);           %Imagine
                Ep_datas = (Ep_data_rs + Ep_data_is * 1.0i)*initial_coefficient;     %Complex
                % E_z
                Name_Ezrs=["/ez_"+0+".r"];
                Name_Ezis=["/ez_"+0+".i"];    
                Ez_data_rs = h5read(Ez_s,Name_Ezrs);          %Real
                Ez_data_is = h5read(Ez_s,Name_Ezis);           %Imagine
                Ez_datas = (Ez_data_rs + Ez_data_is * 1.0i)*initial_coefficient;        %Complex
                
                clearvars Er_data_is Ep_data_is Ez_data_is Er_data_rs Ep_data_rs Ez_data_rs 
                % coordinate for matlab
                Fsize=size(Er_datas);
                R=Fsize(1);
                Z=Fsize(2);

                % E field at singular point (rho=0) depend on m
                Singular_rho=Er_datas(1,:);
                Singular_phi=Ep_datas(1,:);  
                Singular_z=Ez_datas(1,:);
    
                % Full range (rho= -R ~ +R) fields
                Ex_datas=cat(1, flipud(C2*Er_datas(2:R,:)-C1*Ep_datas(2:R,:))*mirror_coefficient, Singular_rho, I2*Er_datas(2:R,:)-I1*Ep_datas(2:R,:));
                Ey_datas=cat(1, flipud(C1*Er_datas(2:R,:)+C2*Ep_datas(2:R,:))*mirror_coefficient, Singular_phi, I1*Er_datas(2:R,:)+I2*Ep_datas(2:R,:));
                Ez_datas=cat(1, flipud(Ez_datas(2:R,:))*mirror_coefficient, Singular_z, Ez_datas(2:R,:));
                if phi==0
                    tempX0=Ex_datas';
                    tempY0=Ey_datas';
                    tempZ0=Ez_datas';
                elseif phi==90
                    tempX90=Ex_datas';
                    tempY90=Ey_datas';
                    tempZ90=Ez_datas';
                end   
    
                clearvars  Er_datas Ep_datas Ez_datas Singular_rho Singular_phi Singular_z Ex_datas Ey_datas 
                clearvars Er_s Ep_s Ez_s Name_Errs Name_Eris Name_Eprs Name_Epis Name_Ezrs Name_Ezis
            end
        end
        for phi=0:179
            initial_phi=pi*phi/180;
            m_number=m_index- m_range_tot- 1;
            mirror_coefficient= exp(pi*1.0i*m_number);
            initial_coefficient= exp(initial_phi*1.0i*m_number);
    
            C1=sin(initial_phi+pi);
            C2=cos(initial_phi+pi);
            I1=sin(initial_phi);
            I2=cos(initial_phi);
            %% Load Flux DFT (Line E field (input), H field (input, R, T, Sensor))
            for Flux_load=0:3
                if Flux_load==0
                    Er_flux= ["Er_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Er
                    Ep_flux= ["Ep_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Ep
                    Ez_flux= ["Ez_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Ez
                    
                    Hr_flux= ["Hr_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hr
                    Hp_flux= ["Hp_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hp
                    Hz_flux= ["Hz_input_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Hz
                elseif Flux_load==1
                    Er_flux= ["Er_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Er
                    Ep_flux= ["Ep_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Ep
                    Ez_flux= ["Ez_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Ez
                    
                    Hr_flux= ["Hr_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hr
                    Hp_flux= ["Hp_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hp
                    Hz_flux= ["Hz_R_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Hz
                elseif Flux_load==2
                    Er_flux= ["Er_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Er
                    Ep_flux= ["Ep_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Ep
                    Ez_flux= ["Ez_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Ez

                    Hr_flux= ["Hr_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hr
                    Hp_flux= ["Hp_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hp
                    Hz_flux= ["Hz_T_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Hz
                elseif Flux_load==3
                    Er_flux= ["Er_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Er
                    Ep_flux= ["Ep_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Ep
                    Ez_flux= ["Ez_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Ez

                    Hr_flux= ["Hr_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hr
                    Hp_flux= ["Hp_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                           %Hp
                    Hz_flux= ["Hz_F_a"+theta_sweep+"_m"+m_index+"_field.h5"];                            %Hz
                end                
                % E_rho
                Name_Err=["/er_"+0+".r"];
                Name_Eri=["/er_"+0+".i"];    
                Er_data_r = h5read(Er_flux,Name_Err);          %Real
                Er_data_i = h5read(Er_flux,Name_Eri);           %Imagine
                E_rho = (Er_data_r + Er_data_i * 1.0i)*initial_coefficient;       %Complex
                % E_phi
                Name_Epr=["/ep_"+0+".r"];
                Name_Epi=["/ep_"+0+".i"];    
                Ep_data_r = h5read(Ep_flux,Name_Epr);          %Real
                Ep_data_i = h5read(Ep_flux,Name_Epi);           %Imagine
                E_phi = (Ep_data_r + Ep_data_i * 1.0i)*initial_coefficient;     %Complex
                % E_z
                Name_Ezr=["/ez_"+0+".r"];
                Name_Ezi=["/ez_"+0+".i"];    
                Ez_data_r = h5read(Ez_flux,Name_Ezr);          %Real
                Ez_data_i = h5read(Ez_flux,Name_Ezi);           %Imagine
                E_z = (Ez_data_r + Ez_data_i * 1.0i)*initial_coefficient;        %Complex
                clearvars Er_data_r Ep_data_r Ez_data_r Er_data_i Ep_data_i Ez_data_i

                % E field at singular point (rho=0) depend on m
                Singular_rho=E_rho(1,:);
                Singular_phi=E_phi(1,:);  
                Singular_z=E_z(1,:);
    
                % Full range (rho= -R ~ +R) fields
                E_X=cat(1, flipud(C2*E_rho(2:R,:)-C1*E_phi(2:R,:))*mirror_coefficient, Singular_rho, I2*E_rho(2:R,:)-I1*E_phi(2:R,:));
                E_Y=cat(1, flipud(C1*E_rho(2:R,:)+C2*E_phi(2:R,:))*mirror_coefficient, Singular_phi, I1*E_rho(2:R,:)+I2*E_phi(2:R,:));
                E_z=cat(1, flipud(E_z(2:R,:))*mirror_coefficient, Singular_z, E_z(2:R,:));

                Singular_Ex= Singular_rho(1);
                Singular_Ey= Singular_phi(1);
                Singular_Ez= Singular_z(1);    

                clearvars E_rho E_phi Singular_rho Singular_phi Singular_z 
                clearvars Er_flux Ep_flux Ez_flux Name_Err Name_Eri Name_Epr Name_Epi Name_Ezr Name_Ezi

                % H_rho
                Name_Hrr=["/hr_"+0+".r"];
                Name_Hri=["/hr_"+0+".i"];    
                Hr_data_r = h5read(Hr_flux,Name_Hrr);          %Real
                Hr_data_i = h5read(Hr_flux,Name_Hri);           %Imagine
                H_rho = (Hr_data_r + Hr_data_i * 1.0i)*initial_coefficient;       %Complex
                % H_phi
                Name_Hpr=["/hp_"+0+".r"];
                Name_Hpi=["/hp_"+0+".i"];    
                Hp_data_r = h5read(Hp_flux,Name_Hpr);          %Real
                Hp_data_i = h5read(Hp_flux,Name_Hpi);           %Imagine
                H_phi = (Hp_data_r + Hp_data_i * 1.0i)*initial_coefficient;     %Complex
                % H_z
                Name_Hzr=["/hz_"+0+".r"];
                Name_Hzi=["/hz_"+0+".i"];    
                Hz_data_r = h5read(Hz_flux,Name_Hzr);          %Real
                Hz_data_i = h5read(Hz_flux,Name_Hzi);           %Imagine
                H_z = (Hz_data_r + Hz_data_i * 1.0i)*initial_coefficient;        %Complex
                clearvars Hr_data_r Hp_data_r Hz_data_r Hr_data_i Hp_data_i Hz_data_i
    
                % H field at singular point (rho=0) depend on m
                Singular_rho=H_rho(1,:);
                Singular_phi=H_phi(1,:);  
                Singular_z=H_z(1,:);
    
                % Full range (rho= -R ~ +R) fields
                H_X=cat(1, flipud(C2*H_rho(2:R,:)-C1*H_phi(2:R,:))*mirror_coefficient, Singular_rho, I2*H_rho(2:R,:)-I1*H_phi(2:R,:));
                H_Y=cat(1, flipud(C1*H_rho(2:R,:)+C2*H_phi(2:R,:))*mirror_coefficient, Singular_phi, I1*H_rho(2:R,:)+I2*H_phi(2:R,:));
                H_z=cat(1, flipud(H_z(2:R,:))*mirror_coefficient, Singular_z, H_z(2:R,:));

                Singular_Hx= Singular_rho(1);
                Singular_Hy= Singular_phi(1);
                Singular_Hz= Singular_z(1);   

                clearvars H_rho H_phi Singular_rho Singular_phi Singular_z 
                clearvars Hr_flux Hp_flux Hz_flux Name_Hrr Name_Hri Name_Hpr Name_Hpi Name_Hzr Name_Hzi

                for rho=1:Sr*2-1
                    X=round((rho-Sr)*cos(phi*pi/180))+ Sr;
                    Y=round((rho-Sr)*sin(phi*pi/180))+ Sr;

                    % stack the fields
                    if Flux_load==0
                        Input_Ex_temp(X,Y)= E_X(rho);
                        Input_Ey_temp(X,Y)= E_Y(rho);
                        Input_Ez_temp(X,Y)= E_z(rho);

                        Input_Hx_temp(X,Y)= H_X(rho);
                        Input_Hy_temp(X,Y)= H_Y(rho);
                        Input_Hz_temp(X,Y)= H_z(rho);

                        if phi==0 && rho==1
                            Input_Ex_temp(Sr,Sr)= Singular_Ex;
                            Input_Ey_temp(Sr,Sr)= Singular_Ey;
                            Input_Ez_temp(Sr,Sr)= Singular_Ez;

                            Input_Hx_temp(Sr,Sr)= Singular_Hx;
                            Input_Hy_temp(Sr,Sr)= Singular_Hy;
                            Input_Hz_temp(Sr,Sr)= Singular_Hz;      
                        end

                    elseif Flux_load==1
                        R_Ex_temp(X,Y)= E_X(rho);
                        R_Ey_temp(X,Y)= E_Y(rho);
                        R_Ez_temp(X,Y)= E_z(rho);

                        R_Hx_temp(X,Y)= H_X(rho);
                        R_Hy_temp(X,Y)= H_Y(rho);
                        R_Hz_temp(X,Y)= H_z(rho);
                        if phi==0 && rho==1
                            R_Ex_temp(Sr,Sr)= Singular_Ex;
                            R_Ey_temp(Sr,Sr)= Singular_Ey;
                            R_Ez_temp(Sr,Sr)= Singular_Ez;    

                            R_Hx_temp(Sr,Sr)= Singular_Hx;
                            R_Hy_temp(Sr,Sr)= Singular_Hy;
                            R_Hz_temp(Sr,Sr)= Singular_Hz;      
                        end

                    elseif Flux_load==2
                        T_Ex_temp(X,Y)= E_X(rho);
                        T_Ey_temp(X,Y)= E_Y(rho);
                        T_Ez_temp(X,Y)= E_z(rho);

                        T_Hx_temp(X,Y)= H_X(rho);
                        T_Hy_temp(X,Y)= H_Y(rho);
                        T_Hz_temp(X,Y)= H_z(rho);
                        if phi==0 && rho==1
                            T_Ex_temp(Sr,Sr)= Singular_Ex;
                            T_Ey_temp(Sr,Sr)= Singular_Ey;
                            T_Ez_temp(Sr,Sr)= Singular_Ez;  

                            T_Hx_temp(Sr,Sr)= Singular_Hx;
                            T_Hy_temp(Sr,Sr)= Singular_Hy;
                            T_Hz_temp(Sr,Sr)= Singular_Hz;      
                        end

                    elseif Flux_load==3
                        F_Ex_temp(X,Y)= E_X(rho);
                        F_Ey_temp(X,Y)= E_Y(rho);
                        F_Ez_temp(X,Y)= E_z(rho);

                        F_Hx_temp(X,Y)= H_X(rho);
                        F_Hy_temp(X,Y)= H_Y(rho);
                        F_Hz_temp(X,Y)= H_z(rho);
                        if phi==0 && rho==1
                            F_Ex_temp(Sr,Sr)= Singular_Ex;
                            F_Ey_temp(Sr,Sr)= Singular_Ey;
                            F_Ez_temp(Sr,Sr)= Singular_Ez; 

                            F_Hx_temp(Sr,Sr)= Singular_Hx;
                            F_Hy_temp(Sr,Sr)= Singular_Hy;
                            F_Hz_temp(Sr,Sr)= Singular_Hz;      
                        end                      
                    end
                end
                clearvars H_X H_Y H_z E_X E_Y E_z
                clearvars Singular_Ex Singular_Ey Singular_Ez Singular_Hx Singular_Hy Singular_Hz
            end
            prograss= (theta_sweep-1+(m_index/(m_range_tot+m_range+1))*(1/(Sr*2-1)^2)*((phi+1)/180))/((7));             
            clc
            disp(State)
            disp("Calculate the fields "+theta_sweep+"/7")
            disp((prograss*100)+"%")
        end

        Oblique_x_geo0= Oblique_x_geo0+ tempX0;
        Oblique_y_geo0= Oblique_y_geo0+ tempY0;
        Oblique_z_geo0= Oblique_z_geo0+ tempZ0;
        clearvars tempX0 tempY0 tempZ0

        Oblique_x_geo90= Oblique_x_geo90+ tempX90;
        Oblique_y_geo90= Oblique_y_geo90+ tempY90;
        Oblique_z_geo90= Oblique_z_geo90+ tempZ90;
        clearvars tempX90 tempY90 tempZ90

        Input_Ex= Input_Ex+ Input_Ex_temp;
        Input_Ey= Input_Ey+ Input_Ey_temp;
        Input_Ez= Input_Ez+ Input_Ez_temp;
        
        Input_Hx= Input_Hx+ Input_Hx_temp;
        Input_Hy= Input_Hy+ Input_Hy_temp;
        Input_Hz= Input_Hz+ Input_Hz_temp;
        clearvars Input_Ex_temp Input_Ey_temp Input_Ez_temp Input_Hx_temp Input_Hy_temp Input_Hz_temp   

        R_Ex= R_Ex+ R_Ex_temp;
        R_Ey= R_Ey+ R_Ey_temp;
        R_Ez= R_Ez+ R_Ez_temp;
        
        T_Ex= T_Ex+ T_Ex_temp;
        T_Ey= T_Ey+ T_Ey_temp;
        T_Ez= T_Ez+ T_Ez_temp;
        
        F_Ex= F_Ex+ F_Ex_temp;
        F_Ey= F_Ey+ F_Ey_temp;
        F_Ez= F_Ez+ F_Ez_temp;
        clearvars R_Ex_temp R_Ey_temp R_Ez_temp T_Ex_temp T_Ey_temp T_Ez_temp F_Ex_temp F_Ey_temp F_Ez_temp

        R_Hx= R_Hx+ R_Hx_temp;
        R_Hy= R_Hy+ R_Hy_temp;
        R_Hz= R_Hz+ R_Hz_temp;
        
        T_Hx= T_Hx+ T_Hx_temp;
        T_Hy= T_Hy+ T_Hy_temp;
        T_Hz= T_Hz+ T_Hz_temp;

        F_Hx= F_Hx+ F_Hx_temp;
        F_Hy= F_Hy+ F_Hy_temp;
        F_Hz= F_Hz+ F_Hz_temp;
        clearvars R_Hx_temp R_Hy_temp R_Hz_temp T_Hx_temp T_Hy_temp T_Hz_temp F_Hx_temp F_Hy_temp F_Hz_temp
    end

    Intensity_xz= Oblique_x_geo0.*conj(Oblique_x_geo0)+ Oblique_y_geo0.*conj(Oblique_y_geo0)+ Oblique_z_geo0.*conj(Oblique_z_geo0);
    clearvars Oblique_x_geo0 Oblique_y_geo0 Oblique_z_geo0
    Intensity_yz= Oblique_x_geo90.*conj(Oblique_x_geo90)+ Oblique_y_geo90.*conj(Oblique_y_geo90)+ Oblique_z_geo90.*conj(Oblique_z_geo90);
    clearvars Oblique_x_geo90 Oblique_y_geo90 Oblique_z_geo90
    Intensity_Sensor= F_Ex.*conj(F_Ex)+ F_Ey.*conj(F_Ey)+ F_Ez.*conj(F_Ez);

    cd Oblique_geo_Data/

    disp(State)
    disp((prograss*100)+"%")
    disp("Save the data [|E|^2] "+theta_sweep+"/7")

    save("Obliq_int_"+theta(theta_sweep)+"deg.mat", ...
                                          'Intensity_xz','Intensity_yz','Intensity_Sensor')
    clearvars Intensity_xz Intensity_yz Intensity_Sensor
    
    local_pr= 0;
    S_input= 0;
    S_refl= 0;
    S_tran= 0;
    S_focal= zeros(Sr*2-1, Sr*2-1);
    Focal_tot=0;
    S_input_tot= zeros(Sr*2-1, Sr*2-1);
    S_refl_tot= zeros(Sr*2-1, Sr*2-1);
    S_tran_tot= zeros(Sr*2-1, Sr*2-1);
    Ran=Sr*2-1;
    parfor (x_temp=1:Ran, Number_of_Core)
        for y_temp=1:Ran
            % Input flux
            E_temp=[Input_Ex(x_temp, y_temp), Input_Ey(x_temp, y_temp), Input_Ez(x_temp, y_temp)];
            H_temp=[conj(Input_Hx(x_temp, y_temp)), conj(Input_Hy(x_temp, y_temp)), conj(Input_Hz(x_temp, y_temp))];            
            P_temp=real(cross(E_temp, H_temp));
            S_input=S_input+ P_temp(3)*0.5;
            S_input_tot(x_temp, y_temp)=P_temp(3)*0.5;

            % Reflected flux
            E_temp=[R_Ex(x_temp, y_temp), R_Ey(x_temp, y_temp), R_Ez(x_temp, y_temp)];
            H_temp=[conj(R_Hx(x_temp, y_temp)), conj(R_Hy(x_temp, y_temp)), conj(R_Hz(x_temp, y_temp))];            
            P_temp=real(cross(E_temp, H_temp));
            S_refl=S_refl+ P_temp(3)*0.5;
            S_refl_tot(x_temp, y_temp)=P_temp(3)*0.5;

            % Transmitted flux
            E_temp=[T_Ex(x_temp, y_temp), T_Ey(x_temp, y_temp), T_Ez(x_temp, y_temp)];
            H_temp=[conj(T_Hx(x_temp, y_temp)), conj(T_Hy(x_temp, y_temp)), conj(T_Hz(x_temp, y_temp))];            
            P_temp=real(cross(E_temp, H_temp));
            S_tran=S_tran+ P_temp(3)*0.5;
            S_tran_tot(x_temp, y_temp)=P_temp(3)*0.5;

            % Flux in the sensor plane
            E_temp=[F_Ex(x_temp, y_temp), F_Ey(x_temp, y_temp), F_Ez(x_temp, y_temp)];
            H_temp=[conj(F_Hx(x_temp, y_temp)), conj(F_Hy(x_temp, y_temp)), conj(F_Hz(x_temp, y_temp))];            
            P_temp=real(cross(E_temp, H_temp));

            Focal_tot=Focal_tot+P_temp(3)*0.5;
            S_focal(x_temp, y_temp)=P_temp(3)*0.5;

            clc
            disp(State)
            disp("Calculate the fluxes "+theta_sweep+"/7")
            disp("total: "+(prograss*100)+"%")
        end
    end
    clearvars Input_Ex Input_Ey Input_Ez Input_Hx Input_Hy Input_Hz 
    clearvars R_Ex R_Ey R_Ez R_Hx R_Hy R_Hz 
    clearvars T_Ex T_Ey T_Ez T_Hx T_Hy T_Hz
    clearvars F_Ex F_Ey F_Ez F_Hx F_Hy F_Hz

    Oblique_Reflection= (S_input-S_refl)/S_input;
    Oblique_Transmission= (S_tran)/S_input;
    Sensor_flux= S_focal./S_input_tot;

    disp(State)
    disp((prograss*100)+"%")
    disp("Save the flux data [R, T ,Sensor] "+theta_sweep+"/7")

    save("raw_data"+theta(theta_sweep)+"deg.mat", ...
        'S_input', "S_tran", "S_refl","Focal_tot","S_input_tot","S_refl_tot","S_tran_tot","S_focal")

    save("Obliq_flux_"+theta(theta_sweep)+"deg.mat", ...
        'Sensor_flux', "Oblique_Transmission", "Oblique_Reflection")
    cd ..
    clearvars Sensor_flux Oblique_Transmission Oblique_Reflection
end

      





  

