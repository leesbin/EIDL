
close all;
color=load('redblue.mat').c;

for seq=0:0
    cd ..
    if seq==0
        %close all
        cd A/
        cd Result/
        cd Etched_result/
        exist Result2 dir
        if ans==7
            fprintf('[Result2] is exist \n')
        else
            mkdir Result2
        end
    end
    
    if seq==1
        %close all
        cd 1_Penalization_result/
    end
    
    
    clearvars -except seq color
    clc;
    
    
    
    geo_check=1;
    Load_input=1;
    View_output_surf=0;
    Result_check=1;
    View_input_pcolor=1;
    View_output_pcolor=1;
    
    rs=50;
    d_rs=50;
    
    d_width=3;
    d_length=3;
    d_height=0.5;
    
    %% ////////////////////////////////////////////////// Input  (Forward, Reverse) /////////////////////////////////////////////////////

    if Load_input==1
        %% Load Forward Input DFT
        for Forward_input_load=0:0       
            Ex_o= ["Ex_fund_field.h5"];                           %Hx
            hinfo_Ex_o= hdf5info("Ex_fund_field.h5");
            Ey_o= ["Ey_fund_field.h5"];                           %Hy
            hinfo_Ey_o= hdf5info("Ey_fund_field.h5");
            Ez_o= ["Ez_fund_field.h5"];                            %Ez
            hinfo_Ez_o= hdf5info("Ez_fund_field.h5");
            
            Name_Exro=["/ex_"+0+".r"];
            Name_Exio=["/ex_"+0+".i"];    
            Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
            Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
            Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
            % Hy
            Name_Eyro=["/ey_"+0+".r"];
            Name_Eyio=["/ey_"+0+".i"];    
            Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
            Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
            Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
            % Ez
            Name_Ezro=["/ez_"+0+".r"];
            Name_Ezio=["/ez_"+0+".i"];    
            Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
            Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
            Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex                
            X0fig=100; Y0fig=100;
            Wfig=1500;Hfig=900;    

            %% Absolute |E|
            Abs_Ex= abs(Ex_datao);
            Abs_Ey= abs(Ey_datao);
            Abs_Ez= abs(Ez_datao);
            Abs_E= sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));

            My_Max=max(Abs_E,[],'all');
            
            MaxE=1;          
            Maxa=max(Ex_data_ro,[],'all');
            Maxb=max(Ey_data_ro,[],'all');
            Maxc=max(Ez_data_ro,[],'all');            
            MMax=[Maxa Maxb Maxc];
            Ma=max(MMax); 
            Fsize=size(Ex_datao);
            X=Fsize(2);
            Y=Fsize(1);
            
            % for aspeact ratio
            G=gcd(X,Y);
            Xr=X/G;
            Yr=Y/G;
        end
        Fow_in=Abs_E./My_Max;
        %% Input field (pcolor)
        for Input_pcolor_ver=0:0
                if View_input_pcolor==1    
                        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                        subplot(2,3,1)
                        PC=pcolor(Ex_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,1),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_x] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,2)
                        PC=pcolor(Ey_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,2),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_y] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,3)
                        PC=pcolor(Ez_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,3),color);
                        caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_z] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,4)
                        PC=pcolor(Abs_Ex./My_Max);%intencity
                        PC.EdgeColor='none';
                       
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,4),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        %view(2);% = view(0,90) < 방위각, 고도각
                        title("|Ex| (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,5)
                        PC=pcolor(Abs_Ey./My_Max);%intencity
                        PC.EdgeColor='none';
                     
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,5),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|Ey| (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,6)
                        PC=pcolor(Abs_Ez./My_Max);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,6),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|Ez| (input)")
                        sgtitle("Input (Forward)")
                        axis off
                        grid off 
                        cd Result2/
                        print("0_Forward_Input",'-dpng')
                        cd ..
                end            
        end

        %% Load Backward Input DFT
        for Backward_input_load=0:0       
            Ex_o= ["Ex_target_field.h5"];                           %Hx
            hinfo_Ex_o= hdf5info("Ex_target_field.h5");
            Ey_o= ["Ey_target_field.h5"];                           %Hy
            hinfo_Ey_o= hdf5info("Ey_target_field.h5");
            Ez_o= ["Ez_target_field.h5"];                            %Ez
            hinfo_Ez_o= hdf5info("Ez_target_field.h5");
            
            Name_Exro=["/ex_"+0+".r"];
            Name_Exio=["/ex_"+0+".i"];    
            Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
            Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
            Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
            % Hy
            Name_Eyro=["/ey_"+0+".r"];
            Name_Eyio=["/ey_"+0+".i"];    
            Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
            Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
            Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
            % Ez
            Name_Ezro=["/ez_"+0+".r"];
            Name_Ezio=["/ez_"+0+".i"];    
            Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
            Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
            Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex                
            X0fig=100; Y0fig=100;
            Wfig=1500;Hfig=900;    

            %% Absolute |E|
            Abs_Ex= abs(Ex_datao);
            Abs_Ey= abs(Ey_datao);
            Abs_Ez= abs(Ez_datao);
            Abs_E= sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));

            My_Max=max(Abs_E,[],'all');
            
            MaxE=1;          
            Maxa=max(Ex_data_ro,[],'all');
            Maxb=max(Ey_data_ro,[],'all');
            Maxc=max(Ez_data_ro,[],'all');            
            MMax=[Maxa Maxb Maxc];
            Ma=max(MMax); 
            Fsize=size(Ex_datao);
            X=Fsize(2);
            Y=Fsize(1);
            
            % for aspeact ratio
            G=gcd(X,Y);
            Xr=X/G;
            Yr=Y/G;
        end
        Rev_in=Abs_E./My_Max;
        %% Input field (pcolor)
        for Input_pcolor_ver=0:0
                if View_input_pcolor==1    
                        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                        subplot(2,3,1)
                        PC=pcolor(Ex_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,1),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_x] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,2)
                        PC=pcolor(Ey_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,2),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_y] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,3)
                        PC=pcolor(Ez_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,3),color);
                        caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_z] (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,4)
                        PC=pcolor(Abs_Ex./My_Max);%intencity
                        PC.EdgeColor='none';
                       
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,4),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        %view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_x| (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,5)
                        PC=pcolor(Abs_Ey./My_Max);%intencity
                        PC.EdgeColor='none';
                     
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,5),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_y| (input)")
                        axis off
                        grid off
                        
                        subplot(2,3,6)
                        PC=pcolor(Abs_Ez./My_Max);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,6),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_z| (input)")
                        sgtitle("Input (Backward)")
                        axis off
                        grid off 
                        cd Result2/
                        print("2_Backward_Input",'-dpng')
                        cd ..
                end            
        end
    end
    %% //////////////////////////////////////////////// Output /////////////////////////////////////////////////////
    
    %% Load Output DFT
    for output_load=0:0
        %free space
        Ex_o= ["Ex_opt_field.h5"];                           %Hx
        hinfo_Ex_o= hdf5info("Ex_opt_field.h5");
        Ey_o= ["Ey_opt_field.h5"];                           %Hy
        hinfo_Ey_o= hdf5info("Ey_opt_field.h5");
        Ez_o= ["Ez_opt_field.h5"];                            %Ez
        hinfo_Ez_o= hdf5info("Ez_opt_field.h5");
        
        
        Name_Exro=["/ex_"+0+".r"];
        Name_Exio=["/ex_"+0+".i"];    
        Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
        Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
        Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
        % Hy
        Name_Eyro=["/ey_"+0+".r"];
        Name_Eyio=["/ey_"+0+".i"];    
        Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
        Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
        Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
        % Ez
        Name_Ezro=["/ez_"+0+".r"];
        Name_Ezio=["/ez_"+0+".i"];    
        Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
        Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
        Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex
        
        
        X0fig=100; Y0fig=100;
        Wfig=1500;Hfig=900;
        
        %% Absolute |E|
        Abs_Ex= abs(Ex_datao);
        Abs_Ey= abs(Ey_datao);
        Abs_Ez= abs(Ez_datao);
        Abs_E= sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));

        My_Max=max(Abs_E,[],'all');
        
        MaxE=1;          
        Maxa=max(Ex_data_ro,[],'all');
        Maxb=max(Ey_data_ro,[],'all');
        Maxc=max(Ez_data_ro,[],'all');            
        MMax=[Maxa Maxb Maxc];
        Ma=max(MMax); 
        Fsize=size(Ex_datao);
        X=Fsize(2);
        Y=Fsize(1);
        
        % for aspeact ratio
        G=gcd(X,Y);
        Xr=X/G;
        Yr=Y/G;
    end
    Fow_out=Abs_E./My_Max;   
    %% Output field (pcolor)
    for output_pcolor_ver=0:0
        if View_output_pcolor==1
                        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                        subplot(2,3,1)
                        PC=pcolor(Ex_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,1),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_x] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,2)
                        PC=pcolor(Ey_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,2),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_y] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,3)
                        PC=pcolor(Ez_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,3),color);
                        caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_z] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,4)
                        PC=pcolor(Abs_Ex./My_Max);%intencity
                        PC.EdgeColor='none';
                       
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,4),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        %view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_x| (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,5)
                        PC=pcolor(Abs_Ey./My_Max);%intencity
                        PC.EdgeColor='none';
                     
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,5),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_y| (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,6)
                        PC=pcolor(Abs_Ez./My_Max);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,6),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_z| (output)")
                        sgtitle("output (Forward)")
                        axis off
                        grid off 
                        cd Result2/
                        print("1_Forward_Output",'-dpng')
                        cd ..
        end    
    end
    
    %% Load Reverse DFT
    for output_load=0:0
        %free space
        Ex_o= ["Ex_rev_field.h5"];                           %Hx
        hinfo_Ex_o= hdf5info("Ex_rev_field.h5");
        Ey_o= ["Ey_rev_field.h5"];                           %Hy
        hinfo_Ey_o= hdf5info("Ey_rev_field.h5");
        Ez_o= ["Ez_rev_field.h5"];                            %Ez
        hinfo_Ez_o= hdf5info("Ez_rev_field.h5");
        
        
        Name_Exro=["/ex_"+0+".r"];
        Name_Exio=["/ex_"+0+".i"];    
        Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
        Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
        Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
        % Hy
        Name_Eyro=["/ey_"+0+".r"];
        Name_Eyio=["/ey_"+0+".i"];    
        Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
        Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
        Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
        % Ez
        Name_Ezro=["/ez_"+0+".r"];
        Name_Ezio=["/ez_"+0+".i"];    
        Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
        Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
        Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex
        
        
        X0fig=100; Y0fig=100;
        Wfig=1500;Hfig=900;
        
        %% Absolute |E|
        Abs_Ex= abs(Ex_datao);
        Abs_Ey= abs(Ey_datao);
        Abs_Ez= abs(Ez_datao);
        Abs_E= sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));

        My_Max=max(Abs_E,[],'all');
        
        MaxE=1;          
        Maxa=max(Ex_data_ro,[],'all');
        Maxb=max(Ey_data_ro,[],'all');
        Maxc=max(Ez_data_ro,[],'all');            
        MMax=[Maxa Maxb Maxc];
        Ma=max(MMax); 
        Fsize=size(Ex_datao);
        X=Fsize(2);
        Y=Fsize(1);
        
        % for aspeact ratio
        G=gcd(X,Y);
        Xr=X/G;
        Yr=Y/G;
    end
    Rev_out=Abs_E./My_Max;
    %% Output field (pcolor)
    for output_pcolor_ver=0:0
        if View_output_pcolor==1
                        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                        subplot(2,3,1)
                        PC=pcolor(Ex_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,1),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_x] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,2)
                        PC=pcolor(Ey_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,2),color);
                             caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_y] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,3)
                        PC=pcolor(Ez_data_ro./Ma);%intencity
                        PC.EdgeColor='none';
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,3),color);
                        caxis([-MaxE MaxE])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        title("Re[E_z] (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,4)
                        PC=pcolor(Abs_Ex./My_Max);%intencity
                        PC.EdgeColor='none';
                       
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,4),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        %view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_x| (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,5)
                        PC=pcolor(Abs_Ey./My_Max);%intencity
                        PC.EdgeColor='none';
                     
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,5),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_y| (output)")
                        axis off
                        grid off
                        
                        subplot(2,3,6)
                        PC=pcolor(Abs_Ez./My_Max);%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,3,6),"turbo");
                             caxis([0 1])
                        
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                         set(gca,'YLim',[1 Y])                         %기본
                         set(gca,'XLim',[1 X])
                        pbaspect([Xr Yr 1]) %figure 종횡비
                        % view(2);% = view(0,90) < 방위각, 고도각
                        title("|E_z| (output)")
                        sgtitle("output (Backward)")
                        axis off
                        grid off 
                        cd Result2/
                        print("3_Backward_Output",'-dpng')
                        cd ..
        end    

    end




    %% Load result
%     Fname='last_purity.txt';
%     Open=fopen(Fname);
%     
%     Purity_bulk=textscan(Open,'%f', 'TreatAsEmpty',{'Autograd', 'ArrayBox', 'with', 'value'});
%     fclose(Open);
%     
%     Purity=cell2mat(Purity_bulk);
%     Xp=round(Purity(1),2)*100;
%     Yp=round(Purity(2),2)*100;
%     Zp=round(Purity(7),2)*100;
    
    %% Optimization result (forward)
    for result=0:0
        if Result_check==1
            T_eff=load('WGeff.txt');
            T=round(T_eff(3), 2);
            Eff=round(T_eff(1), 2);
            Yp=round(T_eff(2), 2);
            cd ..
            cd ..
            FoM=load('evaluation.txt');
            Iters=size(FoM);
            cd Result/
            cd Etched_result/
            figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
            subplot(2,1,1)
                plot(FoM,'-r',LineWidth=1.5)
                set(gca,'XLim',[1 Iters(1)])
        
            subplot(2,2,3)
                        PC=pcolor(rot90(Rev_in,3));%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,2,3),"turbo");
                             caxis([0 1])
                        
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                 set(gca,'YLim',[1 X])                         %기본
                 set(gca,'XLim',[1 Y])
                pbaspect([Y X 1]) %figure 종횡비
                %view(2);% = view(0,90) < 방위각, 고도각
                title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
                axis off
                grid off 

            subplot(2,2,4)
                        PC=pcolor(rot90(Fow_out,3));%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,2,4),"turbo");
                             caxis([0 1])
                        
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                 set(gca,'YLim',[1 X])                         %기본
                 set(gca,'XLim',[1 Y])
                pbaspect([Y X 1]) %figure 종횡비
                %view(2);% = view(0,90) < 방위각, 고도각
                title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
                sgtitle("Transmission = "+T+"%, Eff = "+Eff+"%, Purity(y) = "+Yp+" [%].")
                        axis off
                        grid off 
                        cd Result2/
                        print("4_Opt_result",'-dpng')
                        cd ..
        end
    end
    

    if geo_check==1
        %% ////////////////////////////////////////////// Geometry ///////////////////////////////////////////////////
        Nx=d_rs*d_height+1;
        Ny=d_rs*d_width+1;
        Nz=d_rs*d_length+1;
    
        Geometry=load('Etched_design.txt');
        Gs=size(Geometry);
        My_width= linspace(-d_width/2, d_width/2, Ny);
        My_length= linspace(0, d_length, Nz);
        My_height= linspace(0, d_height, Nx);

        Geometry_new=(reshape(Geometry,Nz,Ny,Nx)); % 
        for i=0:0   
            %% fixed X (Z-Y plane) = topview 
              figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                    X_coordinate=1;
                    subplot(2,3,1)  
                        PC2=pcolor(My_length, My_width, rot90(Geometry_new(:,:,X_coordinate)));
                        PC2.EdgeColor='none';   
                        hold on   
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                     %    set(gca,'YLim',[1 Ny])                         %기본
                     %    set(gca,'XLim',[1 Nx])
                        pbaspect([Nz Ny 1]) %figure 종횡비
                        title('X=bottom')
                        view(2);% = view(0,90) < 방위각, 고도각
        
                    X_coordinate=round(Nx/2);
                    subplot(2,3,2)  
                        PC2=pcolor(My_length, My_width, rot90(Geometry_new(:,:,X_coordinate)));
                        PC2.EdgeColor='none';   
                        hold on   
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                     %    set(gca,'YLim',[1 Ny])                         %기본
                     %    set(gca,'XLim',[1 Nx])
                        pbaspect([Nz Ny 1]) %figure 종횡비
                        title('X=Mid')
                        view(2);% = view(0,90) < 방위각, 고도각
        
                    X_coordinate=Nx;
                    subplot(2,3,3)  
                        PC2=pcolor(My_length, My_width, rot90(Geometry_new(:,:,X_coordinate)));
                        PC2.EdgeColor='none';   
                        hold on   
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
                     %    set(gca,'YLim',[1 Ny])                         %기본
                     %    set(gca,'XLim',[1 Nx])
                        pbaspect([Nz Ny 1]) %figure 종횡비
                        title('X=Top')
                        view(2);% = view(0,90) < 방위각, 고도각
                        
            %% fixed Z (X-Y plane) = sideview
                    Z_coordinate=1;
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,7)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=Input')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각    
        
                    Z_coordinate=round(Nz/5);
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,8)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=Max/5')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각 
        
                    Z_coordinate=2*round(Nz/5);
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,9)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=2*Max/5')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각       
        
                    Z_coordinate=3*round(Nz/5);
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,10)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=3*Max/5')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각           
        
                    Z_coordinate=4*round(Nz/5);
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,11)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=4*Max/5')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각          
        
                    Z_coordinate=Nz;
                    KKK=squeeze(Geometry_new(Z_coordinate,:,:));
                    subplot(2,6,12)   
                        PC3=pcolor(My_height,My_width,KKK);
                        PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nx Ny 1]) 
                        title('Z=Output')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각
                        cd Result2/
                        print("6_Geometry",'-dpng')
                        cd ..
        end       
    end
end
%%    


    %% Optimization result (reverse)
    for result=0:0
        if Result_check==1
            T_eff=load('Reveff.txt');
            T=round(T_eff(3), 2);
            Eff=round(T_eff(1), 2);
            Yp=round(T_eff(2), 2);

            figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')

                    



            KKK=squeeze(Geometry_new(:,round(Ny/2),:));
            subplot(2,1,1)
                PC3=pcolor(My_length,My_height,KKK');
                PC3.EdgeColor='none';
                        % took 1./eps to make the high epsilon region to be black.    
                        hold on
                        colormap(subplot(2,1,1),flipud(gray));
                        c = colorbar();
                        caxis([0 1])
                        set(gca,'fontname','arial','fontsize',10)
                        set(gca,'linewidth',1);
%                         set(gca,'YLim',[1 Ny])                         %기본
%                         set(gca,'XLim',[1 Nx])
                        pbaspect([Nz Nx 1]) 
                        title('Y=center')
                        %pbaspect([Ny Nz 1]) %figure 종횡비
                        view(2);% = view(0,90) < 방위각, 고도각    
        
            subplot(2,2,3)
                        PC=pcolor(rot90(Fow_in,3));%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,2,3),"turbo");
                             caxis([0 1])
                        
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                 set(gca,'YLim',[1 X])                         %기본
                 set(gca,'XLim',[1 Y])
                pbaspect([Y X 1]) %figure 종횡비
                %view(2);% = view(0,90) < 방위각, 고도각
                title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
                        axis off
                        grid off 

            subplot(2,2,4)
                        PC=pcolor(rot90(Rev_out,3));%intencity
                        PC.EdgeColor='none';
                        
                        hold on
                        c = colorbar();
                        colormap(subplot(2,2,4),"turbo");
                             caxis([0 1])
                        
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                 set(gca,'YLim',[1 X])                         %기본
                 set(gca,'XLim',[1 Y])
                pbaspect([Y X 1]) %figure 종횡비
                %view(2);% = view(0,90) < 방위각, 고도각
                title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
                sgtitle("Transmission = "+T+"%, Eff = "+Eff+"%, Purity(y) = "+Yp+" [%].")
                        axis off
                        grid off 
                        cd Result2/
                        print("5_Reverse_result",'-dpng')
                        cd ..
        end
    end



