
close all;
color=load('redblue.mat').c;

for seq=0:0
cd ..
if seq==0
        cd A/
        exist Result dir
        if ans==7
            fprintf('[Result] is exist \n')
        else
            mkdir Result
        end
end

if seq==1
    %close all
    cd 1_Penalization_result/
end

if seq==2
    %close all
    cd 2_Slant_penalization_last_result/
end

if seq==3
    %close all
    cd 3_Slant_penalization_result/
end


clearvars -except seq color
clc;



geo_check=1;
Load_input=0;
View_input_surf=0;
View_output_surf=0;
Result_check=0;
View_input_pcolor=0;
View_output_pcolor=0;

rs=50;
d_rs=50;
d_width= 3;
d_length= 3;
d_height= 0.5;

    X0fig=100; Y0fig=100;
    Wfig=1500;Hfig=900;

    %% ////////////////////////////////////////////// Geometry ///////////////////////////////////////////////////
    Nx=d_rs*d_height+1;
    Ny=d_rs*d_width+1;
    Nz=d_rs*d_length+1;

    Geometry=load('lastdesign.txt');
    cd Result/
    Gs=size(Geometry);
    My_width= linspace(-d_width/2, d_width/2, Ny);
    My_length= linspace(0, d_length, Nz);
    My_height= linspace(0, d_height, Nx);

    Geometry_new=(reshape(Geometry,Nz,Ny,Nx)); % 

    Top_layer= Geometry_new(:,:,13);
    Etched_geo=zeros(Nz,Ny,Nx);
    level= 14;

    deposition_level=[25, 22, 20, 17, 15, 12, 10, 7, 5, 2];
    deposed_layer=Top_layer;
        Test=zeros(Nz,Ny);

    while level < Nx+1 % 
%        mask=zeros(Nz,Ny);
        mask2=zeros(Nz,Ny);
        if ismember(level,deposition_level)
            for i=Ny:-1:2
                for j=Nz:-1:2
                    if deposed_layer(i,j) ~= deposed_layer(i,j-1) 
                        if deposed_layer(i,j)==1
                            mask2(i,j)=1;
                            Test(i,j)=1;

                        elseif deposed_layer(i,j)==0
                            mask2(i,j-1)=1;
                            Test(i,j)=1;
                        end

                    elseif deposed_layer(i,j) ~= deposed_layer(i-1,j)
                        if deposed_layer(i,j)==1
                            mask2(i,j)=1;
                            Test(i,j)=1;

                        elseif deposed_layer(i,j)==0
                            mask2(i-1,j)=1;
                            Test(i,j)=1;
                        end

                    end
                end
                
            end
            deposed_layer=deposed_layer-mask2;
            Etched_geo(:,:,level)=deposed_layer;%cat(2,deposed_layer(:,1:(Ny-1)/2), deposed_layer(:,(Ny-1)/2+1), fliplr(deposed_layer(:,1:(Ny-1)/2)));
        else
            Etched_geo(:,:,level)=deposed_layer;
        end
        exist Etched_grid dir
        if ans==7
            fprintf('[Etched_grid] is exist \n')
        else
            mkdir Etched_grid
        end
        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
            PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,level)));
            PC2.EdgeColor='none';   
            hold on   
            colormap(flipud(gray));
            c = colorbar();
            %caxis([0 1])
            set(gca,'fontname','arial','fontsize',10)
            set(gca,'linewidth',1);
         %    set(gca,'YLim',[1 Ny])                         %기본
         %    set(gca,'XLim',[1 Nx])
            pbaspect([Nz Ny 1]) %figure 종횡비
            title("Top view, level: "+level)
            view(2);% = view(0,90) < 방위각, 고도각
            cd Etched_grid/
            print("Etch"+level,'-dpng')
            cd ..
        exist Design_grid dir
        if ans==7
            fprintf('[Design_grid] is exist \n')
        else
            mkdir Design_grid
        end            
        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
            PC2=pcolor(My_length, My_width, rot90(Geometry_new(:,:,level)));
            PC2.EdgeColor='none';   
            hold on   
            colormap(flipud(gray));
            c = colorbar();
            %caxis([0 1])
            set(gca,'fontname','arial','fontsize',10)
            set(gca,'linewidth',1);
         %    set(gca,'YLim',[1 Ny])                         %기본
         %    set(gca,'XLim',[1 Nx])
            pbaspect([Nz Ny 1]) %figure 종횡비
            title("Top view, level: "+level)
            view(2);% = view(0,90) < 방위각, 고도각
            cd Design_grid/
            print("Geo"+level,'-dpng')
            cd ..
        level =level+1;
    end

deposed_layer=Top_layer;
level=13;
    while level > 0 % 
%        mask=zeros(Nz,Ny);
        mask2=zeros(Nz,Ny);
        if ismember(level,deposition_level)
            for i=Ny:-1:2
                for j=Nz:-1:2
                    if deposed_layer(i,j) ~= deposed_layer(i,j-1) 
                        if deposed_layer(i,j)==0
                            mask2(i,j)=1;
                            Test(i,j)=1;

                        elseif deposed_layer(i,j)==1
                            mask2(i,j-1)=1;
                            Test(i,j)=1;
                        end

                    elseif deposed_layer(i,j) ~= deposed_layer(i-1,j)
                        if deposed_layer(i,j)==0
                            mask2(i,j)=1;
                            Test(i,j)=1;

                        elseif deposed_layer(i,j)==1
                            mask2(i-1,j)=1;
                            Test(i,j)=1;
                        end

                    end
                end
                
            end
            deposed_layer=deposed_layer+mask2;
            Etched_geo(:,:,level)=deposed_layer;%cat(2,deposed_layer(:,1:(Ny-1)/2), deposed_layer(:,(Ny-1)/2+1), fliplr(deposed_layer(:,1:(Ny-1)/2)));
        else
            Etched_geo(:,:,level)=deposed_layer;
        end
        exist Etched_grid dir
        if ans==7
            fprintf('[Etched_grid] is exist \n')
        else
            mkdir Etched_grid
        end
        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
            PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,level)));
            PC2.EdgeColor='none';   
            hold on   
            colormap(flipud(gray));
            c = colorbar();
            %caxis([0 1])
            set(gca,'fontname','arial','fontsize',10)
            set(gca,'linewidth',1);
         %    set(gca,'YLim',[1 Ny])                         %기본
         %    set(gca,'XLim',[1 Nx])
            pbaspect([Nz Ny 1]) %figure 종횡비
            title("Top view, level: "+level)
            view(2);% = view(0,90) < 방위각, 고도각
            cd Etched_grid/
            print("Etch"+level,'-dpng')
            cd ..
        exist Design_grid dir
        if ans==7
            fprintf('[Design_grid] is exist \n')
        else
            mkdir Design_grid
        end            
        figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
            PC2=pcolor(My_length, My_width, rot90(Geometry_new(:,:,level)));
            PC2.EdgeColor='none';   
            hold on   
            colormap(flipud(gray));
            c = colorbar();
            %caxis([0 1])
            set(gca,'fontname','arial','fontsize',10)
            set(gca,'linewidth',1);
         %    set(gca,'YLim',[1 Ny])                         %기본
         %    set(gca,'XLim',[1 Nx])
            pbaspect([Nz Ny 1]) %figure 종횡비
            title("Top view, level: "+level)
            view(2);% = view(0,90) < 방위각, 고도각
            cd Design_grid/
            print("Geo"+level,'-dpng')
            cd ..
        level =level-1;
    end

    exist Etched_result dir
    if ans==7
        fprintf('[Etched_result] is exist \n')
    else
        mkdir Etched_result
    end

    cd Etched_result/
    Etched_weight=reshape(Etched_geo,Nz*Ny*Nx,1);
    fileID = fopen('Etched_design.txt','w');
    fprintf(fileID, '%e\n',Etched_weight);
    fclose(fileID);
    cd ..

%       figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
%             subplot(2,3,1) 
%                 PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,1)));
%                 PC2.EdgeColor='none';   
%                 hold on   
%                 colormap(flipud(gray));
%                 c = colorbar();
%                 caxis([0 1])
%                 set(gca,'fontname','arial','fontsize',10)
%                 set(gca,'linewidth',1);
%              %    set(gca,'YLim',[1 Ny])                         %기본
%              %    set(gca,'XLim',[1 Nx])
%                 pbaspect([Nz Ny 1]) %figure 종횡비
%                 title('X=bottom')
%                 view(2);% = view(0,90) < 방위각, 고도각
% 
%             subplot(2,3,2) 
%                 PC2=pcolor(My_length, My_width, rot90(Test));
%                 PC2.EdgeColor='none';   
%                 hold on   
%                 colormap(flipud(gray));
%                 c = colorbar();
%                 caxis([0 1])
%                 set(gca,'fontname','arial','fontsize',10)
%                 set(gca,'linewidth',1);
%              %    set(gca,'YLim',[1 Ny])                         %기본
%              %    set(gca,'XLim',[1 Nx])
%                 pbaspect([Nz Ny 1]) %figure 종횡비
%                 title('X=bottom')
%                 view(2);% = view(0,90) < 방위각, 고도각

    if geo_check==1
        for i=0:0   
            %% fixed X (Z-Y plane) = topview 
              figure('Name','Results','position',[X0fig Y0fig Wfig Hfig],'color','w')
                    X_coordinate=1;
                    subplot(2,3,1)  
                        PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,X_coordinate)));
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
                        PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,X_coordinate)));
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
                        PC2=pcolor(My_length, My_width, rot90(Etched_geo(:,:,X_coordinate)));
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
                    subplot(2,6,11)   
                        PC3=pcolor(My_height,My_width,KKK);
                        %PC3.EdgeColor='none';
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
                    KKK=squeeze(Etched_geo(Z_coordinate,:,:));
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
                        print("Geo",'-dpng')
        end       
    end
end
cd ..
