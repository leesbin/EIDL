cd ..
cd A/

    X0fig=100; Y0fig=100;
    Wfig=1500;Hfig=900;
    d_rs=50;
%% ////////////////////////////////////////////// Geometry ///////////////////////////////////////////////////
        Nx=d_rs*0.5+1;
        Ny=d_rs*3+1;
        Nz=d_rs*3+1;
    
        Geometry=load('Test3.txt');
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
                        cd Result/
                        print("6_Geometry",'-dpng')
                        cd ..
        end         