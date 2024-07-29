function MEEP_unit_Lorentz(gamma,FileName)
%% unit_Lorentz 에서 하던거(eps계산, 오차율계산) 해주기 단, 여기선 1회만 시행(with 최적의 gamma(최저의 오차율))

% j 정의 해주기
j = complex(0,1);
% 데이터 불러오기
 data = load(FileName);
 freq= data(:,1);  %% input wavelength
 eps1 = data(:,2);
 eps2 = data(:,3);
% 복소유전율 구성
 eps_complex = eps1 + 1j .* eps2;
% gamma 양수로 만들어주기 
for i = 1 : length(gamma)
    gamma(i) = abs(gamma(i));
end

% 유전율 계산 (with gamma: gamma(1+pole): omega_n, gamma(2+pole): gamma_n, gamma(3+pole): sigma)
for m = 1 : length(freq)
    %eps(m) = gamma(4);
    eps(m) = 1.0;
    pole = 0;
    %eps(m) = eps(m) +  1.1^2 * 0.5/(1.1^2 - freq(m)^2 - 1*j*freq(m)*1e-5/2/pi);
%
    for k = 1 : length(gamma)/3
        eps(m) = eps(m) +  gamma(1 + pole*3)^2 * gamma(3 + pole*3)/(gamma(1 + pole*3)^2 - freq(m)^2 - 1*j*freq(m)*gamma(2 + pole*3));
        pole = pole + 1;
    end
end

% 오차 계산
 error_rms=0;
error=zeros(1,length(eps));
 for i = 1 : length(eps)
     
      %% 10~12 계산 안하기
     if(i>20 && i<60)
         error_real =0;
         error_imag =0;
%      
%      
     else
      error_real = abs(real(eps(i)) - real(eps_complex(i)))/abs(real(eps_complex(i)));
%      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)));    
      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)))/abs(imag(eps_complex(i)));    
   % error(i) = (eps(i)-eps_complex(i))/eps_complex(i);
     end
     
     %% 원래것
%       error_real = abs(real(eps(i)) - real(eps_complex(i)))/abs(real(eps_complex(i)));
% %      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)));    
%       error_imag = abs(imag(eps(i)) - imag(eps_complex(i)))/abs(imag(eps_complex(i)));   
      
      
      
   % error(i) = (eps(i)-eps_complex(i))/eps_complex(i);
%     if(error_real > error_imag)
%         error_rms=error_rms+error_real * 3.0;
%     end
%     if(error_real <= error_imag)
%         error_rms=error_rms+error_imag;
%     end
%     if( gamma(1) > 50.0)
%         error_rms = error_rms + 10000;
%     end
%     if real(eps(i)) > 0
%         error_rms = error_rms + 10000;
%     end
%    if imag(eps(i)) < 0
%         error_rms = error_rms + 10000;
%     end
%     if( i  > 5 )
%         if (i < 40)
%             error_real = error_real * 100;
%         end
%     end
     error_rms=error_rms+error_real  + error_imag;
     %error_rms=error_rms+error_real + error_imag;
     
 end

error_rms

%%
% 
% wp=gamma(1);
% gamma_n=gamma(2);
% 
% eps_drude=1-((wp).^2)./(omega.^2+1i.*omega.*(gamma_n));
%eps_drude=-(wp.^2)./(omega.^2+1i.*omega.*gamma_n);
%eps_drude=1-((wp*ehbar).^2)./(omega.^2+1i.*omega.*(gamma_n.*ehbar));  org
%eps_drude=-((8.9.*ehbar).^2).*0.845./(omega.^2+1i.*omega.*(0.048.*ehbar));

%% gamma 값 메모장으로 출력
% % saving  coefficients
fp100=fopen('Drude_Coefficients_MEEP_UNIT.txt','w');
for i = 1 : length (gamma)
    fprintf(fp100,'gamma = %f\n', gamma(i));
end
fclose(fp100);

%% 그래프그리기 (MEEP 주파수)

%% eps1 실수부
figure
% plot(1./freq.*500e-9./1e-9,real(eps),'r-','linewidth',2);
% hold on
% plot(1./freq.*500e-9./1e-9,eps1,'ro','linewidth',2);

% 피팅결과 (실선)
plot(1./freq.*10e-6./1e-6,real(eps),'r-','linewidth',2);
% plot(freq,real(eps),'r-','linewidth',2);
hold on

% 소재 데이터 (동그라미)
plot(1./freq.*10e-6./1e-6,eps1,'ro','linewidth',2);
% plot(freq,eps1,'ro','linewidth',2);
% 라벨링 (폰트,굵기 등등)
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
xlabel ('Wavelength (um)','fontsize',16,'fontweight','bold')
ylabel ('eps1', 'Rotation',90,'fontsize',16,'fontweight','bold')
%set(gca,'XLim',[1 length(particle_track(:,1))])

%% 그래프 축 제한
% set(gca,'YLim',[-0.3 15])
%set(gca,'XLim',[300 850])
% legend('eps1, Palik','eps2, Drude','eps2, Palik','eps2, Drude','location','southeast');
% legend boxoff
%print -dpng -r300 -adobecset Ag_eps
str = ['eps1'];

% png 파일로 출력
print(str,'-dpng') 


%% eps2 허수부
figure
% plot(1./freq.*500e-9./1e-9,imag(eps),'b-','linewidth',2);
% hold on
% plot(1./freq.*500e-9./1e-9,eps2,'bo','linewidth',2);

% 피팅결과 (실선)
 plot(1./freq.*10e-6./1e-6,imag(eps),'b-','linewidth',2);
% plot(freq,imag(eps),'b-','linewidth',2);
hold on

% 소재 데이터 (동그라미)
 plot(1./freq.*10e-6./1e-6,eps2,'bo','linewidth',2);
% plot(freq,eps2,'bo','linewidth',2);
% 라벨링
set(gca,'fontname','arial','fontsize',16,'fontweight','bold')
set(gca,'linewidth',3);
xlabel ('Wavelength (um)','fontsize',16,'fontweight','bold')

%% 그래프 축 제한
% set(gca,'YLim',[-0.3 15])
ylabel ('eps2', 'Rotation',90,'fontsize',16,'fontweight','bold')
str = ['eps2'];

% png 파일로 출력
print(str,'-dpng') 


end