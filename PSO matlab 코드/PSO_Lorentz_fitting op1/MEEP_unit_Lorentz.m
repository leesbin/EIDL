function error_rms=MEEP_unit_Lorentz(gamma,FileName)
%% j 정의 해주기
j = complex(0,1);
%% 데이터 불러오기
 data = load(FileName);
 freq= data(:,1);  %% input wavelength
 eps1 = data(:,2);
 eps2 = data(:,3);
%% 복소유전율 구성
 eps_complex = eps1 + 1j .* eps2;
%% gamma 양수로 만들어주기 
for i = 1 : length(gamma)
    gamma(i) = abs(gamma(i));
end
%% 유전율 계산 (with gamma: gamma(1+pole): omega_n, gamma(2+pole): gamma_n, gamma(3+pole): sigma)
for m = 1 : length(freq)
    %eps(m) = gamma(4);
    eps(m) = 1.0;    % 유전율 무한?
    pole = 0;
    % 첫 극점 수정
    %eps(m) = eps(m) +  1.1^2 * 0.5/(1.1^2 - freq(m)^2 - 1*j*freq(m)*1e-5/2/pi);
    for k = 1 : length(gamma)/3
        eps(m) = eps(m) +  gamma(1 + pole*3)^2 * gamma(3 + pole*3)/(gamma(1 + pole*3)^2 - freq(m)^2 - 1*j*freq(m)*gamma(2 + pole*3));
        pole = pole + 1;
    end
 end
% 
% wp=gamma(1);
% gamma_n=gamma(2);
% 
% eps_drude=1-((wp).^2)./(omega.^2+1i.*omega.*(gamma_n));
%eps_drude=-(wp.^2)./(omega.^2+1i.*omega.*gamma_n);
%eps_drude=1-((wp*ehbar).^2)./(omega.^2+1i.*omega.*(gamma_n.*ehbar));  org
%eps_drude=-((8.9.*ehbar).^2).*0.845./(omega.^2+1i.*omega.*(0.048.*ehbar));
%% 오차 계산
error_rms=0;
error=zeros(1,length(eps));

 for i = 1 : length(eps)
   %% 10~12 계산 안하기 주파수는 파장 역수 >>> 100개 길이 : 0~100 = 13~8 um 범위
     if(i>20 && i<60)
         error_real =0;
         error_imag =0;
     
     
     else
      error_real = abs(real(eps(i)) - real(eps_complex(i)))/abs(real(eps_complex(i)));
%      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)));    
      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)))/abs(imag(eps_complex(i)));    
   % error(i) = (eps(i)-eps_complex(i))/eps_complex(i);
     end
   %% 실수부에 가중치 부여
%     if(error_real > error_imag)
%         error_rms=error_rms+error_real * 3.0;
%     end
%%
%     if(error_real <= error_imag)
%         error_rms=error_rms+error_imag;
%     end
%% 고주파의 극점을 피하기 위함
%     if( gamma(1) > 50.0)
%         error_rms = error_rms + 10000;
%     end
%
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

end