
%%
function error_rms=MEEP_unit_Lorentz(gamma,FileName)
%% j 정의 해주기
j = complex(0,1);
%% 데이터 불러오기
 data = load(FileName);
 freq= data(:,1);  %% input MEEP_frequency
 eps1 = data(:,2);
 eps2 = data(:,3);
 xi=0; % 유전율 에러 값 발산을 막기위한 가중치 상수
%% 복소유전율 구성
 eps_complex = eps1 + 1j .* eps2;
%% gamma 양수로 만들어주기 
for i = 1 : length(gamma)
    gamma(i) = abs(gamma(i));
end
%% 유전율 계산 (with gamma(): gamma(1+pole*3) = omega_n, gamma(2+pole*3): gamma_n, gamma(3+pole*3): delta_epsilon)
for m = 1 : length(freq)

    eps(m) = 1.0;    % 유전율_infty
    pole = 0;
 
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

      error_real = abs(real(eps(i)) - real(eps_complex(i)))/abs(real(eps_complex(i)+xi));
%      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)));    
      error_imag = abs(imag(eps(i)) - imag(eps_complex(i)))/abs(imag(eps_complex(i)+xi));    
   % error(i) = (eps(i)-eps_complex(i))/eps_complex(i);
    
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