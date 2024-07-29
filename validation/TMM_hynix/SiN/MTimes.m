
% Nx x Ny x 4 matrices
function[C] = MTimes(A,B)
    C = zeros(size(A));
    A1 = A(:,:,1);
    A2 = A(:,:,2);
    A3 = A(:,:,3);
    A4 = A(:,:,4);
    B1 = B(:,:,1);
    B2 = B(:,:,2);
    B3 = B(:,:,3);
    B4 = B(:,:,4);
    
    C(:,:,1) = A1.*B1 + A2.*B3;
    C(:,:,2) = A1.*B2 + A2.*B4;
    C(:,:,3) = A3.*B1 + A4.*B3;
    C(:,:,4) = A3.*B2 + A4.*B4;
end