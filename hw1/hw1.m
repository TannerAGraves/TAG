clear all; close all; clc;
load Testdata

L = 15;
n = 64;
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

for j = 1:1
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);
    close all, isosurface(X,Y,Z,abs(Un),0.4)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
    %Part 1: calculating the frequency signature of the marble
    tau = 0.1;
    K = fftn(Un);
    rs = zeros(64,1); cs = rs; ps = rs;
    %along each dimension of the 3D matrix, find the matrix with the
    %highest average frequency. This will correspond 
    for ind = 1:64
        rs(ind) = mean(abs(K(ind,:,:)),'all');
        cs(ind) = mean(abs(K(:,ind,:)),'all');
        ps(ind) = mean(abs(K(:,:,ind)),'all');
    end
    [val,r] = max(abs(rs));
    [val,c] = max(abs(cs));
    [val,p] = max(abs(ps));
    k0x = k(r);k0y = k(c);k0z = k(p);
    filter = exp(-tau*((k-k0x).^2 + (k-k0y).^2 + (k-k0z).^2));
    Kft = filter.*Un;
    Unft = ifft(Kft);
    %figure(2);
    isosurface(X,Y,Z,abs(Unft),5)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
end
%test
%[garbo, ind] = max(abs(Unft(:)));[r,c,p] = ind2sub(size(Unft),ind);abs(Unft(r,c,p))

