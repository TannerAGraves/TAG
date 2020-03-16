clear; close all; clc

load handel;

n=length(y); L = n/Fs;
t2=linspace(0,L,n+1); t=t2(1:n); 
k=(2*pi/L)*[0:n/2-0.5 -n/2+0.5:-1]; 
%k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
S = y';
St=fft(S);

a = 35000;
tslide=linspace(0,L,1000);
Sgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    Sg=g.*S; 
    Sgt=fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt)); % We don't want to scale it
end
figure(1)
imagesc(tslide,ks,Sgt_spec.');
colormap(hot);
