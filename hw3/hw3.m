%% Load and play video
clear all;close all;clc;
load('cam1_1.mat');load('cam2_1.mat');load('cam3_1.mat')
%implay(vidFrames1_1)
%implay(vidFrames2_1)
%implay(vidFrames3_1)
F = vidFrames1_1;
Ft = imresize(diff(F,4),[480 640]);
Ftt = imresize(diff(Ft,4),[480 640]);


%% Sample frames for masks
numFrames = size(vidFrames1_1,4);
for j = 1:44:221
    B1 = vidFrames1_1(:,:,:,j);
    imwrite(B1+1, strcat('sample\vidframep1c1f',int2str(j),'.png'));
    B2 = vidFrames2_1(:,:,:,j);
    imwrite(B2+1, strcat('sample\vidframep1c2f',int2str(j),'.png'));
    B3 = vidFrames3_1(:,:,:,j);
    imwrite(B3+1, strcat('sample\vidframep1c3f',int2str(j),'.png'));
end

%% Process masked frames for desired colors
for j = 1:44:221
    B1m = imread(strcat('sample\vidframep1c1f',int2str(j),'m.png'));
    B1m = max(B1m,[],3);
    B1m = B1m == 0;
    [B1r,B1c] = ind2sub([480 640], find(B1m));
    B2m = imread(strcat('sample\vidframep1c2f',int2str(j),'m.png'));
    B2m = max(B2m,[],3);
    B2m = B2m == 0;
    [B2r,B2c] = ind2sub([480 640], find(B2m));
    B3m = imread(strcat('sample\vidframep1c3f',int2str(j),'m.png'));
    B3m = max(B3m,[],3);
    B3m = B3m == 0;
    [B3r,B3c] = ind2sub([480 640], find(B3m));
    
    Colors = zeros(length(B1r)+length(B2r)+length(B3r),3);
    for k = 1:length(B1r)
        newColor = squeeze(B1(B1r(k),B1c(k),:))';
        Colors(k,:) = newColor;
    end
    for k = 1:length(B2r)
        newColor = squeeze(B2(B2r(k),B2c(k),:))';
        Colors(k+length(B1r),:) = newColor;
    end
    for k = 1:length(B3r)
        newColor = squeeze(B3(B3r(k),B3c(k),:))';
        Colors(k+length(B1r)+length(B3r),:) = newColor;
    end
    %Colors = [Colors; B1(B1r,B1c,:); B2(B2r,B2c,:); B3(B3r,B3c,:)];
end
Colorsm = max(Colors,[],2) < 5;
outliers = find(Colorsm);
for j = outliers
    Colors(j,:) = [];
end
%% Compute color filter
figure(1)
scatter3(Colors(:,1),Colors(:,2),Colors(:,3),'.')
figure(2)
plot3(Colors(:,1),Colors(:,2),Colors(:,3),'.')
hold on
Surf = boundary(Colors,1);
tSurf = trisurf(Surf,Colors(:,1),Colors(:,2),Colors(:,3),'linestyle','none');
camlight()
hold off
figure(6)
bdrDwn = reducepatch(tSurf,0.15);
pts = bdrDwn.vertices;
X = pts(:,1);
Y = pts(:,2);
Z = pts(:,3);
plot3(X,Y,Z,'.');
M = [X.^2,Y.^2,Z.^2,X.*Y,X.*Z,Y.*Z,X,Y,Z,ones(size(X))];
[~,S,V] = svd(M,0);
[Xp,Yp,Zp] = meshgrid(1:10,1:10,1:10);
c = V(:,10);
M = c(1)*Xp.^2+c(2)*Yp.^2+c(3)*Zp.^2+c(4)*Xp.*Yp+c(5)*Xp.*Zp+c(6)*Yp.*Zp+c(7)*Xp+c(8)*Yp+c(9)*Zp+c(10)*ones(size(Xp));

%% Create color filter
F = zeros(256,256,256);
tic
inpolyhedron(tSurf.Faces,tSurf.Vertices, [250 250 250])
toc
%0.021 seconds!
%to run this on all possible colors would take 92 hours
%Instead, it may serve to take the point cloud and diffuse to make a filter
figure(3)
CSurf = convhull(Colors(:,1),Colors(:,2),Colors(:,3),'simplify',true);
tCSurf = trisurf(CSurf,Colors(:,1),Colors(:,2),Colors(:,3),'linestyle','none');
figure(4)
nfv = reducepatch(tCSurf,0.15);
V = nfv.vertices;
T = delaunay(V(:,1),V(:,2),V(:,3));
trisurf(T,V(:,1),V(:,2),V(:,3))
%% Test polynomial interpolation of surfaces
pt1 = [-1 -1];
pt2 = [0 -1];
pt3 = [1 1];
indep = [pt1;pt2];
dep = ones(1,2);
polyfitn(indep, dep,[0 0;1 0;0 1;2 0; 0 2])

%% Frame by frame display
numFrames = size(vidFrames1_1,4);
for j = 1:numFrames
    X = vidFrames1_1(:,:,:,j);
    imwrite(X+1, strcat('vidframe1_1_',int2str(j),'.png'));
    Y = vidFrames2_1(:,:,:,j);
    Z = vidFrames3_1(:,:,:,j);
    Xgr = mean(X,3);
    Ygr = mean(Y,3);
    Zgr = mean(Z,3);
    thresh = 240;
    Xf = Xgr > thresh;
    Yf = Ygr > thresh;
    Zf = Zgr > thresh;
    XYZgr = [Xgr Ygr Zgr];
    XYZf = [Xf Yf Zf];
    XYZ = [X Y Z];
    [Zp,~] = pinkMask(Z);
    [Zp1,~] = pink1Mask(Z);
    Zp = Zp|Zp1|Zf;
    XYZp = [Xf Yf Zp];
    I = [XYZgr;255*XYZf;255*XYZp];
    imshow(I, [0 255]); drawnow
end

%% Frame by frame motion
figure(11)
numFrames1 = size(vidFrames1_1,4);
numFrames2 = size(vidFrames2_1,4);
numFrames3 = size(vidFrames3_1,4);
avg1 = uint8(sum(vidFrames1_1,4)/numFrames1);
avg2 = uint8(sum(vidFrames2_1,4)/numFrames2);
avg3 = uint8(sum(vidFrames3_1,4)/numFrames3);
filter = zeros(480,640);
w = 2; h = 2;
filter(240-h:240+h,320-w:320+w) = ones(2*h+1,2*w+1);
filter = fftshift(filter);
pos1 = zeros(numFrames1,2);
pos2 = zeros(numFrames2,2);
pos3 = zeros(numFrames3,2);
for j = 1:numFrames
    f1 = vidFrames1_1(:,:,:,j);
    f2 = vidFrames2_1(:,:,:,j);
    f3 = vidFrames3_1(:,:,:,j);
    K1 = fft2((double(rgb2gray(f1-avg1))));
    K2 = fft2((double(rgb2gray(f2-avg2))));
    K3 = fft2((double(rgb2gray(f3-avg3))));
    ft1 = abs(ifft2(K1.*filter));
    ft2 = abs(ifft2(K2.*filter));
    ft3 = abs(ifft2(K3.*filter));
    [garbo,ind1] = max(ft1(:));
    [garbo,ind2] = max(ft2(:));
    [garbo,ind3] = max(ft3(:));
    [r1,c1] = ind2sub([480 640], ind1);
    [r2,c2] = ind2sub([480 640], ind2);
    [r3,c3] = ind2sub([480 640], ind3);
    pos1(j,:) = [r1 c1];
    pos2(j,:) = [r2 c2];
    pos3(j,:) = [r3 c3];
    %imagesc(ft);
    %pause(0.1);
end
figure(12)
plot(pos1(:,1));
figure(13)
plot(pos2(:,1));
figure(14)
plot(pos3(:,1));


%%
figure(11)
tic
K = fft2((double(rgb2gray(f-avg))));
filter = zeros(480,640);
w = 10; h = 10;
filter(240-h:240+h,320-w:320+w) = ones(2*h+1,2*w+1);
filter = fftshift(filter);
imagesc(abs(ifft2(K.*filter))>40);
toc
%ff = ifft(K);