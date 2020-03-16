%% Setup clips
%  Seperate training data from test data
load('ABBAClips.mat');
load('BeethovenClips.mat');
load('BeatlesClips.mat');
AClips = full(ABBAClips);
BClips = full(BClips);
CClips = full(BeatlesClips);
trainAind = randsample(100,80);
trainBind = randsample(100,80);
trainCind = randsample(100,80);
trainA = AClips(trainAind);
trainB = BClips(trainBind);
trainC = CClips(trainCind);
testA = AClips;
testA(:,trainAind) = [];
testB = BClips;
testB(:,trainBind) = [];
testC = CClips;
testC(:,trainCind) = [];
TestSet = [testA testB testC];
%beethoven = 1, vivaldi = 2, strauss = 3;
hiddenlabels = [ones(1,20) 2*ones(1,20) 3*ones(1,20)];
rind = randperm(60);
TestSet = TestSet(:,rind);
hiddenlabels = hiddenlabels(rind);
%% Run PCA
feature = 20; % number of PCA modes
[Uab,Sab,Vab,tab,wab,sortAb,sortaB] = trainer(full(AClips),full(BClips),feature);
[Uac,Sac,Vac,tac,wac,sortAc,sortaC] = trainer(full(AClips),full(CClips),feature);
[Ubc,Sbc,Vbc,tbc,wbc,sortBc,sortbC] = trainer(full(BClips),full(CClips),feature);
%% Plot first four principal components
for k = 1:4
    subplot(2,2,k)
    ut1 = reshape(Uac(:,k),100,100);
    ut2 = rescale(ut1);
    imshow(ut2)
end
%% Plot singular values
figure(2)
subplot(2,1,1)
plot(diag(Sab),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 100])
subplot(2,1,2)
semilogy(diag(Sab),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 100])
title('Singular values of')
%% Plot projections of beethoven and vivaldi onto first 3 modes
figure(3)
for k=1:3
    subplot(3,2,2*k-1)
    plot(1:50,Vab(1:50,k),'ko-')
    subplot(3,2,2*k)
    plot(101:150,Vab(101:150,k),'ko-')
end
subplot(3,2,1), set(gca,'Ylim',[-.15 0],'Fontsize',12), title('beethoven') 
subplot(3,2,2), set(gca,'Ylim',[-.15 0],'Fontsize',12), title('vivaldi') 
subplot(3,2,3), set(gca,'Ylim',[-.2 .2],'Fontsize',12) 
subplot(3,2,4), set(gca,'Ylim',[-.2 .2],'Fontsize',12) 
subplot(3,2,5), set(gca,'Ylim',[-.2 .2],'Fontsize',12) 
subplot(3,2,6), set(gca,'Ylim',[-.2 .2],'Fontsize',12) 
%% Plot dog/cat projections onto w
figure(4)
subplot(3,1,1)
plot(sortAb,zeros(100),'ob','Linewidth',2)
hold on
plot(sortaB,ones(100),'dr','Linewidth',2)
ylim([0 2])
ylabel('abba-beethoven')
title('3 group projections onto w - band')
subplot(3,1,2)
plot(sortAc,zeros(100),'ob','Linewidth',2)
hold on
plot(sortaC,ones(100),'dr','Linewidth',2)
ylim([0 2])
ylabel('abba-beatles')
subplot(3,1,3)
plot(sortBc,zeros(100),'ob','Linewidth',2)
hold on
plot(sortbC,ones(100),'dr','Linewidth',2)
ylim([0 2])
ylabel('beethoven-beatles')
%% Test classifier
figure(1)
for k = 1:9
    subplot(3,3,k)
    test=reshape(TestSet(:,k+5),100,100);
    imshow(test)
end

TestNum=size(TestSet,2);
%Test_wave = dc_wavelet(TestSet); % wavelet transformation 
%TestMat = U'*TestSet;  % PCA projection
%pval = w'*TestMat;  % LDA projection

% Beethoven = 0, Vivaldi = 1
%ResVec = (pval>threshold)
vecs = classify3(TestSet,Uab,Uac,Ubc,tab,tac,tbc,wab,wac,wbc);
[~,result] = max(vecs,[],1);

disp('Number of mistakes')
errNum = sum(abs(result-hiddenlabels)>0)

disp('Rate of success');
sucRate = 1-errNum/TestNum
%%
function vec = classify3(TestSet,Uab,Uac,Ubc,tab,tac,tbc,wab,wac,wbc)
    TMab = Uab'*TestSet;
    TMac = Uac'*TestSet;
    TMbc = Ubc'*TestSet;
    PVab = wab'*TMab;
    PVac = wac'*TMac;
    PVbc = wbc'*TMbc;
    %larger values of PV-t favor second group
    % eg. if PV > t, then classify as grp 2
    a = -(PVab-tab) - (PVac-tac);
    b = PVab-tab - (PVbc-tbc);
    c = PVbc-tbc + PVac-tac;
    vec = [a;b;c];
end
function [U,S,V,threshold,w,sortA,sortB] = trainer(A0,B0,feature)
    nd = size(A0,2); nc = size(B0,2);
    
    [U,S,V] = svd([A0 B0],'econ');
    
    sounds = S*V'; % projection onto principal components
    U = U(:,1:feature);
    As = sounds(1:feature,1:nd);
    Bs = sounds(1:feature,nd+1:nd+nc);
    
    md = mean(As,2);
    mc = mean(Bs,2);
    
    Sw = 0; % within class variances
    for k=1:nd
        Sw = Sw + (As(:,k)-md)*(As(:,k)-md)';
    end
    for k=1:nc
        Sw = Sw + (Bs(:,k)-mc)*(Bs(:,k)-mc)';
    end
    
    Sb = (md-mc)*(md-mc)'; % between class 
    
    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);
    
    vA = w'*As; 
    vB = w'*Bs;
    
    if mean(vA)>mean(vB)
        w = -w;
        vA = -vA;
        vB = -vB;
    end
    % A < threshold < B
    
    sortA = sort(vA);
    sortB = sort(vB);
    
    t1 = length(sortA);
    t2 = 1;
    while sortA(t1)>sortB(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (sortA(t1)+sortB(t2))/2;
end