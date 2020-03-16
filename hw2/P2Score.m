clear; close all; clc
%% Handel
figure(1)
load handel;
plot((1:length(y))/Fs,y);
n = length(y);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');
%p8 = audioplayer(y,Fs); playblocking(p8);
[spec,t,ks] = myGabor(y',Fs,1000,750);
spec_zoom = spec(n/2+0.5:end,:);
k_zoom = ks(n/2+0.5:end);
ylim([0 0.35*10^4])
figure(10)
imagesc(t,flipud(k_zoom),spec_zoom);
title('Spectrogram for Messiah');
xlabel('Time [sec]');
ylabel('Frequency [hz]');
set(gca,'YDir','normal');
%% Music 1
clear all;
figure(3)
[y,Fs] = audioread('music1.wav');
n = length(y);
tr_piano=length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
%p8 = audioplayer(y,Fs); playblocking(p8);
%figure(4)
[spec,t,ks] = myGabor(y',Fs,100,100);
spec_zoom = spec(n/2+1:n/2+1+2*10^4,:);
k_zoom = ks(n/2+1:end);
ylim([0 0.35*10^4])
figure(10)
imagesc(t,flipud(k_zoom),spec_zoom);
title('Spectrogram for Mary had a Little Lamb on Piano');
xlabel('Time [sec]');
ylabel('Frequency [hz]');
%colormap(hot);
set(gca,'YDir','normal');
E = spec_zoom(5150,:);
D = spec_zoom(4590,:);
C = spec_zoom(4075,:);
tp = linspace(0,tr_piano,length(E));
E_notes = note_finder(E,tp,100);
D_notes = note_finder(D,tp,100);
C_notes = note_finder(C,tp,100);
E_notes_dur = [E_notes(1,:);E_notes(2,:)-E_notes(1,:)];
D_notes_dur = [D_notes(1,:);D_notes(2,:)-D_notes(1,:)];
C_notes_dur = [C_notes(1,:);C_notes(2,:)-C_notes(1,:)];

figure(11);
subplot(3,1,1);
plot(tp,E);
ylabel('E intensity');
subplot(3,1,2);
plot(tp,D);
ylabel('D intensity');
subplot(3,1,3);
plot(tp,C);
ylabel('C intensity');
xlabel('time [sec]')
sgtitle('Note intensity over time');
%% Music 2
clear all;
figure(5)
[y,Fs] = audioread('music2.wav');
n = length(y);
tr_rec=length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (recorder)');
%p8 = audioplayer(y,Fs); playblocking(p8);
figure(6)
[spec,t,ks] = myGabor(y',Fs,100,100);

spec_zoom = spec(n/2+1:n/2+1+6*10^4,:);
k_zoom = ks(n/2+1:end);
ylim([-0.5*10^4 0.5*10^4])
figure(12)
imagesc(t,flipud(k_zoom),spec_zoom);
title('Spectrogram for Mary had a Little Lamb on Recorder')
xlabel('Time [sec]');
ylabel('Frequency [hz]');
%colormap(hot);
set(gca,'YDir','normal');
%% Functions
function result = cutoff(k, k0)
A = zeros(1,length(k));
for i = 1:length(k)
    if i <= k0
        A(i) = 1;
    else
        %A(i) = max([0 1-0.1*(i-k0)]);
        A(i) = exp(-0.02*(i-k0).^2);
    end
    result = A;
end
end

function arr = note_finder(Sn, t,thresh)
last = 0;
notes = zeros(2,1);
for j = 2:length(Sn)
    if Sn(j) >= thresh
        if Sn(j-1) < thresh
            notes = [notes [t(j);0]];%mark the start of a new note
        end
    elseif Sn(j) < thresh && notes(end,end) == 0
        notes(end,end) = t(j);
    end
end
arr = notes(:,2:end);
end

function [spec,t,ks] = myGabor(S,Fs,a,num_T)
n=length(S); L = n/Fs;
t2=linspace(0,L,n+1); t=t2(1:n);
if mod(n,2) == 1
    k=(2*pi/L)*[0:n/2-0.5 -n/2+0.5:-1];
else
    k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
end
ks=fftshift(k);
tslide=linspace(0,L,num_T);
Sgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2);
    Sg=g.*S;
    Sgt=fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt)); % We don't want to scale it
end
spec = transpose(Sgt_spec);
end
