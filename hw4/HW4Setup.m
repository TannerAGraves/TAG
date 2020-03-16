%https://archive.org/download/lp-00780_BeG/lp-00780_BeG-record_1_side_A.wav
%https://archive.org/download/lp-00780_BeG/lp-00780_BeG-record_1_side_B.wav
%https://archive.org/download/lp-00780_BeG/lp-00780_BeG-record_1_side_A.wav
%https://ia800502.us.archive.org/17/items/lp-00780_BeG/lp-00780_BeG-record_1_side_B.wav


%Beethoven:
BUrls = [
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-2-1-1-pfaul.mp3";
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-2-1-1-hensley.mp3";
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-7-3-sinadinovic.mp3";
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-10-2-1-breemer.mp3";
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-14-1-1-bertram.mp3";
    "http://d19bhbirxx14bg.cloudfront.net/beethoven-14-2-1-bertram.mp3"];
%BFiles = webread(BUrls(1));
%Vivaldi:
VUrls = [
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.01.%20Concerto%20in%20C%20Major%3A%20Largo%20-%20Allegro%20molto.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.02.%20Concerto%20in%20C%20Major%3A%20Largo%20e%20cantabile.mp3";
    "https://ia903107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.03.%20Concerto%20in%20C%20Major%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.04.%20Concerto%20in%20E%20Major%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.05.%20Concerto%20in%20E%20Major%3A%20Andante%20.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/01.06.%20Concerto%20in%20E%20Major%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.01.%20Concerto%20in%20D%20Minor%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.02.%20Concerto%20in%20D%20Minor%3A%20Largo.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.03.%20Concerto%20in%20D%20Minor%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.04.%20Concerto%20in%20A%20Minor%3A%20Allegro.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.05.%20Concerto%20in%20A%20Minor%3A%20Larghetto.mp3";
    "https://ia803107.us.archive.org/2/items/lp_antonio-vivaldi-concertos-in-c-major-e-m_antonio-vivaldi-max-goberman-new-york-sinf/disc1/02.06.%20Concerto%20in%20A%20Minor%3A%20Allegro.mp3"];
%%
for j = 1:length(VUrls)
    websave(strcat('V',int2str(j),'.mp3'),VUrls(j));
end
%%
for j = 1:length(BUrls)
    websave(strcat('B',int2str(j),'.mp3'),BUrls(j));
end
%%
clip_num = 1;
n_freq = 100;
n_times = 100;
DataOut = spalloc(n_freq*n_times,200,4000);
for j = 1:10
    [S,Fs] = audioread(strcat('Christmas\C',int2str(j),'.mp3'));
    S = mean(S,2);
    n = length(S);
    ntrim = n-mod(n,5*Fs);
    S = S(1:ntrim);
    %V = reshape(V,[ntrim/(5*Fs) Fs*5]);
    n_clips = ntrim/(5*Fs);
    smpl = randsample(n_clips,10);
    for k = 1:10%n_clips
        k = smpl(k);
        clip = S(5*(k-1)*Fs+1:5*k*Fs);
        Sp = pspectrum(clip,Fs,'spectrogram');
        %do away with all all week signals for sparsity
        Sp = Sp.*(log(Sp)>-9);
        %Throw away high frequency information
        Sp = Sp(1:100,:);
        %downsample image
        Sp = imresize(Sp,[n_freq n_times]);
%         imagesc(Sp)
%         nnz(Sp)
%         pause(1)
        %turn image into row vector for SVD
        Sp = sparse(reshape(Sp,n_freq*n_times,1));
        %keep only clips that are representitive of music
        %checks the norm to discard clips where not much is happening
        %if norm(Sp) > 0.1
            DataOut(:,clip_num) = Sp;
            clip_num = clip_num+1;
        %end
        [j k norm(Sp) clip_num]
    end
    %keep only about 2% of the data
    %Get rid of overtones and decrease resolution
end