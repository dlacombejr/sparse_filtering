load weights.mat

neurons = size(layer0, 1);
dim = sqrt(size(layer0, 2)); 

for neuron = 1:neurons
%     Y = fft(reshape(layer0(neuron, :), [dim, dim])); 
%     imagesc(mat2gray(abs(fftshift(Y))))
%     pause()
    
    % dft 2d
    NFFTY = 2^nextpow2(dim);
    NFFTX = 2^nextpow2(dim);
    % 'detrend' data to eliminate zero frequency component
     X = reshape(layer0(neuron, :), [dim, dim]);
    av = sum(X(:)) / length(X(:));
    X = X - av;
    % this section I'm not too sure about, I pretty much copied the example
    % from the matlab 1D fft and adapted it slightly for 2D
    samplingFreq = 1;
    spatialFreqsX = samplingFreq/2*linspace(0,1,NFFTX/2+1);
    spatialFreqsY = samplingFreq/2*linspace(0,1,NFFTY/2+1);
    spectrum2D = fft2(X, NFFTY,NFFTX);
    % shift the 2D spectrum. I'm not entirely sure why, but all the examples
    % I've found seem to do it.
    spectrum2D = fftshift(spectrum2D);
    contourf(spatialFreqsX, spatialFreqsY, abs(spectrum2D(1:NFFTY/2+1, 1:NFFTX/2+1)))
    pause()
    
end
