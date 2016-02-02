% make the pseudo-colormap
dimensions = 16;
matrix = zeros(dimensions, dimensions, 3); 

red = repmat(linspace(0, 255, dimensions), [dimensions, 1]);
green = repmat(linspace(0, 255, dimensions), [dimensions, 1])';
blue = repmat(linspace(255, 0, dimensions), [dimensions, 1]);

matrix(:, :, 1) = red;
matrix(:, :, 2) = green; 
matrix(:, :, 3) = blue;

% load in the saved parameters
load saved/sample_gSF/params2.mat params  % params.mat
neurons = size(params, 1);
horizontal = params(:, 6); 
vertical = params(:, 7); 

% clip the values for horizonal and vertical
horizontal(horizontal > 7) = 7;
horizontal(horizontal < -8) = -8;
vertical(vertical > 7) = 7;
vertical(vertical < -8) = -8;

% scale to positive values
horizontal = horizontal + 9;
vertical = vertical + 9;

% create an image showing locations of all neurons
image = zeros(neurons, 3);
for neuron = 1:neurons
    
    image(neuron, :) = matrix(...
        round(vertical(neuron)), ...
        round(horizontal(neuron)), ...
        : ...
    );

end

% reshape into image and flip to be consistent with Python
image = reshape(image, [sqrt(neurons), sqrt(neurons), 3]);
image = permute(image,[2 1 3]);

%display the image 
figure(1)
subplot(1, 2, 1)
imshow(uint8(image))
g = subplot(1, 2, 2);
imshow(uint8(matrix))
axis('equal')
