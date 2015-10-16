load weights.mat

neurons = size(layer0, 1);
plot_size = sqrt(neurons); 
dim = size(layer0, 3);

figure(1)
for i = 1:neurons
    subplot(plot_size, plot_size, i)
    imagesc(reshape(layer0(i, :, :, :), [dim, dim])); 
    colormap gray
    axis equal off
end