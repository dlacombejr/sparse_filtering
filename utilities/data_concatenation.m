path = '~/Documents/research/AWS/sparse_filtering/saved/';
addpath(genpath(path))

fileNames = dir([path, '/activation_raw*.mat']);

master = []; 
for file = 1:numel(fileNames)
    temp = load(fileNames(file).name);
    field = fieldnames(temp); 
    temp = reshape(...
        temp.(field{1}), [...
            size(temp.(field{1}), 1),...
            size(temp.(field{1}), 2),...
            size(temp.(field{1}), 3) * size(temp.(field{1}), 4)...
        ]...
    );
    master = [master; temp];
    disp(num2str(file))
end

save activations_concatenated.mat master