% FITNESS FUNCTION

function fitness = fitnessFunction(weights, base_fis, features, label_mask)
% Evaluate how good a set of rules weights is
%
% INPUTS:
% weights: weights of the FIS rules
% fis: defined Fuzzy Inference System
% features: set of useful features (mean, std, energy, ...)
% goldStandard: mask given in the dataset and what we want to achieve
%
% OUTPUT:
% fitness: difference between obtained mask and goldStandard to minimize
    
    % Define the fis and the number of rules
    fis = base_fis;
    num_rules = length(fis.Rules);
    
    % Apply to each FIS rule the weight 
    for i = 1:num_rules
        fis.Rules(i).Weight = weights(i);
    end
    
    % Define the dimensions
    [dimX, dimY] = size(label_mask);

    opt_scores = zeros(dimX, dimY);
    % Evaluate the FIS for each pixel using all the defined features
    for i = 1 : dimX
        for j = 1 : dimY
            opt_scores(i,j) = evalfis(fis, [ ...
                features{1}(i, j), ...
                features{2}(i, j), ...
                features{3}(i, j), ...
                features{4}(i, j), ...
                features{5}(i, j), ...
                features{6}(i, j), ...
                features{7}(i, j), ...
                features{8}(i, j), ...
                features{9}(i, j), ...
                features{10}(i, j), ...
                features{11}(i, j)
                ]);
        end
    end
    
    % Normalize the obtain scores
    opt_scores = norm_f(opt_scores);

    % Set a threshold to obtain the mask
    opt_mask = opt_scores > 0.55;

    % Define the objective fuction:
    % get the pixels where the obtain maskes is different from the gold
    % standard
    fitness = sum(opt_mask(:) ~= label_mask(:));

end


