% NORMALIZATION FUNCTION

function A_norm = norm_f(A)
    % Normalize the input A to be in range [0, 1]
    % A_norm = norm_f(A) returns a normalized array A_norm of same size as A
    
    A_min = min(A, [], 'all');
    A_max = max(A, [], 'all');
    
    % If the max and min are equal, normalization is not possible
    if A_min == A_max 
        error('Normalization not possible')
    end

    A_norm = (A - A_min) ./ (A_max - A_min);
 
end 



    