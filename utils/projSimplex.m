function v_proj = projSimplex(v_input)
%PROJSIMPLEX Project a vector onto the probability simplex.
%
% Solves
%   min_v ||v - v_input||_2^2  subject to v >= 0, sum(v) = 1.
%
% The implementation keeps the original row/column orientation. It is used
% by updateS for row-wise graph projection and by updateAlpha for view
% weight projection.
%
% Input
%   v_input - Numeric row or column vector.
%
% Output
%   v_proj  - Projection onto the probability simplex.

    assert(~isempty(v_input), 'Error: Input vector v_input cannot be empty.');
    assert(isnumeric(v_input), 'Error: Input must be numeric.');
    assert(all(~isnan(v_input(:))), 'Error: Input contains NaN values.');
    assert(all(~isinf(v_input(:))), 'Error: Input contains Inf values.');
    assert(isvector(v_input), ...
        'Error: projSimplex requires a 1D vector. Matrix inputs must be processed row-wise or column-wise.');

    is_row = isrow(v_input);
    if is_row
        y = v_input';
    else
        y = v_input;
    end

    n = length(y);
    k = 1;

    % Center so that sum(g) equals the simplex mass k.
    g = y - mean(y) + (k / n);

    % Fast path: the centered vector is already feasible.
    if min(g) >= 0
        v_proj = g;
        if is_row
            v_proj = v_proj';
        end
        return;
    end

    eta = 0;
    max_iter = 200;
    tol = 1e-10;
    n_iter = 0;

    while n_iter < max_iter
        n_iter = n_iter + 1;

        v_curr = g - eta;
        posidx = v_curr > 0;
        npos = sum(posidx);

        if npos == 0
            error('projSimplex:ZeroActiveSet', ...
                'Optimization failed: active set is empty. Check input scale or parameters.');
        end

        f_val = sum(v_curr(posidx)) - k;

        if abs(f_val) < tol
            break;
        end

        df_val = -npos;
        eta = eta - f_val / df_val;
    end

    v_proj = max(g - eta, 0);

    if is_row
        v_proj = v_proj';
    end
end
