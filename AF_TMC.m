function [alpha, Z, F, S, Theta, obj_history] = AF_TMC(X, c, params)
%AF_TMC Adaptive Feature-weighted Topological Manifold Clustering.
%
% Syntax
%   [alpha, Z, F, S, Theta, obj_history] = AF_TMC(X, c, params)
%
% Inputs
%   X      - V-by-1 cell array. X{v} is an n-by-d_v data matrix for view v.
%   c      - Number of target clusters / spectral embedding dimensions.
%   params - Structure with scalar fields:
%            gamma, mu, beta, sigma, max_iter, tol.
%
% Outputs
%   alpha       - V-by-1 adaptive view-weight vector.
%   Z           - n-by-n unified topological consensus manifold matrix.
%   F           - n-by-c continuous spectral embedding.
%   S           - V-by-1 cell array of local affinity graphs.
%   Theta       - V-by-1 cell array of feature-weight vectors.
%   obj_history - Objective values recorded after initialization and updates.

    addProjectPaths();
    [X, params] = validateInputs(X, c, params);

    V = numel(X);
    n = size(X{1}, 1);
    max_iter = params.max_iter;
    tol = params.tol;

    alpha = ones(V, 1) / V;
    Theta = cell(V, 1);
    S = cell(V, 1);

    % Warm-start each view with uniform feature weights and a projected
    % cosine-distance graph. The later updateS step performs the main graph
    % optimization.
    for v = 1:V
        d_v = size(X{v}, 2);
        Theta{v} = ones(d_v, 1) / d_v;

        X_safe = X{v} + 1e-12;
        dist_X = pdist2(X_safe, X_safe, 'cosine');
        S_v = zeros(n, n);
        for i = 1:n
            S_v(i, :) = projSimplex(dist_X(i, :));
        end
        S{v} = (S_v + S_v') / 2;
    end

    Z = zeros(n, n);
    for v = 1:V
        Z = Z + alpha(v) * S{v};
    end

    F = updateF(Z, c, params);

    obj_history = zeros(max_iter, 1);
    obj_history(1) = computeObjective(X, Theta, S, Z, F, alpha, params);

    for iter = 2:max_iter
        Theta = updateTheta(X, S);
        S = updateS(X, Theta, Z, alpha, params);
        Z = updateZ(S, F, alpha, params);
        F = updateF(Z, c, params);
        alpha = updateAlpha(S, Z, params);

        obj_history(iter) = computeObjective(X, Theta, S, Z, F, alpha, params);
        rel_change = abs(obj_history(iter - 1) - obj_history(iter)) ...
            / (abs(obj_history(iter - 1)) + eps);

        if rel_change < tol
            obj_history(iter + 1:end) = [];
            break;
        end
    end
end

function addProjectPaths()
%ADDPROJECTPATHS Make core and utility functions available from any caller.

    persistent paths_added
    if isempty(paths_added)
        project_root = fileparts(mfilename('fullpath'));
        addpath(fullfile(project_root, 'core'));
        addpath(fullfile(project_root, 'utils'));
        paths_added = true;
    end
end

function [X, params] = validateInputs(X, c, params)
%VALIDATEINPUTS Fail early with clear messages for public API usage.

    if ~iscell(X) || isempty(X)
        error('AF_TMC:InvalidX', 'X must be a non-empty cell array.');
    end
    X = X(:);

    n = size(X{1}, 1);
    if n < 2
        error('AF_TMC:InvalidSampleCount', 'Each view must contain at least two samples.');
    end

    for v = 1:numel(X)
        if ~isnumeric(X{v}) || ~ismatrix(X{v}) || isempty(X{v})
            error('AF_TMC:InvalidView', 'Each X{v} must be a non-empty numeric matrix.');
        end
        if size(X{v}, 1) ~= n
            error('AF_TMC:InconsistentSamples', 'All views must have the same number of rows.');
        end
        if size(X{v}, 2) < 1
            error('AF_TMC:InvalidFeatureCount', 'Each view must contain at least one feature.');
        end
        if any(~isfinite(X{v}(:)))
            error('AF_TMC:NonFiniteData', 'Input views must not contain NaN or Inf values.');
        end
    end

    if ~isscalar(c) || ~isnumeric(c) || c < 1 || c ~= floor(c) || c > n
        error('AF_TMC:InvalidClusterCount', ...
            'c must be a positive integer not larger than the sample count.');
    end

    required_fields = {'gamma', 'mu', 'beta', 'sigma', 'max_iter', 'tol'};
    for i = 1:numel(required_fields)
        field = required_fields{i};
        if ~isfield(params, field)
            error('AF_TMC:MissingParameter', 'params.%s is required.', field);
        end
        value = params.(field);
        if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value)
            error('AF_TMC:InvalidParameter', ...
                'params.%s must be a finite numeric scalar.', field);
        end
    end

    positive_fields = {'gamma', 'mu', 'beta', 'sigma', 'tol'};
    for i = 1:numel(positive_fields)
        field = positive_fields{i};
        if params.(field) <= 0
            error('AF_TMC:InvalidParameter', 'params.%s must be positive.', field);
        end
    end

    if params.max_iter < 1 || params.max_iter ~= floor(params.max_iter)
        error('AF_TMC:InvalidParameter', 'params.max_iter must be a positive integer.');
    end
end
