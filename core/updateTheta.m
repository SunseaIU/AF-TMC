function Theta = updateTheta(X, S)
%UPDATETHETA Update adaptive feature weights for every view.
%
% The weight of a feature is inversely related to its local structural
% variation under the current view-specific graph.
%
% Inputs
%   X - V-by-1 cell array of data matrices.
%   S - V-by-1 cell array of local affinity graphs.
%
% Output
%   Theta - V-by-1 cell array. Theta{v} is a d_v-by-1 weight vector.

    assert(iscell(X) && iscell(S), 'updateTheta: X and S must be cell arrays.');
    V = numel(X);
    assert(numel(S) == V, 'updateTheta: X and S must have the same number of views.');

    Theta = cell(V, 1);

    for v = 1:V
        X_v = X{v};
        S_v = S{v};
        N = size(X_v, 1);
        assert(all(size(S_v) == [N, N]), 'updateTheta: each S{v} must be n-by-n.');

        % Symmetric graph Laplacian for the current local structure.
        degree_vec = sum(S_v, 2) + sum(S_v, 1)';
        L_sym = spdiags(degree_vec, 0, N, N) - S_v - S_v';

        % h_v measures local structural variation for each feature. The
        % lower bound prevents constant features from causing division by 0.
        h_v = max(diag(X_v' * L_sym * X_v), 1e-10);
        inv_h = 1 ./ h_v;
        theta_v = inv_h / sum(inv_h);

        Theta{v} = theta_v(:);
    end
end
