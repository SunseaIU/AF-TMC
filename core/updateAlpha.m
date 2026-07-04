function alpha = updateAlpha(S, Z, params)
%UPDATEALPHA Update global adaptive view weights.
%
% View weights are inferred from the Frobenius discrepancy between each
% local graph and the current consensus graph Z, then projected onto the
% probability simplex.

    assert(iscell(S), 'updateAlpha: S must be a cell array.');
    V = numel(S);
    assert(V > 0, 'updateAlpha: S cannot be empty.');

    sigma = params.sigma;
    H = zeros(V, 1);

    for v = 1:V
        S_v = S{v};
        assert(all(size(S_v) == size(Z)), ...
            'updateAlpha: each S{v} must have the same size as Z.');

        diff = S_v - Z;
        H(v) = sum(diff(:).^2);
    end

    v_input = -H / (2 * sigma);
    alpha = projSimplex(v_input);
    alpha = alpha(:);
end
