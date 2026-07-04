function S = updateS(X, Theta, Z, alpha, params)
%UPDATES Update local affinity graphs for all views.
%
% Each row of S{v} is obtained by projecting a fused distance score onto
% the probability simplex. The fused score combines feature-space distance
% and the current global consensus manifold discrepancy.

    assert(iscell(X) && iscell(Theta), 'updateS: X and Theta must be cell arrays.');
    V = numel(X);
    assert(numel(Theta) == V && numel(alpha) == V, ...
        'updateS: X, Theta, and alpha must have the same number of views.');

    N = size(X{1}, 1);
    assert(all(size(Z) == [N, N]), 'updateS: Z must be n-by-n.');

    beta = params.beta;
    gamma_fixed = params.gamma;
    S = cell(V, 1);

    % Consensus-manifold discrepancy. The offset prevents undefined cosine
    % distances for zero rows.
    Z_safe = Z + 1e-12;
    DZ = 0.5 * pdist2(Z_safe, Z_safe, 'cosine');

    for v = 1:V
        X_v = X{v};
        theta_v = Theta{v};
        alpha_v = alpha(v);

        assert(size(X_v, 1) == N, 'updateS: all views must have the same sample count.');
        assert(numel(theta_v) == size(X_v, 2), ...
            'updateS: Theta{v} length must match the feature count of X{v}.');

        % Feature-weighted view representation.
        X_scaled = X_v .* sqrt(theta_v(:)');
        X_scaled = X_scaled + 1e-12;
        DX_v = pdist2(X_scaled, X_scaled, 'cosine');

        D_total = DX_v + beta * alpha_v * DZ;

        % Suppress self-loops before row-wise simplex projection.
        D_total(1:N+1:end) = max(D_total(:)) + 1e8;

        S_v = zeros(N, N);
        for i = 1:N
            v_input = -D_total(i, :) / (2 * gamma_fixed);
            S_v(i, :) = projSimplex(v_input);
        end

        S{v} = (S_v + S_v') / 2;
    end
end
