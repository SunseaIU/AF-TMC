function J_val = computeObjective(X, Theta, S, Z, F, alpha, params)
%COMPUTEOBJECTIVE Compute the AF-TMC objective value.
%
% The value is used for convergence monitoring. This helper mirrors the
% update steps in AF_TMC and is intentionally kept side-effect free.

    assert(iscell(X) && iscell(S) && iscell(Theta), ...
        'computeObjective: X, S, and Theta must be cell arrays.');

    V = numel(X);
    assert(numel(S) == V && numel(Theta) == V && numel(alpha) == V, ...
        'computeObjective: view counts in X, S, Theta, and alpha must match.');

    n = size(Z, 1);
    gamma = params.gamma;
    mu = params.mu;
    beta = params.beta;
    sigma = params.sigma;

    loss_local_dist = 0;
    loss_s_reg = 0;
    loss_manifold_align = 0;

    for v = 1:V
        X_v = X{v};
        S_v = S{v};
        theta_v = Theta{v};
        alpha_v = alpha(v);

        X_scaled = X_v .* (theta_v(:)') + 1e-12;
        DX_v = pdist2(X_scaled, X_scaled, 'cosine');
        loss_local_dist = loss_local_dist + sum(sum(S_v .* DX_v));

        loss_s_reg = loss_s_reg + gamma * sum(sum(S_v .^ 2));

        d_v = sum(S_v, 2);
        d_v_inv_sqrt = 1 ./ sqrt(d_v + eps);
        D_v_inv_sqrt = spdiags(d_v_inv_sqrt, 0, n, n);
        L_norm_v = speye(n) - D_v_inv_sqrt * S_v * D_v_inv_sqrt;

        loss_manifold_align = loss_manifold_align + alpha_v * trace(Z' * L_norm_v * Z);
    end

    loss_z_reg = mu * sum(sum((Z - eye(n)) .^ 2));

    D_Z = spdiags(sum(Z, 2), 0, n, n);
    L_Z = D_Z - (Z + Z') / 2;
    loss_spectral = beta * trace(F' * L_Z * F);

    loss_alpha_reg = sigma * (alpha(:)' * alpha(:));

    J_val = loss_local_dist + loss_s_reg + loss_manifold_align ...
        + loss_z_reg + loss_spectral + loss_alpha_reg;

    assert(isfinite(J_val), 'computeObjective: objective became NaN or Inf.');
end
