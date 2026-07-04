function Z = updateZ(S, F, alpha, params)
%UPDATEZ Update the unified topological consensus matrix Z.
%
% Inputs
%   S      - V-by-1 cell array of local affinity graphs.
%   F      - n-by-c continuous spectral embedding.
%   alpha  - V-by-1 view-weight vector.
%   params - Structure with fields mu and beta.
%
% Output
%   Z      - n-by-n nonnegative symmetric consensus matrix.

    assert(iscell(S), 'updateZ: S must be a cell array.');
    assert(~isempty(F) && isnumeric(F), 'updateZ: F must be a non-empty numeric matrix.');
    assert(numel(S) == numel(alpha), ...
        'updateZ: S and alpha must have the same number of views.');

    V = numel(S);
    n = size(F, 1);
    mu = params.mu;
    beta = params.beta;

    % Pairwise embedding discrepancy. A tiny offset avoids undefined cosine
    % distances when a row is exactly zero.
    F_safe = F + 1e-12;
    P = 0.5 * pdist2(F_safe, F_safe, 'cosine');

    L_con = zeros(n, n);
    I_n = eye(n);

    for v = 1:V
        S_v = S{v};
        assert(all(size(S_v) == [n, n]), 'updateZ: each S{v} must be n-by-n.');

        d_v = sum(S_v, 2) + 1e-10;
        D_inv_sqrt = spdiags(d_v.^(-0.5), 0, n, n);
        L_norm_v = I_n - (D_inv_sqrt * S_v * D_inv_sqrt);

        L_con = L_con + alpha(v) * L_norm_v;
    end

    % Solve (L_con + mu*I) * Z = mu*I - beta/2*P without forming inv(A).
    A = L_con + mu * I_n;
    B = mu * I_n - (beta / 2) * P;
    Z = A \ B;

    Z = max(Z, 0);
    Z = (Z + Z') / 2;
end
