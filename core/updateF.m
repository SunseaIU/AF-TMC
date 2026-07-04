function F = updateF(Z, c, ~)
%UPDATEF Update the continuous spectral embedding F.
%
% The embedding is formed by the eigenvectors corresponding to the c
% smallest eigenvalues of the graph Laplacian induced by Z.
%
% Inputs
%   Z - n-by-n consensus matrix.
%   c - Number of embedding dimensions / clusters.
%
% Output
%   F - n-by-c spectral embedding.

    assert(~isempty(Z), 'updateF: Z cannot be empty.');
    assert(ismatrix(Z) && size(Z, 1) == size(Z, 2), 'updateF: Z must be square.');
    assert(c > 0 && mod(c, 1) == 0 && c <= size(Z, 1), ...
        'updateF: c must be a positive integer not larger than size(Z, 1).');

    N = size(Z, 1);
    Z_sym = (Z + Z') / 2;
    L_Z = diag(sum(Z_sym, 2)) - Z_sym + eye(N) * 1e-10;

    % Full eigendecomposition is simple and stable for moderate n. For very
    % large data sets, replace this with eigs for the smallest eigenpairs.
    [V_eig, D_eig] = eig(full(L_Z));

    [~, sorted_idx] = sort(real(diag(D_eig)), 'ascend');
    F = real(V_eig(:, sorted_idx(1:c)));
end
