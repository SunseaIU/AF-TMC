function demo()
% DEMO Out-of-the-box demonstration script for AF-TMC.
%   This script loads a sample multi-view dataset, normalizes the features,
%   runs the core AF-TMC algorithm with default hyperparameters, and evaluates
%   the clustering performance (ACC, NMI, ARI, F-score).

    clear; clc; close all;
    
    %% 1. Environment Setup & Data Loading
    addpath(genpath('.'));    
    
    dataset_name = '3Sources'; 
    fprintf('====== Starting AF-TMC Algorithm Demo ======\n');
    fprintf('Loading sample dataset: %s.mat ...\n', dataset_name);
    
    try
        if exist([dataset_name, '.mat'], 'file')
            load([dataset_name, '.mat']);
        elseif exist(fullfile('data', [dataset_name, '.mat']), 'file')
            load(fullfile('data', [dataset_name, '.mat']));
        else
            error('Dataset file not found. Ensure %s.mat is in the current directory or data/ folder.', dataset_name);
        end
    catch ME
        error('Failed to load dataset: %s', ME.message);
    end
    
    % Handle different naming conventions for ground truth labels
    if exist('y', 'var') && ~exist('Y', 'var'), Y = y; end
    if exist('truth', 'var') && ~exist('Y', 'var'), Y = truth; end
    if exist('gnd', 'var') && ~exist('Y', 'var'), Y = gnd; end
    
    C = length(unique(Y));     % Number of clusters
    V_num = length(X);         % Number of views
    
    %% 2. Data Preprocessing (L2 Normalization)
    fprintf('Preprocessing data (L2 Normalization)...\n');
    for i = 1 : V_num
        X{i} = X{i} ./ max(sqrt(sum(X{i}.^2, 2)), eps); 
    end
    
    %% 3. Hyperparameter Configuration
    % Standard/Robust default hyperparameters for demonstration
    params.k        = 5;       % Number of neighbors
    params.gamma    = 1.0;     % View weight parameter
    params.mu       = 1.0;     % Embedding/Fidelity manifold constraint
    params.beta     = 1e-3;    % Regularization parameter
    params.sigma    = 10;      % Gaussian kernel scale
    params.max_iter = 30;      % Maximum iterations
    params.tol      = 1e-4;    % Convergence tolerance
    
    fprintf('\nRunning with the following configuration:\n');
    disp(params);

    %% 4. Core Algorithm Execution
    fprintf('Executing core AF-TMC algorithm...\n');
    tic;
    try
        % Invoke the core function
        [~, ~, F_res, ~, ~] = AF_TMC(X, C, params); 
        
        % Normalize the embedding representations
        F_norm = F_res ./ sqrt(sum(F_res.^2, 2) + eps);
        F_norm(isnan(F_norm)) = 0; 
        
        % Perform post-processing K-means to extract final labels
        fprintf('Running K-means on the learned embeddings...\n');
        lbl = kmeans(F_norm, C, 'MaxIter', 100, 'Replicates', 10, 'Display', 'off');
        
        runtime = toc;
        fprintf('Algorithm finished in %.2f seconds.\n', runtime);
        
        %% 5. Clustering Evaluation
        % Standard evaluation order: [ACC, NMI, Purity, ARI, Fscore]
        [curr_acc, curr_nmi, ~, curr_ari, curr_fscore] = evalClustering(lbl, Y);
        
        %% 6. Display Results
        fprintf('\n================ Performance Summary ================ \n');
        fprintf('Dataset: %s\n', dataset_name);
        fprintf('Metrics:\n');
        fprintf('  - ACC (Accuracy):                    %.4f\n', curr_acc);
        fprintf('  - NMI (Normalized Mutual Info):      %.4f\n', curr_nmi);
        fprintf('  - ARI (Adjusted Rand Index):         %.4f\n', curr_ari);
        fprintf('  - F-score:                           %.4f\n', curr_fscore);
        fprintf('=====================================================\n');
        
    catch ME
        fprintf('\n❌ Execution Error! Please check core API interfaces or data formats.\n');
        rethrow(ME);
    end
end