function [ACC, NMI, Purity, ARI, Fscore] = evalClustering(predict_labels, true_labels)
%EVALCLUSTERING Evaluate clustering labels with five common metrics.
%
% The function computes Accuracy (ACC), Normalized Mutual Information (NMI),
% Purity, Adjusted Rand Index (ARI), and pairwise F-score. Accuracy is
% computed after optimal label alignment with matchpairs.
%
% Inputs
%   predict_labels - n-by-1 predicted cluster labels.
%   true_labels    - n-by-1 ground-truth labels.
%
% Outputs
%   ACC, NMI, Purity, ARI, Fscore - Scalar evaluation metrics.

    if nargin < 2
        error('evalClustering:MissingInput', 'predict_labels and true_labels are required.');
    end

    predict_labels = predict_labels(:);
    true_labels = true_labels(:);

    if numel(predict_labels) ~= numel(true_labels)
        error('evalClustering:LengthMismatch', ...
            'Predicted and true labels must have the same length.');
    end

    N = numel(true_labels);
    if N < 2
        error('evalClustering:TooFewSamples', 'At least two samples are required.');
    end

    L_pred = unique(predict_labels);
    L_true = unique(true_labels);
    num_class = max(numel(L_pred), numel(L_true));

    cost_matrix = zeros(num_class, num_class);
    for i = 1:numel(L_pred)
        for j = 1:numel(L_true)
            cost_matrix(i, j) = -sum((predict_labels == L_pred(i)) ...
                & (true_labels == L_true(j)));
        end
    end

    assignment = matchpairs(cost_matrix, 1e5);

    mapped_labels = zeros(N, 1);
    for i = 1:size(assignment, 1)
        pred_idx = assignment(i, 1);
        true_idx = assignment(i, 2);
        if pred_idx <= numel(L_pred) && true_idx <= numel(L_true)
            idx = (predict_labels == L_pred(pred_idx));
            mapped_labels(idx) = L_true(true_idx);
        end
    end

    ACC = sum(mapped_labels == true_labels) / N;

    C = confusionmat(true_labels, predict_labels);

    P_joint = C / N;
    P_true = sum(P_joint, 2);
    P_pred = sum(P_joint, 1);

    idx_true = P_true > 0;
    idx_pred = P_pred > 0;

    H_true = -sum(P_true(idx_true) .* log2(P_true(idx_true)));
    H_pred = -sum(P_pred(idx_pred) .* log2(P_pred(idx_pred)));

    MI = 0;
    for i = 1:size(P_joint, 1)
        for j = 1:size(P_joint, 2)
            if P_joint(i, j) > 0
                MI = MI + P_joint(i, j) ...
                    * log2(P_joint(i, j) / (P_true(i) * P_pred(j)));
            end
        end
    end

    if (H_true * H_pred) == 0
        NMI = 0;
    else
        NMI = MI / sqrt(H_true * H_pred);
    end

    Purity = 0;
    for i = 1:numel(L_pred)
        idx = (predict_labels == L_pred(i));
        if any(idx)
            cluster_true_labels = true_labels(idx);
            [~, freq] = mode(cluster_true_labels);
            if isempty(freq)
                freq = 0;
            elseif numel(freq) > 1
                freq = freq(1);
            end
            Purity = Purity + freq;
        end
    end
    Purity = Purity / N;

    [~, ~, u_id] = unique(predict_labels);
    [~, ~, v_id] = unique(true_labels);
    N_u = max(u_id);
    M_v = max(v_id);

    cont_tab = zeros(N_u, M_v);
    for i = 1:N
        cont_tab(u_id(i), v_id(i)) = cont_tab(u_id(i), v_id(i)) + 1;
    end

    sum_rows = sum(cont_tab, 2);
    sum_cols = sum(cont_tab, 1);
    n_choose_2 = @(x) x .* (x - 1) / 2;

    TP = sum(sum(n_choose_2(cont_tab)));
    pred_pairs = sum(n_choose_2(sum_rows));
    true_pairs = sum(n_choose_2(sum_cols));

    expected_index = (pred_pairs * true_pairs) / n_choose_2(N);
    max_index = (pred_pairs + true_pairs) / 2;

    if expected_index == max_index
        ARI = 1;
    else
        ARI = (TP - expected_index) / (max_index - expected_index);
    end

    FP = pred_pairs - TP;
    FN = true_pairs - TP;

    if TP == 0
        Fscore = 0;
    else
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        Fscore = 2 * (precision * recall) / (precision + recall);
    end
end
