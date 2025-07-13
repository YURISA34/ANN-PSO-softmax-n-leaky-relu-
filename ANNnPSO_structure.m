clc; clear;

% Step 1: Load data
data = readtable('iris_numeric.csv');
X = table2array(data(:,1:4));           
Y_labels = table2array(data(:,5));      

% One-hot encoding for training
Y = zeros(size(Y_labels,1), 3);
for i = 1:size(Y_labels,1)
    Y(i, Y_labels(i)+1) = 1;
end

%% Step 2: PSO Initialization
num_particles = 100;
num_weights = 44;  % 16 (W1) + 16 (W2) + 12 (W3), no bias
num_iterations = 100;
w = 0.5; c1 = 1.2; c2 = 1.2;

X_pso = randn(num_particles, num_weights);   % Random weights
V = zeros(size(X_pso));                      % Initial velocity

pbest = X_pso;
pbest_error = arrayfun(@(i) computeError(X, Y_labels, X_pso(i,:)), 1:num_particles);

[gbest_error, idx] = min(pbest_error);
gbest = pbest(idx,:);
convergence = zeros(1, num_iterations);
convergence(1) = gbest_error;

%% Step 3: PSO Optimization Loop
for t = 1:num_iterations
    for i = 1:num_particles
        r1 = rand(1, num_weights);
        r2 = rand(1, num_weights);
        V(i,:) = w * V(i,:) + ...
                 c1 * r1 .* (pbest(i,:) - X_pso(i,:)) + ...
                 c2 * r2 .* (gbest - X_pso(i,:));
        X_pso(i,:) = X_pso(i,:) + V(i,:);
        X_pso(i,:) = max(min(X_pso(i,:), 5), -5);  % clip to range
        err = computeError(X, Y_labels, X_pso(i,:));
        if err < pbest_error(i)
            pbest(i,:) = X_pso(i,:);
            pbest_error(i) = err;
        end
    end
    [gbest_error, idx] = min(pbest_error);
    gbest = pbest(idx,:);
    convergence(t) = gbest_error;
    fprintf('Iteration %3d | Best Error: %.2f%%\n', t, gbest_error);
end

%% Step 4: Plot Convergence
figure;
plot(1:num_iterations, convergence, '-o', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Classification Error (%)');
title('PSO Convergence - Iris Classification ');
grid on;

%% Step 5: Final Evaluation
fprintf('\n%-4s %-8s %-10s %-36s\n', 'Row', 'Actual', 'Predicted', 'Softmax Probabilities');
fprintf('%s\n', repmat('-', 1, 65));

correct = 0;
for i = 1:size(X,1)
    probs = forwardPass(X(i,:), gbest);
    [~, pred] = max(probs);
    actual = Y_labels(i);
    if pred-1 == actual
        correct = correct + 1;
    end
    fprintf('%-4d %-8d %-10d %.4f  %.4f  %.4f\n', i, actual, pred-1, probs(1), probs(2), probs(3));
end
accuracy = (correct / size(X,1)) * 100;
fprintf('\n Final Accuracy: %.2f%%\n', accuracy);
fprintf(' Final Error: %.2f%%\n', 100 - accuracy);

%% Step 6: gbest Weights Output
fprintf('\n=== Final Optimized Weights (gbest, 44 weights) ===\n');
for i = 1:44
    fprintf('W%02d = %8.4f\t', i, gbest(i));
    if mod(i, 4) == 0, fprintf('\n'); end
end

%% ===== Functions =====

function error_pct = computeError(X, Y_labels, W)
    correct = 0;
    for i = 1:size(X,1)
        probs = forwardPass(X(i,:), W);
        [~, pred] = max(probs);
        if pred-1 == Y_labels(i)
            correct = correct + 1;
        end
    end
    error_pct = (1 - correct / size(X,1)) * 100;
end

function probs = forwardPass(I, W)
    % Reshape weight matrices (no biases)
    W1 = reshape(W(1:16), [4,4]);       % Input to Hidden 1
    W2 = reshape(W(17:32), [4,4]);      % Hidden 1 to Hidden 2
    W3 = reshape(W(33:44), [4,3]);      % Hidden 2 to Output

    J = I * W1;     % Layer 1 linear output
    K = J * W2;     % Layer 2 linear output
    O = K * W3;     % Output layer linear output
    probs = softmax(O);  % Final softmax
end

function s = softmax(x)
    ex = exp(x - max(x));  % numerical stability
    s = ex / sum(ex);
end