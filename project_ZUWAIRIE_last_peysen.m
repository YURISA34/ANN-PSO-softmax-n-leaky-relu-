clc; clear;

%% Step 1: Load data
data = readtable('iris_numeric.csv');
X = table2array(data(:,1:4));           
Y_labels = table2array(data(:,5));      

% One-hot encoding
Y = zeros(size(Y_labels,1), 3);
for i = 1:size(Y_labels,1)
    Y(i, Y_labels(i)+1) = 1;
end

%% Step 2: PSO Initialization
num_particles =10d0;
num_weights = 52;
num_iterations = 100;
w = 0.5; c1 = 1.2; c2 = 1.2;

X_pso = randn(num_particles, num_weights);
V = zeros(size(X_pso));
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
        X_pso(i,:) = max(min(X_pso(i,:), 5), -5);
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
title('PSO Convergence - Iris Classification'); grid on;

%% Step 5: Display Results Neatly
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
fprintf('\n✅ Final Accuracy: %.2f%%\n', accuracy);
fprintf('❌ Final Error: %.2f%%\n', 100 - accuracy);

%% Step 6: gbest Weights Output
fprintf('\n=== Final Optimized Weights (gbest) ===\n');
for i = 1:52
    fprintf('W%02d = %8.4f\t', i, gbest(i));
    if mod(i, 4) == 0, fprintf('\n'); end
end

%% Functions
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
    W1 = reshape(W(1:16), [4,4]);
    W2 = reshape(W(17:32), [4,4]);
    W3 = reshape(W(33:44), [4,3]);
    b1 = W(45:48);
    b2 = W(49:52);
    J = relu(I * W1 + b1);
    K = relu(J * W2 + b2);
    O = K * W3;
    probs = softmax(O);
end

function y = relu(x)
    y = max(0, x);
end

function s = softmax(x)
    ex = exp(x - max(x));
    s = ex / sum(ex);
end