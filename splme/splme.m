% function [W U param] = splme(DS, param, local_env)

% init
[U param] = initU(DS, param);
W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');


n = 0;
highest_acc = 0.3;
highest_W = W;
highest_U = U;
iter_condition = 1;

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW(DS, W, U, param);
    U = learnU(DS, W, U, param);

    [~, accuracy] = dispAccuracy(param.method, DS, W, U, param);

    if accuracy > highest_acc
        saveResult(param.method, param.dataset, accuracy, {param, W, U, accuracy}, local_env);

        highest_acc = accuracy;
        highest_W = W;
        highest_U = U;
        fprintf('[splme] highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;

    n = n + 1;
end

W = highest_W;
U = highest_U;


coord_idx = visualizePrototypes(U, param, [], []);