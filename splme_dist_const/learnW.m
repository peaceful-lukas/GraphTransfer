function W = learnW(DS, W, U, param)

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    % pPairs = samplePullingPairs(DS, W, U, param);
    pTriplets = samplePullingTriplets(DS, W, U, param);
    % dW = computeGradient(DS, W, U, cTriplets, pPairs, param);
    dW = computeGradient(DS, W, U, cTriplets, pTriplets, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle) || n == 1
        timeElapsed = toc;
        fprintf('W%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end


% update
function W = update(W, dW, param)

W = W - param.lr_W * dW;



% gradient computation
% function dW = computeGradient(DS, W, U, cTriplets, pPairs, param)
function dW = computeGradient(DS, W, U, cTriplets, pTriplets, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);
% num_pPairs = size(pPairs, 1);
num_pTriplets = size(pTriplets, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    % grad_yi = 2*(W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*bsxfun(@times, X(:, cTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(cTriplets(:, 1))));
    % grad_c  = 2*(W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*bsxfun(@times, X(:, cTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(cTriplets(:, 1))));
    % c_dW = grad_yi - grad_c;
    c_dW = -2*(U(:, cTriplets(:, 2)) - U(:, cTriplets(:, 3)))*bsxfun(@times, X(:, cTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(cTriplets(:, 1))));
    c_dW = c_dW/size(W, 2); % normalize by the feature dimension
    c_dW = c_dW/param.c_batchSize;
end

% p_dW = zeros(size(W));
% if num_pPairs > 0
%     p_dW = (W*X(:, pPairs(:, 1)) - U(:, pPairs(:, 2)))*bsxfun(@times, X(:, pPairs(:, 1))', 1./param.numInstancesPerClass(DS.DL(pPairs(:, 1))));
%     p_dW = p_dW/size(W, 2);
%     p_dW = 2*p_dW/param.p_batchSize;
% end

p_dW = zeros(size(W));
if num_pTriplets > 0
    p_dW = -2*(U(:, pTriplets(:, 2)) - U(:, pTriplets(:, 3)))*bsxfun(@times, X(:, pTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(pTriplets(:, 1))));
    p_dW = p_dW/size(W, 2); % normalize by the feature dimension
    p_dW = p_dW/param.p_batchSize;
end

bal_c = param.bal_c/(param.bal_c + param.bal_p);
bal_p = param.bal_p/(param.bal_c + param.bal_p);

dW = bal_c*c_dW + bal_p*p_dW + param.lambda_W*W/size(W, 2);