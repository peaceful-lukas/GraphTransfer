function U = learnU(DS, W, U, param)

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    pPairs = samplePullingPairs(DS, W, U, param);
    dU = computeGradient(DS, WX, U, cTriplets, pPairs, aux, param);
    U = update(U, dU, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('U%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end

% update
function U = update(U, dU, param)

U = U - param.lr_U * dU;


% gradient computation
function dU = computeGradient(DS, WX, U, cTriplets, pPairs, aux, param)

num_cTriplets = size(cTriplets, 1);
num_pPairs = size(pPairs, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    c_dU = WX(:, cTriplets(:, 1))*(aux(:, cTriplets(:, 3)) - aux(:, cTriplets(:, 2)))';
    c_dU = c_dU/param.c_batchSize;
    % c_dU = c_dU/norm(c_dU, 'fro');

    % grad_yi = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*aux(:, cTriplets(:, 2))';
    % grad_c  = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*aux(:, cTriplets(:, 3))';
    % c_dU = grad_yi - grad_c;
    % c_dU = c_dU/param.c_batchSize;
    % % c_dU = c_dU/norm(c_dU, 'fro');
end

p_dU = zeros(size(U));
if num_pPairs > 0
    p_dU = WX(:, pPairs(:, 1))*aux(:, pPairs(:, 2))';
    p_dU = p_dU/param.p_batchSize;
    % p_dU = p_dU/norm(p_dU, 'fro');

    % p_dU = (U(:, pPairs(:, 2)) - WX(:, pPairs(:, 1)))*aux(:, pPairs(:, 2))';
    % p_dU = 2*p_dU/param.p_batchSize;
    % p_dU = p_dU/norm(p_dU, 'fro');
end

bal_c = param.bal_c/(param.bal_c + param.bal_p);
bal_p = param.bal_p/(param.bal_c + param.bal_p);

dU = bal_c*c_dU + bal_p*p_dU + param.lambda_U*U;

