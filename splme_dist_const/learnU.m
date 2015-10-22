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

    if ~mod(n, dispCycle) || n == 1
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

sTriplets = validStructurePreservingTriplets(U, param);

num_sTriplets = size(sTriplets, 1);
num_cTriplets = size(cTriplets, 1);
num_pPairs = size(pPairs, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    grad_yi = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*aux(:, cTriplets(:, 2))';
    grad_c  = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*aux(:, cTriplets(:, 3))';
    c_dU = grad_yi - grad_c;
    c_dU = c_dU/size(U, 2); % normalize by the number of prototypes
    c_dU = bsxfun(@rdivide, c_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
    c_dU = c_dU/param.c_batchSize; % normalize by the number of samples for SGD
end

p_dU = zeros(size(U));
if num_pPairs > 0
    p_dU = (U(:, pPairs(:, 2)) - WX(:, pPairs(:, 1)))*aux(:, pPairs(:, 2))';
    p_dU = p_dU/size(U, 2); % normalize by the number of prototypes
    p_dU = bsxfun(@rdivide, p_dU, repelem(param.numInstancesPerClass', param.numPrototypes));
    p_dU = 2*p_dU/param.p_batchSize; % normalize by the number of samples for SGD
end

s_dU = zeros(size(U));
if num_sTriplets > 0
    s1 = -2*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 1))';
    s2 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 2))';
    s3 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 3))';

    s_dU = s1 + s2 + s3;
    s_dU = s_dU/size(U, 2); % normalize by the number of prototypes
    s_dU = s_dU/size(param.sTriplets, 1); % normalize by the number of samples for SGD
end


bal_c = param.bal_c/(param.bal_c + param.bal_p + param.bal_s);
bal_p = param.bal_p/(param.bal_c + param.bal_p + param.bal_s);
bal_s = param.bal_s/(param.bal_c + param.bal_p + param.bal_s);

dU = bal_c*c_dU + bal_p*p_dU + bal_s*s_dU + param.lambda_U*U/size(U, 2);










