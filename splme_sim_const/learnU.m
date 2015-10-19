function U = learnU(DS, W, U, param)

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU
    mTriplets = sampleMembershipTriplets(DS, W, U, param);
    pPairs = samplePullingPairs(DS, W, U, param);
    dU = computeGradient(DS, WX, U, mTriplets, pPairs, aux, param);
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
function dU = computeGradient(DS, WX, U, mTriplets, pPairs, aux, param)

[gPairs total_num_gPairs] = sampleGatheringPairs(DS, U, param);

num_mTriplets = size(mTriplets, 1);
num_pPairs = size(pPairs, 1);
num_gPairs = size(gPairs, 1);


m_dU = zeros(size(U));
if num_mTriplets > 0
    m_dU = WX(:, mTriplets(:, 1))*(aux(:, mTriplets(:, 3)) - aux(:, mTriplets(:, 2)))';
    m_dU = m_dU/param.m_batchSize;
end

p_dU = zeros(size(U));
if num_pPairs > 0
    p_dU = WX(:, pPairs(:, 1))*aux(:, pPairs(:, 2))';
    p_dU = p_dU/param.p_batchSize;
end

g_dU = zeros(size(U));
if num_gPairs > 0
    g_dU_1 = 2*(U(:, gPairs(:, 1)) - U(:, gPairs(:, 2)))*aux(:, gPairs(:, 1))';
    g_dU_2 = 2*(U(:, gPairs(:, 2)) - U(:, gPairs(:, 1)))*aux(:, gPairs(:, 2))';
    g_dU = g_dU_1 + g_dU_2;
    g_dU = g_dU/total_num_gPairs;
end

bal_m = param.bal_m/(param.bal_m + param.bal_p + param.bal_g);
bal_p = param.bal_p/(param.bal_m + param.bal_p + param.bal_g);
bal_g = param.bal_g/(param.bal_m + param.bal_p + param.bal_g);

dU = bal_m*m_dU + bal_p*p_dU + bal_g*g_dU + param.lambda_U*U;




function [gPairs total_num_gPairs] = sampleGatheringPairs(DS, U, param)

gPairs = [];

protoOffset = [0; cumsum(param.numPrototypes)];
for classNum=1:param.numClasses
    [p1_idx, p2_idx] = find(triu(param.knnGraphs{classNum}));

    p1_idx = p1_idx + protoOffset(classNum);
    p2_idx = p2_idx + protoOffset(classNum);

    gPairs = [gPairs; p1_idx p2_idx];
end
total_num_gPairs = size(gPairs, 1);

loss_vec = sum((U(:, gPairs(:, 1)) - U(:, gPairs(:, 2))).^2, 1) - param.g_sigma;
valids = find(loss_vec > 0);

gPairs = gPairs(valids, :);

