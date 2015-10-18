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
end

p_dU = zeros(size(U));
if num_pPairs > 0
    p_dU = WX(:, pPairs(:, 1))*aux(:, pPairs(:, 2))';
    p_dU = p_dU/param.p_batchSize;
end

bal_c = param.bal_c/(param.bal_c + param.bal_p);
bal_p = param.bal_p/(param.bal_c + param.bal_p);

dU = bal_c*c_dU + bal_p*p_dU + param.lambda_U*U;


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Gathering term
% gPairs = sampleGatheringPairs(U, param);
% num_gPairs = size(gPairs, 1);

% g_dU = zeros(size(U));
% if num_gPairs > 0
%     g_dU = (U(:, gPairs(:, 1)) - U(:, gPairs(:, 2)))*
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% function gPairs = sampleGatheringPairs(U, param)

% gPairs = [];

% protoStartIdx = [0; cumsum(param.numPrototypes)];
% for classNum=1:param.numClasses
%      [p1_idx, p2_idx] = find(param.knnGraphs{classNum});
     
%      p1_idx = p1_idx + protoStartIdx(classNum);
%      p2_idx = p2_idx + protoStartIdx(classNum);

%      gPairs = [gPairs; p1_idx p2_idx];
% end


