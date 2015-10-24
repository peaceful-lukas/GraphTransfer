function U = learnU(DS, W, U, param)

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    % pPairs = samplePullingPairs(DS, W, U, param);
    pTriplets = samplePullingTriplets(DS, W, U, param);
    % dU = computeGradient(DS, WX, U, cTriplets, pPairs, aux, param);
    dU = computeGradient(DS, WX, U, cTriplets, pTriplets, aux, param);
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
% function dU = computeGradient(DS, WX, U, cTriplets, pPairs, aux, param)
function dU = computeGradient(DS, WX, U, cTriplets, pTriplets, aux, param)

% sTriplets = validStructurePreservingTriplets(U, param);
% [sPairs totalNum_sPairs] = validStructurePreservingUnaryPairs(U, param);


[spLPairs total_num_spLViol] = validStructurePreservingLboundPairs(U, param);
[spUPairs total_num_spUViol] = validStructurePreservingUboundPairs(U, param);

num_cTriplets = size(cTriplets, 1);
% num_pPairs = size(pPairs, 1);
num_pTriplets = size(pTriplets);
% num_sTriplets = size(sTriplets, 1);
% num_sPairs = size(sPairs, 1);
num_spLPairs = size(spLPairs, 1);
num_spUPairs = size(spUPairs, 1);


c_dU = zeros(size(U));
if num_cTriplets > 0
    grad_yi = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*aux(:, cTriplets(:, 2))';
    grad_c  = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*aux(:, cTriplets(:, 3))';
    c_dU = grad_yi - grad_c;
    c_dU = c_dU/size(U, 2); % normalize by the number of prototypes
    c_dU = bsxfun(@rdivide, c_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
    c_dU = c_dU/param.c_batchSize; % normalize by the number of samples for SGD
end

% p_dU = zeros(size(U));
% if num_pPairs > 0
%     p_dU = (U(:, pPairs(:, 2)) - WX(:, pPairs(:, 1)))*aux(:, pPairs(:, 2))';
%     p_dU = p_dU/size(U, 2); % normalize by the number of prototypes
%     p_dU = bsxfun(@rdivide, p_dU, repelem(param.numInstancesPerClass', param.numPrototypes));
%     p_dU = 2*p_dU/param.p_batchSize; % normalize by the number of samples for SGD
% end

p_dU = zeros(size(U));
if num_pTriplets > 0
    p1 = (U(:, pTriplets(:, 2)) - WX(:, pTriplets(:, 1)))*aux(:, pTriplets(:, 2))';
    p2 = -(U(:, pTriplets(:, 3)) - WX(:, pTriplets(:, 1)))*aux(:, pTriplets(:, 3))';
    p_dU = p1 + p2;
    p_dU = p_dU/size(U, 2); % normalize by the number of prototypes
    p_dU = bsxfun(@rdivide, p_dU, repelem(param.numInstancesPerClass', param.numPrototypes));
    p_dU = 2*p_dU/param.p_batchSize; % normalize by the number of samples for SGD
end

% sTriplets
% s_dU = zeros(size(U));
% if num_sTriplets > 0
%     s1 = -2*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 1))';
%     s2 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 2))';
%     s3 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 3))';

%     s_dU = s1 + s2 + s3;
%     s_dU = s_dU/size(U, 2); % normalize by the number of prototypes
%     s_dU = s_dU/size(param.sTriplets, 1); % normalize by the number of samples for SGD
% end

% % sPairs
% sp_dU = zeros(size(U));
% if num_sPairs > 0
%     sp1 = 2*(U(:, sPairs(:, 1)) - U(:, sPairs(:, 2)))*aux(:, sPairs(:, 1))';
%     sp2 = 2*(U(:, sPairs(:, 2)) - U(:, sPairs(:, 1)))*aux(:, sPairs(:, 2))';

%     sp_dU = sp1 + sp2;
%     sp_dU = sp_dU/size(U, 2);
%     sp_dU = sp_dU/totalNum_sPairs;
% end



% neighbors ( should be close )
spU_dU = zeros(size(U));
if num_spUPairs > 0
    spU1 = 2*(U(:, spUPairs(:, 1)) - U(:, spUPairs(:, 2)))*aux(:, spUPairs(:, 1))';
    spU2 = 2*(U(:, spUPairs(:, 2)) - U(:, spUPairs(:, 1)))*aux(:, spUPairs(:, 2))';

    spU_dU = spU1 + spU2;
    spU_dU = spU_dU/size(U, 2);
    spU_dU = spU_dU/total_num_spLViol;
end


% non-neighbors ( should be distant )
spL_dU = zeros(size(U));
if num_spLPairs > 0
    spL1 = -2*(U(:, spLPairs(:, 1)) - U(:, spLPairs(:, 2)))*aux(:, spLPairs(:, 1))';
    spL2 = -2*(U(:, spLPairs(:, 2)) - U(:, spLPairs(:, 1)))*aux(:, spLPairs(:, 2))';

    spL_dU = spL1 + spL2;
    spL_dU = spL_dU/size(U, 2);
    spL_dU = spL_dU/total_num_spLViol;
end




bal_c = param.bal_c/(param.bal_c + param.bal_p + param.bal_s);
bal_p = param.bal_p/(param.bal_c + param.bal_p + param.bal_s);
bal_s = param.bal_s/(param.bal_c + param.bal_p + param.bal_s);

dU = bal_c*c_dU + bal_p*p_dU + bal_s*spL_dU + bal_s*spU_dU + param.lambda_U*U/size(U, 2);


