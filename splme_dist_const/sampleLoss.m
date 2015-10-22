function loss = sampleLoss(DS, W, U, param)

X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);
% pPairs = samplePullingPairs(DS, W, U, param);
pTriplets = samplePullingTriplets(DS, W, U, param);
sTriplets = validStructurePreservingTriplets(U, param);

num_cTriplets = size(cTriplets, 1);
% num_pPairs = size(pPairs, 1);
num_pTriplets = size(pTriplets, 1);
num_sTriplets = size(sTriplets, 1);


cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    dist_yi_vec = sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2))).^2, 1);
    dist_c_vec = sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3))).^2, 1);
    cErr_vec = param.c_lm + dist_yi_vec - dist_c_vec;
    viol = find(cErr_vec > 0);
    num_cV = length(viol);
    if viol > 0
        cErr = sum(cErr_vec(viol));
    end
end
cErr = cErr/param.c_batchSize;

% pErr = 0;
% num_pV = 0;
% if num_pPairs > 0
%     pErr_vec = sum((W*X(:, pPairs(:, 1)) - U(:, pPairs(:, 2))).^2, 1) - param.p_sigma;
%     viol = find(pErr_vec > 0);
%     num_pV = length(viol);
%     if viol > 0
%         pErr = sum(pErr_vec(viol));
%     end
% end
% pErr = pErr/param.p_batchSize;

pErr = 0;
num_pV = 0;
if num_pTriplets > 0
    pErr_vec = param.p_lm + sum((W*X(:, pTriplets(:, 1)) - U(:, pTriplets(:, 2))).^2, 1) - sum((W*X(:, pTriplets(:, 1)) - U(:, pTriplets(:, 3))).^2, 1);
    viol = find(pErr_vec > 0);
    num_pV = length(viol);
    if viol > 0
        pErr = sum(pErr_vec(viol));
    end
end
pErr = pErr/param.p_batchSize;


sErr = 0;
num_sV = 0;
if num_sTriplets > 0
    sErr_vec = param.s_lm + sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)).^2) - sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)).^2);
    viol = find(sErr_vec > 0);
    num_sV = length(viol);
    if viol > 0
        sErr = sum(sErr_vec(viol));
    end
end
sErr = sErr/size(param.sTriplets, 1);

bal_c = param.bal_c/(param.bal_c + param.bal_p + param.bal_s);
bal_p = param.bal_p/(param.bal_c + param.bal_p + param.bal_s);
bal_s = param.bal_s/(param.bal_c + param.bal_p + param.bal_s);

cErr = bal_c*cErr;
pErr = bal_p*pErr;
sErr = bal_s*sErr;

loss = bal_c*cErr + bal_p*pErr + bal_s*sErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('cV: %d / pV: %d / sV: %d / cE: %f / pE: %f / sE: %f / normW: %f / normU: %f / ', num_cV, num_pV, num_sV, cErr, pErr, sErr, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));