function loss = sampleLoss(DS, W, U, param)

X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);
% pPairs = samplePullingPairs(DS, W, U, param);
pTriplets = samplePullingTriplets(DS, W, U, param);
% sTriplets = validStructurePreservingTriplets(U, param);
% [sPairs totalNum_sPairs] = validStructurePreservingUnaryPairs(U, param);
[spLPairs total_num_spLViol] = validStructurePreservingLboundPairs(U, param);
[spUPairs total_num_spUViol] = validStructurePreservingUboundPairs(U, param);


num_cTriplets = size(cTriplets, 1);
% num_pPairs = size(pPairs, 1);
num_pTriplets = size(pTriplets, 1);
% num_sTriplets = size(sTriplets, 1);
% num_sPairs = size(sPairs, 1);
num_spLPairs = size(spLPairs, 1);
num_spUPairs = size(spUPairs, 1);


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


% sErr = 0;
% num_sV = 0;
% if num_sTriplets > 0
%     sErr_vec = param.s_lm + sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)).^2) - sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)).^2);
%     viol = find(sErr_vec > 0);
%     num_sV = length(viol);
%     if viol > 0
%         sErr = sum(sErr_vec(viol));
%     end
% end
% sErr = sErr/size(param.sTriplets, 1);

% spErr = 0;
% num_spV = 0;
% if num_sPairs > 0
%     spErr_vec = param.s_sigma - sum((U(:, sPairs(:, 1)) - U(:, sPairs(:, 2))).^2, 1);
%     viol = find(spErr_vec > 0);
%     num_spV = length(viol);
%     if viol > 0
%         spErr = sum(spErr_vec(viol));
%     end
% end


spLErr = 0;
num_spLV = 0;
if num_spLPairs > 0
    spLErr_vec = param.s_sigma - sum((U(:, spLPairs(:, 1)) - U(:, spLPairs(:, 2))).^2, 1);
    viol = find(spLErr_vec > 0);
    num_spLV = length(viol);
    if viol > 0
        spLErr = sum(spLErr_vec(viol));
    end
end


spUErr = 0;
num_spUV = 0;
if num_spUPairs > 0
    spUErr_vec = sum((U(:, spUPairs(:, 1)) - U(:, spUPairs(:, 2))).^2, 1) - param.s_sigma;
    viol = find(spUErr_vec > 0);
    num_spUV = length(viol);
    if viol > 0
        spUErr = sum(spUErr_vec(viol));
    end
end



bal_c = param.bal_c/(param.bal_c + param.bal_p + param.bal_s);
bal_p = param.bal_p/(param.bal_c + param.bal_p + param.bal_s);
bal_s = param.bal_s/(param.bal_c + param.bal_p + param.bal_s);

cErr = bal_c*cErr;
pErr = bal_p*pErr;
spLErr = bal_s*spLErr;
spUErr = bal_s*spUErr;

loss = bal_c*cErr + bal_p*pErr + bal_s*spLErr + bal_s*spUErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('cV: %d / pV: %d / spLV: %d / spUV: %d / cE: %f / pE: %f / spLE: %f / spUE: %f / normW: %f / normU: %f / ', num_cV, num_pV, num_spLV, num_spUV, cErr, pErr, spLErr, spUErr, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));


