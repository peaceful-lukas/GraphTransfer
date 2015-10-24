function loss = sampleLoss(DS, W, U, param)

X = DS.D;

cTriplets = sampleClassificationTriplets(DS, W, U, param);
mTriplets = sampleMembershipTriplets(DS, W, U, param);
[sTriplets total_num_sTriplets] = sampleStructurePreservingTriplets(U, param);

num_cTriplets = size(cTriplets, 1);
num_mTriplets = size(mTriplets, 1);
num_sTriplets = size(sTriplets, 1);



cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    cErr_vec = param.m_lm + sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2))).^2, 1) - sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3))).^2, 1);
    viol_vec = find(cErr_vec > 0);
    num_cV = length(viol_vec);
    cErr = sum(cErr_vec);
    cErr = cErr/param.c_batchSize;
end


mErr = 0;
num_mV = 0;
if num_mTriplets > 0
    mErr_vec = param.m_lm + sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 2))).^2, 1) - sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 3))).^2, 1);
    viol_vec = find(mErr_vec > 0);
    num_mV = length(viol_vec);
    mErr = sum(mErr_vec);
    mErr = mErr/param.m_batchSize;
end


sErr = 0;
num_sV = 0;
if num_sTriplets > 0
    sErr_vec = 2 * diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
            - diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                                   ...
            + diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

    viol_vec = find(sErr_vec > 0);
    num_sV = length(viol_vec);
    sErr = sum(sErr_vec);
    sErr = sErr/total_num_sTriplets;
end

loss = param.bal_c*cErr + param.bal_m*mErr + param.bal_s*sErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('cV: %d / mV: %d / sV: %d / cE: %f / mE: %f / sE: %f / normW: %f / normU: %f / ', num_cV, num_mV, num_sV, cErr, mErr, sErr, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));


