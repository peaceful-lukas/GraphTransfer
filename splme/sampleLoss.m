function loss = sampleLoss(DS, W, U, param)

X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);
sTriplets = sampleStructurePreservingTriplets(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);
num_sTriplets = size(sTriplets, 1);

cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    cErr_vec = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
    viol = find(cErr_vec > 0);
    num_cV = length(viol);

    if viol > 0
        cErr = sum(cErr_vec(viol));
    end
end

sErr = 0;
num_sV = 0;
if num_sTriplets > 0
    sErr_vec = param.s_lm + diag( U(:, sTriplets(:, 1))' * (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2))) );
    viol = find(sErr_vec > 0);
    num_sV = length(viol);

    if viol > 0
        sErr = sum(sErr_vec(viol));
    end
end

bal_c = param.bal_c/(param.bal_c + param.bal_s);
bal_s = param.bal_s/(param.bal_c + param.bal_s);

cErr = bal_c*cErr;
sErr = bal_s*sErr;

loss = param.bal_c*cErr + param.bal_s*sErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('cV: %d / sV: %d / cE: %f / sE: %f / normW: %f / normU: %f / ', num_cV, num_sV, cErr, sErr, norm(W, 'fro'), norm(U, 'fro'));

