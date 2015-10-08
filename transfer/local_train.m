function [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses)

% regenerate structure-preserving triplets
fprintf('Local Training LME ... \n');
param_new.sTriplets = generateLocalStructurePreservingTriplets(param_new);

% locally learn prototypes (U)
targetProtoIdx = [];
protoStartIdx = [0; cumsum(param_new.numPrototypes)];
targetProtoMat = protoStartIdx([trainTargetClasses trainTargetClasses+1]);
targetProtoMat(:, 1) = targetProtoMat(:, 1) + 1;
for i=1:length(trainTargetClasses)
    targetProtoIdx = [targetProtoIdx; (targetProtoMat(i, 1):targetProtoMat(i, 2))'];
end

% W_retrained = local_learnW(DS, W, U_new, param_new, trainTargetClasses);
U_retrained = local_learnU(DS, W, U_new, param_new, trainTargetClasses, targetProtoIdx);

fprintf('Local LME RESULT\n');
fprintf('\tbefore\n');
[~, accuracy] = dispAccuracy(param_new.method, DS, W, U_new, param_new);
fprintf('\tafter\n');
[~, accuracy] = dispAccuracy(param_new.method, DS, W, U_retrained, param_new);
fprintf('\n\n\n\n');