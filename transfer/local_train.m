function [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses)

% regenerate structure-preserving triplets
fprintf('Local Training LME ... \n');
param_new.sTriplets = generateLocalStructurePreservingTriplets(param_new);

% locally learn prototypes (U)
% W_retrained = local_learnW(DS, W, U_new, param_new, trainTargetClasses);
U_retrained = local_learnU(DS, W, U_new, param_new, trainTargetClasses);

fprintf('Local LME RESULT\n');
fprintf('\tbefore\n');
[~, accuracy] = dispAccuracy(param_new.method, DS, W, U_new, param_new);
fprintf('\tafter\n');
[~, accuracy] = dispAccuracy(param_new.method, DS, W, U_retrained, param_new);
fprintf('\n\n\n\n');

