function scores = perClassScore(DS, W, U, param)

fprintf('computing per class scores...\n');

scores = zeros(param.numClasses, 1);
X = DS.D;
classList = (1:param.numClasses)';

for cls=1:param.numClasses
    fprintf('.');
    if mod(cls, 10) == 0, fprintf('\t'); end

    cls_idx = find(DS.DL == cls);

    cTriplets = zeros(length(cls_idx)*(param.numClasses-1), 3);
    rep_ivec = repmat(cls_idx, 1, param.numClasses-1)';
    cTriplets(:, 1) = rep_ivec(:);
    cTriplets(:, 2) = cls;
    rep_cvec = classList;
    rep_cvec(cls) = [];
    cTriplets(:, 3) = repmat(rep_cvec, length(cls_idx), 1);
    
    perClassLoss = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
    valids = find(perClassLoss > 0);
    perClassLoss = sum(perClassLoss(valids));
    
    scores(cls) = perClassLoss/sqrt(length(cls_idx));
end

