function [sPairs totalNum_sPairs] = validStructurePreservingUnaryPairs(U, param)

sPairs = [];
protoStartIdx = [0; cumsum(param.numPrototypes)];

for classNum=1:param.numClasses
    protoIdx = protoStartIdx(classNum)+1:protoStartIdx(classNum+1);
    sPairs_c = nchoosek(protoIdx, 2);
    sPairs = [sPairs; sPairs_c];
end

totalNum_sPairs = size(sPairs, 1);

loss_vec = param.s_sigma - sum((U(:, sPairs(:, 1)) - U(:, sPairs(:, 2))).^2, 1);
valids = find(loss_vec > 0);
sPairs = sPairs(valids, :);

