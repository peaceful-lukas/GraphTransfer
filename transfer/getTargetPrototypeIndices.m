function targetProtoIdx = getTargetPrototypeIndices(class_idx, param)

targetProtoIdx = [];
protoStartIdx = [0; cumsum(param.numPrototypes)];
targetProtoMat = protoStartIdx([class_idx class_idx+1]);
targetProtoMat(:, 1) = targetProtoMat(:, 1) + 1;
for i=1:length(class_idx)
    targetProtoIdx = [targetProtoIdx; (targetProtoMat(i, 1):targetProtoMat(i, 2))'];
end