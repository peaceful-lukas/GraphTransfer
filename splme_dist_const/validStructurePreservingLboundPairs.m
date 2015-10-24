function [spLPairs total_num_spLViol] = validStructurePreservingLboundPairs(U, param)

spLPairs = [];
proto_offset = [0; cumsum(param.numPrototypes)];

for classNum=1:param.numClasses
    nnGraph = param.epsilonGraphs{classNum};
    [k_vec, l_vec] = find(triu(~nnGraph - eye(size(nnGraph))));
    k_vec = k_vec + proto_offset(classNum);
    l_vec = l_vec + proto_offset(classNum);

    loss_vec = param.s_sigma - sum((U(:, k_vec) - U(:, l_vec)).^2, 1);
    valids = find(loss_vec > 0);
    spLPairs = [spLPairs; k_vec, l_vec];
end

total_num_spLViol = size(spLPairs, 1);