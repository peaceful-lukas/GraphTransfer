function [spUPairs total_num_spUViol] = validStructurePreservingUboundPairs(U, param)

spUPairs = [];
proto_offset = [0; cumsum(param.numPrototypes)];

for classNum=1:param.numClasses
    nnGraph = param.epsilonGraphs{classNum};
    [k_vec, k_prime_vec] = find(triu(nnGraph));
    k_vec = k_vec + proto_offset(classNum);
    k_prime_vec = k_prime_vec + proto_offset(classNum);

    loss_vec = sum((U(:, k_vec) - U(:, k_prime_vec)).^2, 1) - param.s_sigma;
    valids = find(loss_vec > 0);
    spUPairs = [spUPairs; k_vec, k_prime_vec];
end

total_num_spUViol = size(spUPairs, 1);