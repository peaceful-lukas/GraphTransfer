function [sPairs total_num_sPairs] = sampleStructurePreservingUnaryPairs(U, param)
% (k, k')


sPairs = [];

proto_offset = cumsum([0; param.numPrototypes]);

for classNum=1:param.numClasses
    A_c = param.nnGraphs{classNum};
    [k_vec, k_prime_vec] = find(A_c);

    k_vec = k_vec + proto_offset(classNum);
    k_prime_vec = k_prime_vec + proto_offset(classNum);

    sPairs = [sPairs; k_vec, k_prime_vec];
end

total_num_sPairs = size(sPairs, 1);

spErr_vec =  param.s_sigma - sum((U(:, sPairs(:, 1)) - U(:, sPairs(:, 2))).^2, 1);
viol_vec = find(spErr_vec > 0);

sPairs = sPairs(viol_vec, :);