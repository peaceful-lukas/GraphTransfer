function nnGraphs = constructNNGraphs(P, param)

numClasses = param.numClasses;
numPrototypes = param.numPrototypes;

% generate epsilon nearest neighbor graph
proto_offset = [0; cumsum(param.numPrototypes)];
nnGraphs = {};

for classNum=1:numClasses
    nnGraphs{classNum} = zeros(numPrototypes(classNum), numPrototypes(classNum));
    P_c = P(:, proto_offset(classNum)+1:proto_offset(classNum+1));
    
    D_c = pdist(P_c');
    D_c = squareform(D_c);
    D_c(find(eye(size(D_c)))) = Inf;

    edge_dist_vec = sort(D_c(:), 'ascend');
    eps = edge_dist_vec(4*numPrototypes(classNum));

    A_c = D_c;
    A_c(find(A_c > eps)) = 0;
    A_c(find(A_c)) = 1;

    no_neighbors = find(all(A_c == 0, 1));
    [min_D_c, min_idx] = min(D_c, [], 2);
    for n=no_neighbors
        A_c(n, min_idx(n)) = 1;
        A_c(min_idx(n), n) = 1;
    end
    
    nnGraphs{classNum} = A_c;
end



