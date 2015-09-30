function MatchingScores = matchingScores(U, param)

MatchingScores = zeros(param.numClasses, param.numClasses);
for i=1:param.numClasses
    for j=i+1:param.numClasses
        c1 = i;
        c2 = j;


        numProto_c1 = param.numPrototypes(c1);
        numProto_c2 = param.numPrototypes(c2);
        protoStartIdx = [0; cumsum(param.numPrototypes)];
        U_c1 = U(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
        U_c2 = U(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

        simMatrix = U_c1'*U_c2;
        sim_scores = sort(simMatrix(:), 'descend');

        param_gm.maxIterGM = 10;
        param_gm.match_thrsh = sim_scores(min(numProto_c1, numProto_c2));
        param_gm.match_sim_thrsh = sim_scores(max(numProto_c1, numProto_c2));
        param_gm.knn1 = 3;
        param_gm.knn2 = 4;
        param_gm.voting_alpha = 10;


        [X_sol candidate_matches score_GM] = progGM(U_c1, U_c2, param_gm);
        MatchingScores(i, j) = score_GM;
    end
end