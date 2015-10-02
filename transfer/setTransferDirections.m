function [tPairs, str_tPairs] = setTransferDirections(tPairs, str_tPairs, perClassScores)

tPairs_scores = [perClassScores(tPairs(:, 1)) perClassScores(tPairs(:, 2))];
reverse_idx = find(tPairs_scores(:, 1) > tPairs_scores(:, 2));

% lower scores(lower loss) means more accurate to be transfer sources
tPairs(reverse_idx, :) = [tPairs(reverse_idx, 2) tPairs(reverse_idx, 1)];
str_tPairs(reverse_idx, :) = [str_tPairs(reverse_idx, 2) str_tPairs(reverse_idx, 1)];