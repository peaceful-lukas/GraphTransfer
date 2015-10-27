function retrieve_knn_images(DS, W, U, param, protoIdx, k, visualize, coord_idx)

% X = DS.D;
% proto = U(:, protoIdx);

% WX_norm = normc(W*X);
% proto_norm = proto/norm(proto);
% score_vec = WX_norm'*proto_norm;
% % score_vec = (W*X)'*proto;

% [~, sorted_idx] = sort(score_vec, 'descend');

% knn_idx = sorted_idx(1:k);

% if visualize
%     visualizePrototypes(U, param, coord_idx, protoIdx);
% end

% fig = figure;
% set(fig, 'Position', [0, 700, 1300, 200]);    
% for i=1:k
%     subplot(1, k, i);
%     % resized = imresize(DS.DI{knn_idx(i)}, [100, 100]);
%     imagesc(DS.DI{knn_idx(i)});
%     axis image;
%     axis off;
% end




% X = DS.D;
% WX_norm = normc(W*X);
% proto_norm = normc(U);
% scoreMatrix = exp(100*WX_norm'*proto_norm);
% score_vec = scoreMatrix(:, protoIdx);
% softmax_score_vec = score_vec/sum(score_vec);

% [~, sorted_idx] = sort(softmax_score_vec, 'descend');

% knn_idx = sorted_idx(1:k);
% disp(softmax_score_vec(sorted_idx(1:k)));

% if visualize
%     visualizePrototypes(U, param, coord_idx, protoIdx);
% end

% fig = figure;
% set(fig, 'Position', [0, 700, 1300, 200]);    
% for i=1:k
%     subplot(1, k, i);
%     % resized = imresize(DS.DI{knn_idx(i)}, [100, 100]);
%     imagesc(DS.DI{knn_idx(i)});
%     axis image;
%     axis off;
% end


protoStartIdx = [0; cumsum(param.numPrototypes)];
classNum = length(find(protoIdx > protoStartIdx));
targetExamples_idx = find(DS.DL == classNum);

X = DS.D(:, targetExamples_idx);
I = DS.DI(targetExamples_idx);

WX_norm = normc(W*X);
proto_norm = normc(U);
scoreMatrix = 1-WX_norm'*proto_norm;
scoreMatrix = exp(10*scoreMatrix);
score_vec = scoreMatrix(:, protoIdx);
softmax_score_vec = score_vec/sum(score_vec);

[~, sorted_idx] = sort(softmax_score_vec, 'descend');

knn_idx = sorted_idx(1:k);
disp(softmax_score_vec(sorted_idx(1:k)));

if visualize
    visualizePrototypes(U, param, coord_idx, protoIdx);
end

fig = figure;
set(fig, 'Position', [0, 700, 1300, 200]);    
for i=1:k
    subplot(1, k, i);
    % resized = imresize(DS.DI{knn_idx(i)}, [100, 100]);
    imagesc(I{knn_idx(i)});
    axis image;
    axis off;
end