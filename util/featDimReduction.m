function [DS param] = featDimReduction(DS, param, reducedDim)

% train dataset
X = DS.D;
[~, pca_score, latent] = pca(X');
reduced_X = pca_score(:, 1:reducedDim)';
DS.D = reduced_X;

X = DS.T;
[~, pca_score, latent] = pca(X');
reduced_X = pca_score(:, 1:reducedDim)';
DS.T = reduced_X;


param.featureDim = reducedDim;
