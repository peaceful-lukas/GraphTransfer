% LOCAL
param.numClasses = 50;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 100;
param.m_batchSize = 100;
param.p_batchSize = 100;
param.lowDim = 50;
param.featureDim = 9216;

param.knn_const = 3; % constant for constructing k-nn graph.

param.m_lm = 5; % large margin for classification
param.p_sigma = 0.5; % 1 % distance to the prototypes
param.bal_m = 1;
param.bal_p = 1;

param.lambda_W = 0.01; % regularizer coefficient
param.lambda_U = 0.001; % regularizer coefficient

param.lr_W = 0.01; % learning rate for W
param.lr_U = 0.01; % learning rate for U
