param.numClasses = 20;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 50;
param.c_batchSize = 100;
param.p_batchSize = 100;
param.lowDim = 30;
param.featureDim = 9216;

param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 10000; % large margin for classification
param.p_sigma = 10000; % large margin for structure preserving
param.lambda_W = 0.001; % regularizer coefficient
param.lambda_U = 0.00001; % regularizer coefficient
param.lr_W = 0.1; % learning rate for W
param.lr_U = 0.1; % learning rate for U
param.bal_c = 10;
param.bal_p = 1;


param.lambda_W_local = 10;
param.lambda_U_local = 1; % or 5
param.lr_U_local = 0.001;
param.lr_W_local = 0.001;