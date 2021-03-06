% ORIGINAL
param.numClasses = 50;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 100;
param.c_batchSize = 100;
param.p_batchSize = 100;
param.lowDim = 200;
param.featureDim = 9216;

param.knn_const = 3;
param.c_lm = 10000; % large margin for classification
param.p_sigma = 10000; % large margin for structure preserving
param.lambda_W = 0.1; % regularizer coefficient
param.lambda_U = 0.0001; % regularizer coefficient
param.lr_W = 1; % learning rate for W
param.lr_U = 1; % learning rate for U
param.bal_c = 10;
param.bal_p = 1;

param.lambda_W_local = 1;
param.lambda_U_local = 0.1; % or 5
param.lr_U_local = 0.01;
param.lr_W_local = 0.001;
param.bal_c = 1;
param.bal_p = 1;
