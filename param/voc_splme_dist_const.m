% param.numClasses = 20;
% param.maxIterW = 1000;
% param.maxIterU = 1000;
% param.maxAlter = 50;
% param.c_batchSize = 100;
% param.p_batchSize = 100;
% param.lowDim = 100;
% param.featureDim = 9216;

% param.num_clusters = 10;
% param.knn_const = 3; % constant for constructing k-nn graph.
% param.c_lm = 10000; % large margin for classification
% param.p_sigma = 10000; % large margin for structure preserving
% % param.lambda_W = 0.001; % regularizer coefficient
% % param.lambda_U = 0.00001; % regularizer coefficient
% param.lambda_W = 1; % regularizer coefficient
% param.lambda_U = 1; % regularizer coefficient
% param.lr_W = 1; % learning rate for W
% param.lr_U = 1; % learning rate for U
% param.bal_c = 10;
% param.bal_p = 1;


% param.lambda_W_local = 0.001;
% param.lambda_U_local = 0.00001; % or 5
% param.lr_U_local = 0.1;
% % param.lr_W_local = 0.1;

param.numClasses = 20;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 50;
param.c_batchSize = 100;
param.p_batchSize = 100;
param.s_batchSize = 100;
param.lowDim = 100;
param.featureDim = 9216;

param.num_clusters = 10;
param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 10; % large margin for classification
param.p_sigma = 0.1; % large margin for structure preserving
param.s_lm = 1;

% param.lambda_W = 100; % degeneracy occurred
param.lambda_W = 1; % regularizer coefficient
param.lambda_U = 0.1; % regularizer coefficient
param.lr_W = 1; % learning rate for W
param.lr_U = 10; % learning rate for U
param.bal_c = 5;
param.bal_p = 1;
param.bal_s = 5;


param.lambda_W_local = 0.001;
param.lambda_U_local = 0.00001; % or 5
param.lr_U_local = 0.1;
% param.lr_W_local = 0.1;