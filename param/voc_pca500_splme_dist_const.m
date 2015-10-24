param.numClasses = 20;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 200;
param.c_batchSize = 100;
param.p_batchSize = 100;
% param.s_batchSize = 100;
param.lowDim = 100;
param.featureDim = 500;

param.num_clusters = 10;
% param.knn_const = 3;
% param.epsilon_const = 850; % now automatically set as data-driven parameter
param.c_lm = 50; % large margin for classification
param.p_sigma = 1000;  % pulling term
param.p_lm = 1;
param.s_lm = 5; % large margin for structure preserving
param.s_sigma = 1;

param.lambda_W = 0.1; % regularizer coefficient
param.lambda_U = 0.01; % regularizer coefficient
param.lr_W = 1; % learning rate for W
param.lr_U = 1; % learning rate for U
param.bal_c = 0.0001;
param.bal_p = 0.0001;
param.bal_s = 10;

% param.lambda_W_local = 0.001;
% param.lambda_U_local = 0.00001; % or 5
% param.lr_U_local = 0.1;
% param.lr_W_local = 0.1;