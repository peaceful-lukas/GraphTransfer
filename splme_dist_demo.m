
local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'ddcrp/'
addpath 'splme_dist/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'transfer_splme_new/'
addpath 'transfer_splme_new/local_lme/'
addpath 'transfer_splme_new/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


dataset = 'voc';
% dataset = 'awa';
method = 'splme_dist';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme_dist(DS, param, local_env);
splme_dist;



