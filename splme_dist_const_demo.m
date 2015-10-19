
local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'sc/'
addpath 'ddcrp/'
addpath 'clustering/'
addpath 'splme_dist_const/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'transfer_splme_dist/'
addpath 'transfer_splme_dist/local_lme/'
addpath 'transfer_splme_dist/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


dataset = 'voc';
% dataset = 'awa';
method = 'splme_dist_const';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme_dist(DS, param, local_env);
splme_dist_const;




