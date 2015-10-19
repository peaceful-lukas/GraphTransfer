
local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'ddcrp/'
addpath 'sc/'
addpath 'clustering/'
addpath 'splme_sim_const/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'transfer_splme_new/'
addpath 'transfer_splme_new/local_lme/'
addpath 'transfer_splme_new/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


dataset = 'voc';
% dataset = 'awa';
method = 'splme_sim_const';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme_sim_const(DS, param, local_env);
splme_sim_const;



