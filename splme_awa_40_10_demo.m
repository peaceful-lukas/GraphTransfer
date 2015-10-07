
local_env = 1;

% if ~local_env
%     cd /v9/code/GraphTransfer
% end

addpath 'util/'
addpath 'param/'
addpath 'ddcrp/'
addpath 'splme/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'transfer/'
addpath 'transfer/local_lme/'
addpath 'transfer/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


dataset = 'AwA_official';
method = 'splme';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% splme;

