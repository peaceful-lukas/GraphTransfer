
local_env = 1;

% if ~local_env
%     cd /v9/code/GraphTransfer
% end

addpath 'util'
addpath 'param'
addpath 'ddcrp'
addpath 'splme'
addpath 'pgm'
addpath 'pgm/RRWM'
addpath 'transfer'
addpath 'transfer/local_lme'
% addpath(genpath(pwd));

% dataset = 'pascal3d_pascal';
dataset = 'awa';
method = 'splme';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme(DS, param, local_env);
splme

% transfer