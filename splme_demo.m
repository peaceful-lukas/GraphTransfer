
local_env = 0;

if ~local_env
    cd /v9/code/GraphTransfer
end

addpath 'util'
addpath 'param'
addpath 'ddcrp'
addpath 'splme'
addpath 'pgm'
addpath 'pgm/RRWM'
addpath 'transfer'

dataset = 'pascal3d_pascal';
method = 'splme';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

[W U] = splme(DS, param, local_env);

% while true
%     [W U] = transfer;
% end