function DS = loadDataset(dataset, local)

if local, ds_dir = datasetDirLocal(dataset);
else,     ds_dir = datasetRootDir(dataset); end

% load([ds_dir 'trF.mat']);
% load([ds_dir 'trL.mat']);
% load([ds_dir 'trI.mat']);
% load([ds_dir 'teF.mat']);
% load([ds_dir 'teL.mat']);
% load([ds_dir 'teI.mat']);
% 
% DS = {};
% DS.D = trF;
% DS.T = teF;
% 
% DS.DL = trL;
% DS.TL = teL;



DS = {};

data_fname = [ds_dir 'trF.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.D = trF;
end

data_fname = [ds_dir 'trL.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.DL = trL;
end

data_fname = [ds_dir 'teF.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.T = teF;
end

data_fname = [ds_dir 'teL.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.TL = teL;
end

data_fname = [ds_dir 'trI.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.DI = trI;
end

data_fname = [ds_dir 'teI.mat'];
if exist(data_fname) == 2
    load(data_fname);
    DS.TI = teI;
end





function ds_dir = datasetRootDir(dataset)

ds_dir = '';

if     strcmp(dataset, 'awa'),                  ds_dir = '/v9/AwA/allclass/proc/';
elseif strcmp(dataset, 'pascal3d_pascal'),      ds_dir = '/v9/PASCAL3D/pascal/proc/';
elseif strcmp(dataset, 'voc'),                  ds_dir = '/v9/voc/proc/';
elseif strcmp(dataset, 'AwA_official'),         ds_dir = '/v9/AwA/official/proc/';
elseif strcmp(dataset, 'AwA_30_only'),          ds_dir = '/v9/AwA/allclass_30_only/proc/';

else
    fprintf('[loadDataset] no such a dataset.\nDataset dir has been set to be empty.\n');
end


function ds_dir_local = datasetDirLocal(dataset)

ds_dir_local = '';

if     strcmp(dataset, 'awa'),                  ds_dir_local = '/Users/lukas/Desktop/awa/';
elseif strcmp(dataset, 'pascal3d_pascal'),      ds_dir_local = '/Users/lukas/Desktop/pascal3d_pascal/';
elseif strcmp(dataset, 'voc'),                  ds_dir_local = '/Users/lukas/Desktop/voc/';
elseif strcmp(dataset, 'AwA_30_only'),          ds_dir_local = '/Users/lukas/Desktop/AwA_30_only/';
end