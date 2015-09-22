function DS = loadDataset(dataset, local)

if local
    ds_dir = '/Users/lukas/Desktop/pascal3d_pascal/';
    load([ds_dir 'trF.mat']);
    load([ds_dir 'trL.mat']);
    load([ds_dir 'teF.mat']);
    load([ds_dir 'teL.mat']);

% server
else
    ds_dir = datasetRootDir(dataset);
    load([ds_dir 'proc/trF.mat']);
    load([ds_dir 'proc/trL.mat']);
    load([ds_dir 'proc/teF.mat']);
    load([ds_dir 'proc/teL.mat']);
end

DS = {};
DS.D = trF;
DS.DL = trL;
DS.T = teF;
DS.TL = teL;



function ds_dir = datasetRootDir(dataset)

ds_dir = '';

if     strcmp(dataset, 'awa'),                  ds_dir = '';
elseif strcmp(dataset, 'pascal3d_pascal'),      ds_dir = '/v9/PASCAL3D/pascal/';

else
    fprintf('[loadDataset] no such a dataset.\nDataset dir has been set to be empty.\n');
end