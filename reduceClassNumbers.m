% VOC : 'bicycle', 'motorbike', 'bus', 'train'
targetClasses = [2, 14, 6, 19];

targetExampleIdx = find(ismember(DS.DL, targetClasses));
DS.D = DS.D(:, targetExampleIdx);
DS.DL = DS.DL(targetExampleIdx);
DS.DI = DS.DI(targetExampleIdx);
DS.DL(find(DS.DL == 2)) = 1;
DS.DL(find(DS.DL == 14)) = 2;
DS.DL(find(DS.DL == 6)) = 3;
DS.DL(find(DS.DL == 19)) = 4;


targetExampleIdx = find(ismember(DS.TL, targetClasses));
DS.T = DS.T(:, targetExampleIdx);
DS.TL = DS.TL(targetExampleIdx);
DS.TI = DS.TI(targetExampleIdx);
DS.TL(find(DS.TL == 2)) = 1;
DS.TL(find(DS.TL == 14)) = 2;
DS.TL(find(DS.TL == 6)) = 3;
DS.TL(find(DS.TL == 19)) = 4;







% AwA
targetClasses = 1:10;


targetExampleIdx = find(ismember(DS.DL, targetClasses));
DS.D = DS.D(:, targetExampleIdx);
DS.DL = DS.DL(targetExampleIdx);
DS.DI = DS.DI(targetExampleIdx);


targetExampleIdx = find(ismember(DS.TL, targetClasses));
DS.T = DS.T(:, targetExampleIdx);
DS.TL = DS.TL(targetExampleIdx);
DS.TI = DS.TI(targetExampleIdx);


