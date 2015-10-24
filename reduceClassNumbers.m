targetClasses = 1:10;


targetExampleIdx = find(ismember(DS.DL, targetClasses));
DS.D = DS.D(:, targetExampleIdx);
DS.DL = DS.DL(targetExampleIdx);
DS.DI = DS.DI(targetExampleIdx);


targetExampleIdx = find(ismember(DS.TL, targetClasses));
DS.T = DS.T(:, targetExampleIdx);
DS.TL = DS.TL(targetExampleIdx);
DS.TI = DS.TI(targetExampleIdx);


