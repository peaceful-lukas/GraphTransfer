reducedNumClasses = 5;
reducedClasses= randperm(param.numClasses, reducedNumClasses);

reducedTrainExampleIdx = find(ismember(DS.DL, reducedClasses));
DS.D = DS.D(:, reducedTrainExampleIdx);
DS.DL = DS.DL(reducedTrainExampleIdx);
DS.DI = DS.DI(reducedTrainExampleIdx);

reducedTestExampleIdx = find(ismember(DS.TL, reducedClasses));
DS.T = DS.T(:, reducedTestExampleIdx);
DS.TL = DS.TL(reducedTestExampleIdx);
DS.TI = DS.TI(reducedTestExampleIdx);

param.numClasses = reducedNumClasses;



