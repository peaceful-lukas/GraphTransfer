function U_test_new = officialLocalTrain(DS, W, U_test_new, param_test_new)

fprintf('Local Training LME ... \n');
U_test_new = officialLearnU(DS, W, U_test_new, param_test_new, true);