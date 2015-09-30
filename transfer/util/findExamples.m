function result = findExamples(exampleType, classNum, DS, W, U, param, W_new, U_new, param_new)

result = [];

switch(exampleType)
case 'gotTrueAfterTransfer'
    result = gotTrueAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new);

case 'gotFalseAfterTransfer'
    
end




function gotTrue = gotTrueAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new)

classIdx = find(DS.DL == classNum);

% before transfer
cumNumProto = cumsum(param.numPrototypes);
[~, predicted_raw] = max(DS.D'*W'*U, [], 2);
predicted = zeros(numel(predicted_raw), 1);
for c = 1:param.numClasses
    t = find(predicted_raw <= cumNumProto(c));
    predicted(t) = c;
    predicted_raw(t) = Inf;
end
wrong_predicts = find(DS.DL(classIdx) ~= predicted(classIdx));


% after transfer
cumNumProto_new = cumsum(param_new.numPrototypes);
[~, predicted_raw] = max(DS.D'*W_new'*U_new, [], 2);
predicted_new = zeros(numel(predicted_raw), 1);
for c = 1:param_new.numClasses
    t = find(predicted_raw <= cumNumProto_new(c));
    predicted_new(t) = c;
    predicted_raw(t) = Inf;
end
true_predicts = find(DS.DL(classIdx) == predicted_new(classIdx));

gotTrue = intersect(true_predicts, wrong_predicts);
gotTrue = classIdx(gotTrue);


function gotFalse = gotFalseAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new)

gotFalse = [];




