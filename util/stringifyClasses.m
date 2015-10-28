function clsnames = stringifyClasses(dataset)

clsnames = {};
if strcmp(dataset, 'awa') || strcmp(dataset, 'AwA_30_only') || strcmp(dataset, 'awa50_pca500')
    clsnames = {'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', ...
                'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', ...
                'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', ...
                'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', ...
                'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', ...
                'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', ...
                'wolf', 'chihuahua', 'rat', 'weasel', 'otter', ...
                'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', ...
                'pig', 'lion', 'mouse', 'polar+bear', 'collie',  ...
                'walrus', 'raccoon', 'cow', 'dolphin'};

elseif strcmp(dataset, 'voc') || strcmp(dataset, 'voc_pca500')
    clsnames = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', ...
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

elseif strcmp(dataset, 'voc_high') 
    clsnames = {'bicycle', 'motorbike', 'bus', 'train'};

elseif strcmp(dataset, 'pascal3d_pascal') || strcmp(dataset, 'pascal3d_imagenet') || strcmp(dataset, 'pascal3d_all')
    clsnames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
else
    fprintf('\nno class name list on %s\n\n', dataset);
end
