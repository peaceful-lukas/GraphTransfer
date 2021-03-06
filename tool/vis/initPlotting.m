function [classNames, colorList] = initPlotting(param)

if strcmp(param.dataset, 'pascal3d_pascal')
    classNames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'dtable', 'motorbike', 'sofa', 'train', 'tv'};

elseif strcmp(param.dataset, 'voc') || strcmp(param.dataset, 'voc_pca500') || strcmp(param.dataset, 'voc_high')
    classNames = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', ...
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

elseif strcmp(param.dataset, 'voc4_pca500') || strcmp(param.dataset, 'voc4_high_pca500')
    classNames = {'bicylce', 'motorbike', 'bus', 'train'};

elseif strcmp(param.dataset, 'awa') || strcmp(param.dataset, 'AwA_30_only') || strcmp(param.dataset, 'awa50') || strcmp(param.dataset, 'awa50_pca500')
    classNames = {'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', ...
                'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', ...
                'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', ...
                'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', ...
                'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', ...
                'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', ...
                'wolf', 'chihuahua', 'rat', 'weasel', 'otter', ...
                'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', ...
                'pig', 'lion', 'mouse', 'polar+bear', 'collie',  ...
                'walrus', 'raccoon', 'cow', 'dolphin'};

elseif strcmp(param.dataset, 'awa10_pca500')
    classNames = {'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', ...
                'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat'};
end

colorList = distinguishable_colors(param.numClasses);

