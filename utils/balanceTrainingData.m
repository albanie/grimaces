function balancedTrainIdx = balanceTrainingData(state, opts)
% returns a set of indices 'train' which have been sampled to 
% to balance out the training classes.

% Put the class labels in variables to keep things easy to read
class1 = 1; 
class2 = 2;

% Find the indices of each class.
class1Idx = find(state.imdb.images.labels(opts.train) == class1);
class2Idx = find(state.imdb.images.labels(opts.train) == class2);

% Find the number of instances in each class.
sizeClass1 = size(class1Idx, 2);
sizeClass2 = size(class2Idx, 2);

% find the number of instances in the smaller class
if sizeClass1 > sizeClass2
    smallerClassIdx = class2Idx;
    largerClassSize = sizeClass1;
    smallerClassSize = sizeClass2;
elseif sizeClass1 < sizeClass2
    smallerClassIdx = class1Idx;
    largerClassSize = sizeClass2;
    smallerClassSize = sizeClass1;
else
    % classes are balanced, no resampling required, 
    % just shuffle the training idx.
    balancedTrainIdx = opts.train(randperm(numel(opts.train)));
    return
end

% randomly sample from the smaller class 
classSizeDiff = largerClassSize - smallerClassSize;
sampleIdx = randsample(smallerClassIdx, classSizeDiff);

% these indices are then repeated in the final training Idx
balancedTrainIdx = horzcat(opts.train, sampleIdx);

% and finally, we shuffle
balancedTrainIdx = balancedTrainIdx(randperm(numel(balancedTrainIdx)));

end
