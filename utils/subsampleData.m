function [balancedData, balancedLabels] = subsampleData(data, labels)
% returns a balanced dataset i.e. one in which both classes are equally
% represented.  This is achieved by sampling from the larger class 
% without replacement until the sample size matches the smaller class.

% Ensure that randomness is repeatable
rng('default')
rng(0)

% Put the class labels in variables to keep things easy to read
class1 = 1; 
class2 = 2;

% Find the number of instances in each class.
sizeClass1 = size(find(labels == class1), 2);
sizeClass2 = size(find(labels == class2), 2);

% find the number of instances in the smaller class
if sizeClass1 >= sizeClass2
    largerClass = class1;
    smallerClass = class2;
    smallerClassSize = sizeClass2;
else
    largerClass = class2;
    smallerClass = class1;
    smallerClassSize = sizeClass1;
end

% randomly sample from the larger class without replacement
largerClassIdx = find(labels == largerClass);
sampleIdx = randsample(largerClassIdx, smallerClassSize);

% and find the index of all members of the smaller class
smallerClassIdx = find(labels == smallerClass);

% Finally, use these to createa set of balanced dataset idx
balancedIdx = horzcat(sampleIdx, smallerClassIdx);

% shuffle the balancedIdx 
balancedIdx = balancedIdx(randperm(length(balancedIdx)));

% and create the balanced data and labels
balancedData = data(:,:,:,balancedIdx);
balancedLabels = labels(:, balancedIdx);
end
