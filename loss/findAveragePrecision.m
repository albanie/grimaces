function averagePrecision = findAveragePrecision(X, c)
% averagePrecision = findAveragePrecision(X, c)
% It does NOT use interpolation.

%   The prediction scores X are organised as a field of prediction
%   vectors, represented by a H x W x D x N array. The first two
%   dimensions, H and W, are spatial and correspond to the height and
%   width of the field; the third dimension D is the number of
%   categories or classes; finally, the dimension N is the number of
%   data items (images) packed in the array.

%   c contains the ground truth class labels.

% add path to the vlfeat toolbox functions
addpath '../../vlfeat/toolbox/plotop/';
addpath '../../vlfeat/toolbox/misc/';

% transform weights into confidence scalars
confidences1 = squeeze(X(:,:,1,:))';
confidences2 = squeeze(X(:,:,2,:))';
scores = confidences2 - confidences1;

% Convert our labels from {1,2} to {-1,1} to work with 
% the standard terminology
labels = zeros(size(c));
labels(c==1) = -1;
labels(c==2) = 1;

[~,~,info] = vl_pr(labels, scores, 'interpolate', false) ;

% We return the averagePrecision as a vector of replicated values
% rather than a scalar so it jives with vl_nnloss
averagePrecision = info.ap * ones(size(labels));

end