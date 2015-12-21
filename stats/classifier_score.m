function score = classifier_score(net, imdb_test)
% Takes in a struct 'net' which contains the trained parameters
% and a test instance.
% Returns the score produced by the network for this instance.

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('utils');
vl_setupnn;

batchSize = 5 ;
mergeScores = @(x,y) [x, y];

% load a dagnn object
net = dagnn.DagNN.loadobj(net);
net.conserveMemory = false;

testSet = find(imdb_test.images.set == 3);
% testSetSize = size(imdb_test.images.set, 2);
% testSet = [randsample(testSetSize, 100)'];
% testSet = [1:500];

% setup the network evluation
state.getBatch = @getBatch ;
state.imdb = imdb_test;

subsetSize = numel(testSet);
subsetStart = 1;
subsetEnd = subsetSize;
subset = testSet(subsetStart:subsetEnd);

start = tic;
num = 0;

score = [];

for t=1:numel(subset)
    % get image batch
    batchStart = t;
    batchEnd = min(t, numel(subset));
    batch = subset(batchStart : batchEnd);
    num = num + numel(batch);
    if numel(batch) == 0, continue; end
    
    inputs = state.getBatch(state.imdb, batch);
    net.eval(inputs);
    
    % extract stats
    if isempty(score)
        score = extractScores(net) ;
    else
        score = mergeScores(score, extractScores(net)) ;
    end
    
    time = toc(start);
    
    fprintf('batch %3d/%3d: %.1f Hz', ...
    fix(t/batchSize)+1, ceil(numel(subset)/batchSize), ...
    num/time) ;
    
    fprintf('\n');
end

net.reset()
end

% -------------------------------------------------------------------------
function scores = extractScores(net)
% -------------------------------------------------------------------------
lossLayerIdx = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
scores = struct() ;
for i = 1:numel(lossLayerIdx)
  lossLayer = net.layers(lossLayerIdx(i));
  scores.(lossLayer.name) = lossLayer.block.average;
end

scoreLayerIdx = find(cellfun(@ (x) (strcmp(x,'fc8')), {net.layers.name}));
scoreLayer = net.layers(scoreLayerIdx);
scoreOutputIdx = scoreLayer.outputIndexes;
flat_weight = net.vars(scoreOutputIdx).value(:,:,1);
steep_weight = net.vars(scoreOutputIdx).value(:,:,2);
scores.confidence = flat_weight - steep_weight;

end

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);
inputs = { 'input', input, ...
           'label', label};
end