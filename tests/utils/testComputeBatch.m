% test computeBatch

% set up some example parameters
subset = 1:1000;
opts.batchSize = 256;
opts.numSubBatches = 1;
subBatchIdx = 1;

%% Test batch indices correctly calculated for the first batch
subsetIdx = 1;
expectedBatch = 1:256;
batch = computeBatch(subsetIdx, subBatchIdx, subset, opts);
assert(isequal(expectedBatch, batch));

%% Test batch indices correctly calculated for the last batch
subsetIdx = 769;
expectedBatch = 769:1000;
batch = computeBatch(subsetIdx, subBatchIdx, subset, opts);
assert(isequal(expectedBatch, batch));

%% Test batch indices correctly calculated when subBatching is used
subsetIdx = 1;
subBatchIdx = 2;
opts.numSubBatches = 2;
expectedBatch = 2:2:256;
batch = computeBatch(subsetIdx, subBatchIdx, subset, opts);
assert(isequal(expectedBatch, batch));