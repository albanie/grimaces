% test findLastCheckpoint

% set up opts structure
opts.expDir = '../data/savedEpochs';
expectedEpoch = 361;

%% Test last checkpoint correctly calculated
epoch = findLastCheckpoint(opts);

% check that the checkpoint takes the expected value
assert(isequal(expectedEpoch, epoch));
