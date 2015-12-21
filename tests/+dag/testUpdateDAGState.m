% test updateDAGState

% setup paths and code
% addpath('../../dag');
% addpath('../../../matlab');
% vl_setupnn;

% set up opts structure
opts.expDir = '../data/savedEpochs';
list = fullfile(opts.expDir, 'net-epoch-*.mat');
fileList = dir(list);
fileList.name
expectedEpoch = 361;

%% Test last checkpoint correctly calculated
epoch = findLastCheckpoint(opts);

% check that the checkpoint takes the expected value
assert(isequal(expectedEpoch, epoch));
