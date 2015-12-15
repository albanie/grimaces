%% Run all tests

% setup paths and code
addpath(GetFullPath('../IO'));
addpath(GetFullPath('../utils'));
addpath(GetFullPath('../../matlab'));
vl_setupnn;

runtests('IO/testLoadState.m');
runtests('IO/testSaveState.m');

runtests('utils/testFindLastCheckpoint.m');
runtests('utils/TestGetLearningRate.m');
runtests('utils/testComputeBatch.m');