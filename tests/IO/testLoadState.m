% test loadState

% set path to sample data
fileName = '../data/net-epoch-sample.mat';

% preconditions
assert(exist(fileName, 'file') == 2);

%% Test loaded state
[dagnet, stats] = loadState(fileName);

% check that net is a DagNN object
assert(isa(dagnet, 'dagnn.DagNN'));

% check that stats is a struct with correct fields
assert(isa(stats, 'struct'));
assert(isequal(fieldnames(stats), {'train';'val'}));