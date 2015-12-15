% test saveState

% set path to sample data
fileName = '../data/net-epoch-sample.mat';
[dagnet, stats] = loadState(fileName);

% set path to where network and stats will be stored
targetFileName = '../data/net-epoch-target.mat';
%% Test saved state

saveState(targetFileName, dagnet, stats);

% check that net is saved as a vanilla matlab struct, 
% rather than a dagnet object
s = load(targetFileName, 'net', 'stats');
assert(isa(s.net, 'struct'));

% check that stats remains unchanged
assert(isequal(stats, s.stats));