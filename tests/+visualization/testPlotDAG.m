% test plotDAG
% For the graphical tests to pass, plotted graphs can be checked
% visually.

% set up some example parameters
stats.train(1).prob = 0.55;
stats.train(1).error = 0.45;
stats.val(1).prob = 0.65;
stats.val(1).error = 0.6;

opts.modelFigPath = 'data/testFig';

%% Test plotDAG for the first epoch during training
epoch = 1;
opts.testMode = false;
plotDAG(stats, epoch, opts);

%% Test plotDAG during testing
% create additional stats
epoch = 1;
opts.testMode = true;
plotDAG(stats, epoch, opts);

%% Test plotDAG for the second epoch during training
% create additional stats
stats.train(2).prob = 0.5;
stats.train(2).error = 0.4;
stats.val(2).prob = 0.55;
stats.val(2).error = 0.5;
opts.testMode = false;
epoch = 2;
plotDAG(stats, epoch, opts);



