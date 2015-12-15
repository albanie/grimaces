function stats = testCNN(net)
% Takes in a struct 'net' which contains the trained parameters
% and  uses it to make predictions on the test set defined in imdb.

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('IO');
addpath('dag');
addpath('utils');
addpath('stats');
addpath('visualization');
vl_setupnn;

% set path to the expected imdb test file
opts.expDir = 'experiments/AlexNet-Binary/test';
opts.imdbPath = fullfile(opts.expDir, 'imdb_test.mat');

% load a dagnn object
dagnet = dagnn.DagNN.loadobj(net);


%% DEBUGING MODE
dagnet.conserveMemory = false;

%END DEBUGGING MODE
%%

% Load training dataset
imdb_test = loadImdb(opts, dagnet, 'test');

% retrieve the test items
testSet = find(imdb_test.images.set == 3);

% set DAG to testMode
opts.test.testMode = true;
opts.test.expDir = opts.expDir;
dagnet.mode = 'test';

% finally we can evaluate the network
stats = runDAG(dagnet, ...
    imdb_test, ...
    @getBatch, ...
    opts.test, ...
    'val', ...
    testSet);
end