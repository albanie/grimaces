function testResults = testCNN(net, varargin)
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

% if the imdb test doesn't exist, we need to create it from scratch.
opts.dataDir = 'data';

% set threshold for classification
opts.gradientThreshold = 6.5 ;

% load a dagnn object
dagnet = dagnn.DagNN.loadobj(net);


%% DEBUGING MODE
% dagnet.conserveMemory = false;
%END DEBUGGING MODE
%%

% Load training dataset
imdb_test = loadImdb(opts, dagnet, 'test');

% retrieve the test items
testSet = find(imdb_test.images.set == 3);

% set DAG to testMode
dagnet.mode = 'test';

% set testing options
opts.test.gpus = [];
opts.test.testMode = true;
opts.test.expDir = opts.expDir;

% only one epoch is needed.
opts.test.numEpochs = 1;
opts = vl_argparse(opts, varargin);

% finally we can evaluate the network
stats = runDAG(dagnet, ...
    imdb_test, ...
    @getBatch, ...
    opts.test, ...
    'val', ...
    testSet);


% save the test results
testResults = stats.val ;
save(strcat(opts.expDir, '/test-results'), 'testResults') ;
end
