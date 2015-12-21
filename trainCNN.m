function [dagnet, stats] = trainCNN(varargin)

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('IO');
addpath('dag');
addpath('utils');
addpath('stats');
addpath('visualization');
vl_setupnn;

% set default destination path for results
opts.expDir = 'experiments/AlexNet-Binary/train';

% set path to the raw input data
opts.dataDir = 'data';

% set path to where the preprocessed training data will be stored
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

[opts, varargin] = vl_argparse(opts, varargin);

% set the default training parameters for the CNN
opts.useBnorm = false;
opts.fineTuningRate = 0.05;
opts.train.batchSize = 100;
opts.train.numEpochs = 500;
opts.train.continue = false;
opts.train.gpus = [];
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir;
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------
% Initialize a CNN dagnet using (pretrained) Alexnet
dagnet = initAlexnet(opts);

% Load training dataset
imdb = loadImdb(opts, dagnet, 'train');

% --------------------------------------------------------------------
%                                                     Evaluate network
% --------------------------------------------------------------------
trainIdx = find(strcmp(imdb.images.set, 'train'));
valIdx = find(strcmp(imdb.images.set, 'val'));

stats = runDAG(dagnet, ...
    imdb, ...
    @getBatch, ...
    opts.train, ...
    'expDir', opts.expDir, ...
    'train', trainIdx, ...
    'val',  valIdx);
end