function [dagnet, stats] = trainCNN(varargin)

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('IO');
addpath('dag');
addpath('utils');
addpath('stats');
addpath('loss');
addpath('visualization');
vl_setupnn;

% set pretrained network of choice
opts.pretrainedNet = 'Alexnet';

% set default destination path for results
opts.expDir = strcat('experiments/', opts.pretrainedNet,'-Binary/train');

% set path to the raw input data
opts.dataDir = 'data/raw';

% set path to where the preprocessed training data will be stored
opts.imdbPath = fullfile('data', 'imdb.mat');

% set threshold for classification
opts.gradientThreshold = 6.5 ;

[opts, varargin] = vl_argparse(opts, varargin);

% set the default training parameters for the CNN
opts.useBnorm = false;
opts.train.batchSize = 100;
opts.train.numEpochs = 500;
opts.train.continue = false;
opts.train.gpus = [];
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir;
opts = vl_argparse(opts, varargin);

% save the experiment parameters 
saveExperimentParams(opts, 'train');

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------
% Initialize a CNN dagnet using (pretrained) network
dagnet = initPretrainedNet(opts);


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