function [dagnet, stats] = trainCNN(opts)

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('IO');
addpath('dag');
addpath('utils');
addpath('stats');
addpath('loss');
addpath('visualization');
vl_setupnn;

% NOTE 
% All parameters are set in the experimental config file.

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------
% Initialize a CNN dagnet using (pretrained) network
dagnet = initPretrainedNet(opts);

% retain prediction layer to calculate AP
dagnet.vars(dagnet.getVarIndex('prediction')).precious = true;

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