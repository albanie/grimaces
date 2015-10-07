function [net, info] = cnn_faces(varargin)

% setup the MatConvNet toolbox and add utils
vl_setupnn;
addpath('../examples');
addpath('utils');

% set destination path for results
opts.expDir = 'results';
[opts, varargin] = vl_argparse(opts, varargin);

% set path to the raw input data
opts.dataDir = 'data';

% set path to where preprocessed data will be stored
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

% set training parameters for the CNN
opts.useBnorm = false;
opts.fineTuningRate = 0.05;
opts.train.batchSize = 100;
opts.train.numEpochs = 20;
opts.train.continue = false;
opts.train.gpus = [];
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir;
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

% Initialize a CNN using Alexnet
net = cnn_alexnet_init(opts);
         
%if the imdb file has already been created, load into memory.
% Otherwise, build it from scratch.
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    imdb = getFacesImdb(opts, net);
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end

% --------------------------------------------------------------------
%                                                     Evaluate network
% --------------------------------------------------------------------
trainIdx = find(strcmp(imdb.images.set, 'train'));
valIdx = find(strcmp(imdb.images.set, 'val'));
testIdx = find(strcmp(imdb.images.set, 'test'));

info = cnn_train_face_dag(net, imdb, @getBatch, opts.train, ...
                    'train', trainIdx, ...
                    'val',  valIdx);
end

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);
inputs = { 'input', input, ...
           'label', label};
end

