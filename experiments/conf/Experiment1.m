function Experiment1(mode, gpus)
% Experiment configuration file.  
% This serves to document the parameters used in the experiment.

addpath 'utils'

% set shared options
opts.dataDir = 'data/raw';
opts.gradientThreshold = 6.5 ;

% --------------------------------------------------------------------
%                                                   Training the model
% --------------------------------------------------------------------

if strcmp(mode, 'train')    
    opts.pretrainedNet = 'Alexnet';
    opts.expDir = strcat('experiments/results/', mfilename ,'/train');
    opts.imdbPath = fullfile('data', 'imdb.mat');
    opts.useBnorm = false;
    opts.train.batchSize = 100;
    opts.train.numEpochs = 500;
    opts.train.continue = false;
    opts.train.gpus = gpus;
    opts.train.learningRate = 0.001;
    opts.train.expDir = opts.expDir;
    trainCNN(opts)
end

% --------------------------------------------------------------------
%                                                    Testing the model
% --------------------------------------------------------------------

if strcmp(mode, 'test')
    opts.test.testMode = true;    
    opts.test.batchSize = 100;
    opts.test.numEpochs = 1;
    opts.test.gpus = gpus;
    rootExpPath = strcat('experiments/results/', mfilename);
    opts.imdbPath = fullfile('data', 'imdb_test.mat');
    
    % Load the network from the final epoch of training
    finalEpoch = findLastCheckpoint(strcat(rootExpPath, '/train'));
    data = load(fullfile(rootExpPath, '/train/', ...
                    strcat('net-epoch-', num2str(finalEpoch), '.mat')));
    finalNet = data.net;
    opts.expDir = strcat(rootExpPath, '/test');
    opts.test.expDir = opts.expDir;
    testCNN(finalNet, opts)
end