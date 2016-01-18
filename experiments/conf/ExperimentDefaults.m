function [opts, finalNet] = ExperimentDefaults(mode, experimentName)
% Returns default options used in experiments

addpath 'utils'

% set shared options
opts.dataDir = 'data/raw';
opts.gradientThreshold = 6.5 ;

% --------------------------------------------------------------------
%                                                   Training the model
% --------------------------------------------------------------------

if strcmp(mode, 'train')
    
    opts.imdbPath = fullfile('data', 'imdb.mat');
    opts.useBnorm = true;
    opts.train.batchSize = 100;
    opts.train.numEpochs = 500;
    opts.train.continue = false;
    opts.train.learningRate = 0.001;
    opts.expDir = strcat('experiments/results/', experimentName ,'/train');
    opts.train.expDir = opts.expDir;
    opts.train.expDir = opts.expDir;
end

% --------------------------------------------------------------------
%                                                    Testing the model
% --------------------------------------------------------------------

if strcmp(mode, 'test')
    opts.test.testMode = true;
    opts.test.batchSize = 100;
    opts.test.numEpochs = 1;
    rootExpPath = strcat('experiments/results/', experimentName);
    opts.imdbPath = fullfile('data', 'imdb_test.mat');
    % Load the network from the best epoch of training
    bestEpoch = findBestCheckpoint(strcat(rootExpPath, '/train'));
    data = load(fullfile(rootExpPath, '/train/', ...
        strcat('net-epoch-', num2str(bestEpoch), '.mat')));
    finalNet = data.net;
    opts.test.bestEpoch = bestEpoch;
    opts.expDir = strcat(rootExpPath, '/test');
    opts.test.expDir = opts.expDir;
end