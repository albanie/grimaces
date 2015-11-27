function [results,prediction_stats] = cnn_predict(net)
% Takes in a struct 'net' which contains the trained parameters
% and  uses it to make predictions on the test set defined in imdb. 

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('utils');
vl_setupnn;

% set path to the expected imdb file
opts.expDir = 'AlexNet-Binary4';
opts.imdbTestPath = fullfile(opts.expDir, 'imdb_test.mat');

% load a dagnn object
net = dagnn.DagNN.loadobj(net);
         
% if the imdb_test file has already been created, load into memory.
% Otherwise, build it from scratch.
if exist(opts.imdbTestPath, 'file')
    imdb_test = load(opts.imdbTestPath);
else
    imdb_test = getTestFacesImdb(opts, net);
    mkdir(opts.expDir);
    save(opts.imdbTestPath, '-struct', 'imdb_test');
end

% retrieve the test items 
testSet = find(imdb_test.images.set == 3);

% finally we can evaluate the network
stats = cnn_run_dag(net, imdb_test, @getBatch, 'testSet', testSet);

prediction_stats = stats;

results = {};
results.loss = mean([stats.loss]);
results.error = mean([stats.error]);
fprintf('Error across the test set %.2f', results.error)
end

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);
inputs = { 'input', input, ...
           'label', label};
end

