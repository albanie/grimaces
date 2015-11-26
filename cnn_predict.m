function results = cnn_predict(net)
% Takes in a struct 'net' which contains the trained parameters
% and  uses it to make predictions on the test set defined in imdb. 
vl_setupnn ;

% set path to the expected imdb file
opts.imdbPath = fullfile('results', 'imdb.mat');

% load a dagnn object
net = dagnn.DagNN.loadobj(net);
         
% if the imdb file has already been created, load into memory.
% Otherwise, build it from scratch.
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    imdb = getFacesImdb(opts, net);
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end

% retrieve the test items 
testSet = find(strcmp(imdb.images.set, 'test'));

% finally we can evalue the network
stats = cnn_run_dag(net, imdb, @getBatch, 'testSet', testSet);

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

