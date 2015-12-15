function imdb = loadImdb(opts, dagnet, mode)
% Returns imdb (the image database). If the imdb file has already been 
% created previously, load into memory. Otherwise, build it from scratch.

% In 'train' mode, the training and validation imdb is loaded.
% In 'test' mode, the test imdb is loaded.

% Define the minimum face size that will be accepted by the network
opts.minFaceArea = 3600;

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    mkdir(opts.expDir);
    if strcmp(mode, 'train')
        imdb = getFacesImdb(opts, dagnet);
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
    else % mode == 'test'
        imdb = getTestFacesImdb(opts, dagnet);
        save(opts.imdbPath, '-struct', 'imdb_test', '-v7.3');
    end
end
   
end

% --------------------------------------------------------------------
function imdb = getFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

% get faces and labels
[trainData, trainLabels] = getFaceData(opts, dagnet, 'training');
[valData, valLabels] = getFaceData(opts, dagnet, 'validation');

% get the set labels for the training data
trainSet = ones(1, size(trainLabels,2));
train_label = 1;
[trainSet(:)] = deal(train_label);

% get the set labels for the validation data
valSet = 2 * ones(1, size(valLabels,2));
val_label = 2;
[valSet(:)] = deal(val_label);

% create the imdb training/validation structure expected by matconvnet
train_val_data = cat(4, trainData, valData);
train_val_labels = cat(2, trainLabels, valLabels); 
train_val_set = cat(2, trainSet, valSet);

imdb.images.data = train_val_data;
imdb.images.labels = train_val_labels;
imdb.images.set = train_val_set;
imdb.meta.sets = {'train', 'val'};
imdb.meta.classes = 1:2;

end

% --------------------------------------------------------------------
function imdb_test = getTestFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

% Define the minimum face size that will be accepted by the network
minFaceArea = 3600;

% get faces and labels
[testData, testLabels] = getFaceData(opts, dagnet, 'testing');

% get the set labels for the test data
testSet = 3 * ones(1, size(testLabels,2));
test_label = 3;
[testSet(:)] = deal(test_label);

imdb_test.images.data = testData;
imdb_test.images.labels = testLabels;
imdb_test.images.set = testSet;
imdb_test.meta.sets = {'test'} ;
imdb_test.meta.classes = 3;
end