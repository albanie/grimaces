function imdb_test = getTestFacesImdb(opts, net)
% --------------------------------------------------------------------

% Define the minimum face size that will be accepted by the network
minFaceArea = 3600;

% get faces and labels
[testData, testLabels] = getFaceData(opts, minFaceArea, net, 'testing');

% get the set labels for the test data
testSet = cell(1, size(testLabels,2));
[testSet{:}] = deal('test');

imdb_test.images.data = testData;
imdb_test.images.labels = testLabels;
imdb_test.images.set = testSet;
imdb_test.meta.sets = {'test'} ;
imdb_test.meta.classes = 1;
end


