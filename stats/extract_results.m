function test_scores = extract_results(numEpochs)
% Takes in a struct 'net' which contains the trained parameters
% and  uses it to make predictions on the test set defined in imdb.

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('utils');
vl_setupnn;

% set path to the expected imdb file
opts.expDir = '/Volumes/sam_backup/Tour_project/results/AlexNet-Binary4';

test_scores = struct();

for epoch = 1:numEpochs
        data = strcat(opts.expDir, '/net-epoch-', num2str(epoch), '.mat');
        vars = load(data);
        [results, pred_stats] = cnn_predict(vars.net)
        test_scores(epoch).epochNum = epoch
        test_scores(epoch).results = results
        test_scores(epoch).pred_stats = pred_stats
end

end
