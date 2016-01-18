function plotDAG(stats, epoch, opts)
% plots the training and evaluation statistics
% of the DAG.

% add path to the vlfeat toolbox functions
addpath '../../vlfeat/toolbox/plotop/';
addpath '../../vlfeat/toolbox/misc/';

% In test mode, we just we the precision recall curve
if opts.testMode
    labels = stats.val.APstored.labels;
    predictions = stats.val.APstored.predictions;
    plotPRCurve(predictions, labels, epoch, opts);
    return
end

figure(1) ; clf ;
plotNames = getPlotNames(stats, opts);

for p = plotNames
    plotName = char(p);
    if ~strcmp(plotName, 'APstored')
        createSubPlot(plotName, plotNames, stats, epoch, opts);
    end
end

drawnow ;
print(1, opts.modelFigPath, '-dpdf') ;


end

% -------------------------------------------
function plotPRCurve(predictions, labels, epoch, opts)
% -------------------------------------------
scores = predictions(2,:) - predictions(1,:);

% Convert our labels from {1,2} to {-1,1} to work with
% the standard terminology
convertedLabels = zeros(size(labels));
convertedLabels(labels==1) = -1;
convertedLabels(labels==2) = 1;
figure(1); clf;
vl_pr(convertedLabels, scores, 'interpolate', false);
experimentPath = strsplit(opts.expDir, '/');
dim = [.6 .6 .3 .1];
str = strcat(experimentPath(3), ', tested using network from epoch: ', num2str(epoch));
annotation('textbox',dim,'String',str,'FitBoxToText','on');
% text(0,-0.5, );
drawnow;
print(1, opts.modelTestFigPath, '-dpdf');
end

% -------------------------------------------
function plotNames = getPlotNames(stats, opts)
% -------------------------------------------
% Returns a cell array of strings
% containing the names of the variables we are
% interested in plotting.
if opts.testMode
    targets = fieldnames(stats.val)';
else
    targets = cat(2, ...
        fieldnames(stats.train)', ...
        fieldnames(stats.val)');
end

plotNames = setdiff(targets, {'num', 'time'}) ;
end

% -------------------------------------------
function createSubPlot(plotName, plotNames, stats, epoch, opts)
% -------------------------------------------
values = zeros(0, epoch) ;
leg = {} ;

if opts.testMode
    statsFields = {'val'};
else
    statsFields = {'train', 'val'};
end

for f = statsFields
    field = char(f) ;
    if isfield(stats.(field), plotName)
        tmp = [stats.(field).(plotName)] ;
        values(end+1,:) = tmp(1,:)' ;
        leg{end+1} = field ;
    end
end
subplot(1,numel(plotNames),find(strcmp(plotName,plotNames))) ;
plot(1:epoch, values','o-') ;
xlabel('epoch') ;
title(plotName) ;
legend(leg{:}) ;
grid on ;
end
