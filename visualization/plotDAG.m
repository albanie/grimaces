function plotDAG(stats, epoch, opts)
% plots the training and evaluation statistics
% of the DAG.

% Graphs are only created while training, not testing
if opts.testMode
    return
end

figure(1) ; clf ;
plotNames = getPlotNames(stats, opts);

for p = plotNames
    plotName = char(p) ;
    createSubPlot(plotName, plotNames, stats, epoch, opts);
end

drawnow ;
print(1, opts.modelFigPath, '-dpdf') ;


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