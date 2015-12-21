function plotDAG(stats, epoch, opts)
% plots the training and evaluation statistics of the 
% DAG.

figure(1) ; clf ;
plots = {'error', 'loss'};

for p = plots
    plotName = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    for mode = {'train', 'val'}
        mode = char(mode) ;
        if isfield(stats.(mode), plotName)
            tmp = [stats.(mode).(plotName)] ;
            values(end+1,:) = tmp(1,:)' ;
            leg{end+1} = mode ;
        end
    end
    subplot(1, numel(plots), find(strcmp(plotName,plots))) ;
    plot(1:epoch, values','o-') ;
    xlabel('epoch') ;
    title(plotName) ;
    legend(leg{:}) ;
    grid on ;
end
drawnow ;
print(1, opts.modelFigPath, '-dpdf') ;
end