function stats = processSequentially(state, dagnet, epoch, stats, opts)
% PROCESSSEQUENTIALLY is used when at most one GPU is available.

    stats.train(epoch) = processEpoch(state, dagnet, opts, 'train') ;
    stats.val(epoch) = processEpoch(state, dagnet, opts, 'val') ;
end