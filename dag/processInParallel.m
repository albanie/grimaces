function [dagnet, stats] = processInParallel(state, dagnet, epoch, stats, opts)
% PROCESSINPARALLEL distributes the work of processing an epoch across
% the available gpus.

    % convert dagnet to vanilla matlab struct
    savedNet = dagnet.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = processEpoch(net_, state, opts, 'train') ;
      stats_.val = processEpoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    % TODO: Understand why we load from the first GPU
    dagnet = dagnn.DagNN.loadobj(savedNet_{1}) ; 
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
end