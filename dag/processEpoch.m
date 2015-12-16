function stats = processEpoch(state, dagnet, opts, mode)

% initialize momentum to zero
state = initMomentum(state, dagnet, opts, mode);

% perform memory mapping if required
mmap = mapMemory(dagnet, opts);

% initialize processing variables
stats.time = 0;
stats.scores = [];
subset = state.(mode);
start = tic;
num = 0;

for subsetIdx=1:opts.batchSize:numel(subset)
    
    % If the size of the subset is not divisible by batchSize,
    % the last batch will be smaller
    batchSize = min(opts.batchSize, numel(subset) - subsetIdx + 1) ;
    
    for subBatchIdx=1:opts.numSubBatches
        % get the indices for this image batch
        batch = computeBatch(subsetIdx, subBatchIdx, subset, opts);
        
        % if the batch is empty, we are done
        if numel(batch) == 0, continue ; end
        
        % increment num to keep track of progress
        num = num + numel(batch) ;
        
        % get batch of network inputs
        inputs = state.getBatch(state.imdb, batch) ;
        
        % prefetch next batch if required
        if opts.prefetch
            prefetchNextBatch(batchStart, state, opts)
        end
        
        % evaluate the netowrk on this batch
        if strcmp(mode, 'train')
            dagnet.paramDersAccumulate = (subBatchIdx ~= 1) ;
            dagnet.eval(inputs, opts.derOutputs) ;
        else
            dagnet.eval(inputs) ;
        end
        
    end
    
    % extract stats from the loss layers
    stats = opts.extractStatsFn(dagnet) ;
    
    % accumulate gradients if training
    if strcmp(mode, 'train')
        state = updateGradients(state, dagnet, opts, batchSize, mmap);
    end
    
    % print learning statistics
    printNetworkBatchStats(state, stats, opts, subsetIdx, subset, start, num, mode);
end

dagnet.reset() ;
dagnet.move('cpu') ;
end