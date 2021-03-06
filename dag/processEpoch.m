function stats = processEpoch(state, dagnet, opts, mode)

% initialize momentum to zero
state = initMomentum(state, dagnet, opts, mode);

% perform memory mapping if required
mmap = mapMemory(dagnet, opts);

% initialize processing variables
stats.num = 0;
stats.time = 0;
stats.scores = [];
subset = state.(mode);
start = tic;
num = 0;

% we need to store values from each batch to calculate average precision
APstored.labels = [];
APstored.predictions = [];

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
        inputs = state.getBatch(state.imdb, batch, opts) ;
        
        % prefetch next batch if required
        if opts.prefetch
            prefetchNextBatch(batchStart, subsetIdx, subBatchIdx, subset, state, opts)
        end
        
        % evaluate the netowrk on this batch
        if strcmp(mode, 'train')
            dagnet.accumulateParamDers = (subBatchIdx ~= 1) ;
            dagnet.eval(inputs, opts.derOutputs) ;  
        else
            % To calculate AP statistics, we must retain the variables
            dagnet.eval(inputs) ;
            predictions = squeeze(dagnet.vars(dagnet.getVarIndex('prediction')).value);
            APstored.labels = horzcat(APstored.labels, inputs{4});
            APstored.predictions = horzcat(APstored.predictions, predictions);
        end
        
    end
    
    % extract stats from the loss layers
    stats = opts.extractStatsFn(dagnet) ;
    
    % accumulate gradients if training
    if strcmp(mode, 'train')
        state = updateGradients(state, dagnet, opts, batchSize, mmap);
    end
    
    % print learning statistics
    stats.num = num ;
    printNetworkBatchStats(state, stats, opts, subsetIdx, subset, start, mode);
end

if strcmp(mode, 'val')
    % calculate average precision
    scores = APstored.predictions(2,:) - APstored.predictions(1,:);
    
    % Convert our labels from {1,2} to {-1,1} to work with
    % the standard terminology
    convertedLabels = zeros(size(APstored.labels));
    convertedLabels(APstored.labels==1) = -1;
    convertedLabels(APstored.labels==2) = 1;
    [~,~,info] = vl_pr(convertedLabels, scores, 'interpolate', false);
     
    stats.averagePrecision = info.ap;
    stats.APstored = APstored;
    
end
    dagnet.reset() ;
    dagnet.move('cpu') ;