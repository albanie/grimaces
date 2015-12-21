function stats = runDAG(dagnet, imdb, getBatch, varargin)
% Returns statistics generated by running a CNN using the
% given options.

% Configure the DAG to use sensible values
opts = configureDAG(varargin{:});

% Check that we have a directory for the experiment
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% Find training and validation indices (in case they are not specified)
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------
% determine whether we are in training or validation mode
evaluateMode = getDAGMode(opts);

% Create an empty array to store statistics about each run
stats = [] ;

% If they are available, setup GPUs
opts.numGpus = setupGPUs(opts);

% Skip epochs that have already been completed, and initialize
% from saved state
start = opts.continue * findLastCheckpoint(opts);
if start >= 1
    fprintf('resuming by loading epoch %d\n', start) ;
    [dagnet, state] = loadState(opts.modelPath(start)) ;
end

% Add the imdb and getBatch function to a structure
% that will be passed around the network (must be done
% after loading historical state).
state.imdb = imdb;
state.getBatch = getBatch;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

for epoch=start+1:opts.numEpochs
    
    % update state for current epoch, shuffling the training indices
    state = updateDAGState(state, epoch, opts);
    
    % compute stats for epoch
    [dagnet, stats] = computeEpoch(state, dagnet, epoch, stats, opts);
    
    % save state after an epoch
    if ~evaluateMode
        saveState(opts.modelPath(epoch), dagnet, stats) ;
    end
    
    plotDAG(stats, epoch, opts)
end

end