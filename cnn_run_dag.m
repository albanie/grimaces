function stats = cnn_run_dag(net, imdb, getBatch, varargin)

opts.batchSize = 100 ;
opts.testSet = [] ;
opts.gpus = [] ;
opts.numThreads = 1;
opts.extractStatsFn = @extractStats ;
opts.mergeStatsFn = @(x,y) [x, y];
opts.printStatsFn = @(x) [];
opts.prefetch = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 0
  numThreads = numGpus ;
else
  numThreads = opts.numThreads ;
end

if numThreads > 1
  pool = gcp('nocreate');
  if isempty(pool), pool = parpool('local',numThreads); end
  
  numThreads = min(pool.NumWorkers, numThreads);
end

if numGpus > 1
  spmd(numThreads), gpuDevice(opts.gpus(labindex)), end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

% train one epoch
state.imdb = imdb ;

if numThreads == 1
  stats = run_network(net, state, opts) ;
	else
	  savedNet = net.saveobj() ;
	  spmd(numThreads)
	    net_ = dagnn.DagNN.loadobj(savedNet) ;
	    stats_ = run_network(net_, state, opts) ;
	  end
	  stats = stats_{1};
	  
	  for i=2:numel(stats_)
	    stats = opts.mergeStatsFn(stats, stats_{i});
	  end
	end

	% -------------------------------------------------------------------------
	function stats = run_network(net, state, opts)
	% -------------------------------------------------------------------------

	numGpus = numel(opts.gpus) ;
	if numGpus >= 1
	  net.move('gpu') ;
	end

	subsetSize = ceil(numel(opts.testSet) / numlabs);
	subsetStart = 1 + (labindex-1) * subsetSize;
	subsetEnd = min(subsetStart + subsetSize - 1, numel(opts.testSet));
	subset = opts.testSet(subsetStart:subsetEnd);
	start = tic ;
	num = 0 ;

	stats = [];

	for t=numel(subset) : -opts.batchSize : 1
	  % get this image batch and prefetch the next
	  batchStart = t ;
	  batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
	  batch = subset(batchStart : batchEnd) ;
	  num = num + numel(batch) ;
	  if numel(batch) == 0, continue ; end

	  inputs = state.getBatch(state.imdb, batch) ;

	  if opts.prefetch
	    batchStart = t + opts.batchSize ;
	    batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
	    
	    nextBatch = subset(batchStart : batchEnd) ;
	    state.getBatch(state.imdb, nextBatch) ;
	  end

	  net.eval(inputs) ; 

  % extract stats
  if isempty(stats)
    stats = opts.extractStatsFn(net) ;
  else
    stats = opts.mergeStatsFn(stats, opts.extractStatsFn(net)) ;
  end

  % print learning statistics
  time = toc(start) ;

  fprintf('batch %3d/%3d: %.1f Hz', ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    num/time) ;
  
  opts.printStatsFn(stats) ;
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
lossLayerIdx = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(lossLayerIdx)
  lossLayer = net.layers(lossLayerIdx(i));
  stats.(lossLayer.name) = lossLayer.block.average;
end

