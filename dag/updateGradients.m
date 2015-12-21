function state = updateGradients(state, dagnet, opts, batchSize, mmap)

if ~isempty(mmap)
    write_gradients(mmap, dagnet) ;
    labBarrier() ;
end
state = accumulate_gradients(state, dagnet, opts, batchSize, mmap) ;

end

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  thisDecay = opts.weightDecay * net.params(i).weightDecay ;
  thisLR = state.learningRate * net.params(i).learningRate ;

  if ~isempty(mmap)
    tmp = zeros(size(mmap.Data(labindex).(net.params(i).name)), 'single') ;
    for g = setdiff(1:numel(mmap.Data), labindex)
      tmp = tmp + mmap.Data(g).(net.params(i).name) ;
    end
    net.params(i).der = net.params(i).der + tmp ;
  end

  state.momentum{i} = opts.momentum * state.momentum{i} ...
    - thisDecay * net.params(i).value ...
    - (1 / batchSize) * net.params(i).der ;

  net.params(i).value = net.params(i).value + thisLR * state.momentum{i} ;
end

end

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

end