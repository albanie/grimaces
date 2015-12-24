function inputs = getBatch(imdb, batch, opts)
% returns the inputs and associated labels in the 
% batch specified by 'batch', contained in 'imdb'.

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);

% Move input to the GPU if needed.
if numel(opts.gpus) > 0
  input = gpuArray(input) ;
end

inputs = { 'input', input, ...
           'label', label};
end
