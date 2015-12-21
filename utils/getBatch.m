function inputs = getBatch(imdb, batch)
% returns the inputs and associated labels in the 
% batch specified by 'batch', contained in 'imdb'.

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);
inputs = { 'input', input, ...
           'label', label};
end