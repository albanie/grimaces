function dagnet = initAlexnet( varargin )
% returns a structure dagnet with the structure of the
% pretrained network Alexnet, together with a few modifications.

% set defaults
opts.gradientThreshold = 0 ;
opts.numOutputs = 2;
opts.useDropout = true ;
opts.fineTuningRate = 1;
opts.lossType = 'softmaxloss';
opts.pretrainedNetModel = fullfile('models', 'alexnet.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

% load alexnet
pretrainedNet = load(opts.pretrainedNetModel);

% modify fc8 layer for binary classification
rng('default'); rng(0) ;
fScale=1/100 ;
filters = fScale*randn(1,1,4096, opts.numOutputs, 'single');
biases = zeros(1, opts.numOutputs,'single');
modifiedFc8 = { filters, biases };
pretrainedNet.layers{end-1}.weights = modifiedFc8;

% modify the meta attributes of the net
pretrainedNet.classes.name = {'flat', 'steep'} ;
pretrainedNet.classes.description = {'gradient < threshold', 'gradient > threshold'};
pretrainedNet.classes.threshold = opts.gradientThreshold ;

% switch final layer to softmaxloss (for training)
pretrainedNet.layers{end}.type = opts.lossType;

% optionally switch to batch normalization
if opts.useDropout
    pretrainedNet = insertDropout(pretrainedNet, 17) ;
    pretrainedNet = insertDropout(pretrainedNet, 20) ;
end

% Use the dagnn wrapper to set the fine tuning rate on
% each layer
dagnet = dagnn.DagNN.fromSimpleNN(pretrainedNet, 'canonicalNames', true);
paramIdx = dagnet.getParamIndex([dagnet.layers(1:end-2).params]);
[dagnet.params(paramIdx).learningRate] = deal(opts.fineTuningRate);

% Finally, we add a second loss layer to AveragePrecsion 
% (technically it is  1 - AP)
layer = dagnn.Loss('loss', 'averageprecision');

inputs = {'prediction','label'};
output = 'error';
dagnet.addLayer('error', layer, inputs, output) ;

% --------------------------------------------------------------------
function net = insertDropout(net, num)
% --------------------------------------------------------------------
% inserts a dropout layer at the index specified by 'num'.
layer = struct('type', 'dropout', 'rate', 0.5) ;
net.layers = horzcat(net.layers(1:num), layer, net.layers(num+1:end)) ;