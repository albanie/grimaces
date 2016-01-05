function dagnet = initRandomNet( varargin )
% returns a structure dagnet with the structure of the 
% pretrained network Alexnet, but every filter weight randomised.

% set defaults
opts.gradientThreshold = 0 ;
opts.numOutputs = 2;
opts.useDropout = true ;
opts.fineTuningRate = 1;
opts.lossType = 'softmaxloss';
opts.pretrainedNet = fullfile('models', 'alexnet.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

% load alexnet
pretrainedNet = load(opts.pretrainedNet);

% setup random number generation
rng('default'); rng(0) ;
fScale=1/100 ;

% modify fc8 layer for binary classification
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

% Use the dagnn wrapper to set the fine tuning rate on 
% each layer
dagnet = dagnn.DagNN.fromSimpleNN(pretrainedNet, 'canonicalNames', true);
paramIdx = dagnet.getParamIndex([dagnet.layers(1:end-2).params]);
[dagnet.params(paramIdx).learningRate] = deal(opts.fineTuningRate);


% RANDOMIZATION OF ALL TEH LAYERZ!!!!!!

% randomize conv1 layer
dagnet = randomizeLayer(dagnet, 'conv1b');
dagnet = randomizeLayer(dagnet, 'conv1f');
dagnet = randomizeLayer(dagnet, 'conv2b');
dagnet = randomizeLayer(dagnet, 'conv2f');
dagnet = randomizeLayer(dagnet, 'conv3b');
dagnet = randomizeLayer(dagnet, 'conv3f');
dagnet = randomizeLayer(dagnet, 'conv4b');
dagnet = randomizeLayer(dagnet, 'conv4f');
dagnet = randomizeLayer(dagnet, 'conv5b');
dagnet = randomizeLayer(dagnet, 'conv5f');
dagnet = randomizeLayer(dagnet, 'fc6b');
dagnet = randomizeLayer(dagnet, 'fc6f');
dagnet = randomizeLayer(dagnet, 'fc7b');
dagnet = randomizeLayer(dagnet, 'fc7f');

% Finally, we add a second loss layer to calculate class error
layer = dagnn.Loss('loss', 'classerror');
inputs = {'prediction','label'};
output = 'error';
dagnet.addLayer('error', layer, inputs, output) ;

% --------------------------------------------------------------------
function dagnet = randomizeLayer(dagnet, layerName)
% --------------------------------------------------------------------
% replace the params in the given layer with randomized
% weights of the same size
fScale=1/100 ;

paramIdx = dagnet.getParamIndex(layerName);
paramSize = size(dagnet.params(paramIdx).value);
dagnet.params(paramIdx).value = fScale * randn(paramSize, 'single');

