function dagnet = initVggFaceNet( varargin )
% returns a structure dagnet with the structure of the
% pretrained network VggFaceNet, together with a few modifications.

% set defaults
opts.gradientThreshold = 0 ;
opts.numOutputs = 2;
opts.useDropout = true ;
opts.fineTuningRate = 0.05;
opts.lossType = 'softmaxloss';
opts.pretrainedNetModel = fullfile('models', 'vgg-face.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

% load vggfacenet
pretrainedNet = load(opts.pretrainedNetModel);

% modify last fully connected layer for binary classification
rng('default'); rng(0);
fScale = 1/100;
filters = fScale * randn(1,1,4096, opts.numOutputs, 'single');
biases = zeros(1, opts.numOutputs, 'single');
modifiedFc8 = { filters, biases };
pretrainedNet.layers{end -1}.weights = modifiedFc8;

% modify the meta attributes of the net
pretrainedNet.classes.name = {'flat', 'steep'} ;
pretrainedNet.classes.description = {'gradient < threshold', 'gradient > threshold'};
pretrainedNet.classes.threshold = opts.gradientThreshold ;

% switch final layer to softmaxloss (for training)
pretrainedNet.layers{end}.type = opts.lossType;

% set the fine tuning rate on each layer
dagnet = dagnn.DagNN.fromSimpleNN(pretrainedNet, 'canonicalNames', true);
paramIdx = dagnet.getParamIndex([dagnet.layers(1:end-2).params]);
[dagnet.params(paramIdx).learningRate] = deal(opts.fineTuningRate);

% Finally, we add a second loss layer to measure AveragePrecsion 
% (with matconvnet, it is easier to comput 
layer = dagnn.Loss('loss', 'averageprecision');

inputs = {'prediction','label'};
output = 'averagePrecision';
dagnet.addLayer('averagePrecision', layer, inputs, output) ;