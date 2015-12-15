function net = initVggNet( varargin )
opts.fineTuningRate = 1;
opts.outputUnits = 2;
opts.useDropout = true ;
opts.lossType = 'softmaxloss';
opts.pretrainedNet = fullfile('models', 'vgg_face_16.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;

vggData = load(opts.pretrainedNet);
vggFaceNet = vggData.net;
% we only have 5 classes for the SOS dataset
vggFaceNet.layers{end-1}.weights = {f*randn(1,1,4096,opts.outputUnits, 'single'), zeros(1,opts.outputUnits,'single')};

% use softmaxloss layer for training
vggFaceNet.layers{end}.type = opts.lossType;

% optionally switch to batch normalization
if opts.useDropout
  vggFaceNet = insertDropout(vggFaceNet, 17) ;
  vggFaceNet = insertDropout(vggFaceNet, 20) ;
end

net = dagnn.DagNN.fromSimpleNN(vggFaceNet);
[net.params(net.getParamIndex([net.layers(1:end-2).params])).learningRate] = ...
  deal(opts.fineTuningRate);

net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;

% --------------------------------------------------------------------
function net = insertDropout(net, l)
% --------------------------------------------------------------------
layer = struct('type', 'dropout', ...
               'rate', 0.5) ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;