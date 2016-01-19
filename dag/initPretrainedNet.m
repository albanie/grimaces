function dagnet = initPretrainedNet(opts)
% return the pretrained network specified in opts

if strcmp(opts.pretrainedNet, 'Alexnet')
    dagnet = initAlexnet(opts);
end

if strcmp(opts.pretrainedNet, 'ScaledAlexnet')
    dagnet = initScaledAlexnet(opts);
end

if strcmp(opts.pretrainedNet, 'Randomnet')
    dagnet = initRandomnet(opts);
end

if strcmp(opts.pretrainedNet, 'Vggfacenet')
    dagnet = initVggFaceNet(opts);
end

end

