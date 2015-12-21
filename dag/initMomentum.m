function state = initMomentum(state, dagnet, opts, mode)
% SETMOMENTUM zero-initializes a momentum variable for each network
% layer with parameters, and stores it in the state structure.

if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(dagnet.params))) ;
end

if numel(opts.gpus) >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
end
