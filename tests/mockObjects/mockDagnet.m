classdef mockDagnet
    
    properties
        params
    end
    
    methods
        function move(obj, device)
        % A mock of the dagnn.DagNN.move() method
        switch device
            case 'gpu'
                disp 'moved to gpu';
            case 'cpu'
                disp 'moved to cpu';
            otherwise
                error('device must be gpu or cpu');
        end
        end
    end
end
