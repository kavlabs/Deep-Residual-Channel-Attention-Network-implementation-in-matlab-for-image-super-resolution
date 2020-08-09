classdef maeRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = maeRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Mean absolute error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            imshow(Y);
            % Calculate MAE.
            R = size(Y,3);
            meanAbsoluteError = sum(abs(Y-T),3)/R;
            % Take mean over mini-batch.
            N1 = size(Y,2);
            loss1 = sum(meanAbsoluteError)/N1;
            N2 = size(loss1,2);
            loss2 = sum(loss1)/N2;
            N = size(Y,4);
            loss = sum(loss2)/N;
            disp(loss);
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the MAE loss with respect to the predictions Y

            R = size(Y,3);
            N = size(Y,4);
            dLdY = sign(Y-T)/(N*R);
        end
    end
end