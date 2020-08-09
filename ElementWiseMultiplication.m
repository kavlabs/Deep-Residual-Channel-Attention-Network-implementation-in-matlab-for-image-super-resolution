% classdef ElementWiseMultiplication < nnet.layer.Layer
%     % Example custom ElementWiseMultiplication layer.
%     properties (Learnable)
%         % Layer learnable parameters
%             
%         % Scaling coefficients
%     end
%     
%     methods
%         function layer = ElementWiseMultiplication(numInputs,name) 
%             layer.NumInputs = numInputs;
%             layer.Name = name;
%             layer.Description = "Element Wise Multiplication of " + numInputs +  ... 
%                 " inputs";
%         
%         end
%         
%         function Z = predict(layer, varargin)
%             X = varargin;
%             X1 = X{1};
%             X2 = X{2};
%             Z = X1 .* X2;
% %                         Z = X1 .*X2;
%         
%         end
%         function varargout = backward(layer, varargin,~,~,~,~)
%             % [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
%             % backward propagates the derivative of the loss function
%             % through the layer.
%             
%             numInputs = layer.NumInputs;
%             W = layer.Weights;
%             X = varargin(1:numInputs);
%             dLdZ = varargin{numInputs+2};
%             
%             % Calculate derivatives
%             dLdX = cell(1,numInputs);
%             dLdW = zeros(1,numInputs,'like',W);
%             for i = 1:numInputs                
%                 dLdX{i} = dLdZ * W(i);
%                 dLdW(i) = sum(dLdZ .* X{i},'all');
%             end
%             
%             % Pack output arguments.
%             varargout(1:numInputs) = dLdX;
%             varargout{numInputs+1} = dLdW;
%         end
% %         function dLdX = backward(layer, X ,Z,dLdZ,memory,~)
% %             % Backward propagate the derivative of the loss function through 
% %             % the layer 
% %             W = layer.Weights;
% %             numInputs = layer.NumInputs;
% %             dLdX = dLdZ;
% %     end
% % methods
% %     function layer = ElementWiseMultiplication(numInputs,name) 
% %             layer.NumInputs = numInputs;
% %             layer.Name = name;
% %             layer.Description = "Element Wise Multiplication of " + numInputs +  ... 
% %                 " inputs";
% %     end
% %     function outputs = predict(layer,obj, inputs, params)
% %       if numel(inputs) ~= 2
% %         error('Number of inputs is not 2');
% %       end
% %       outputs{1} = inputs{1} .* inputs{2} ;
% %     end
% %     
% %     function varargout = backward(layer, varargin)
% % %             [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
% % %             backward propagates the derivative of the loss function
% % %             through the layer.
% %             
% %             numInputs = layer.NumInputs;
% %             W = layer.Weights;
% %             X = varargin(1:numInputs);
% %             dLdZ = varargin{numInputs+2};
% %             
% %             % Calculate derivatives
% %             dLdX = cell(1,numInputs);
% %             dLdW = zeros(1,numInputs,'like',W);
% %             for i = 1:numInputs                
% %                 dLdX{i} = dLdZ * W(i);
% %                 dLdW(i) = sum(dLdZ .* X{i},'all');
% %             end
% %             
% %             % Pack output arguments.
% %             varargout(1:numInputs) = dLdX;
% %             varargout{numInputs+1} = dLdW;
% %         end
% %     
% % %     function [derInputs, derParams] = backward(layer,obj, inputs, params, derOutputs)
% % %       derInputs = cell(1,2) ;
% % %       derInputs{1} = derOutputs{1} .* inputs{2}  ;
% % %       derInputs{2} = derOutputs{1} .* inputs{1}  ;
% % %       derParams = {} ;
% % %     end
% % %     
% %     function obj = Times(layer,varargin)
% %       obj.load(varargin) ;
% %     end
% % %     
% % %     function rfs = getReceptiveFields(obj)
% % %       rfs.size = [1 1] ;
% % %       rfs.stride = [1 1] ;
% % %       rfs.offset = [1 1] ;
% % %     end
% % % 
% % %     function outputSizes = getOutputSizes(obj, inputSizes)
% % %       outputSizes = inputSizes(1) ;
% % %     end
%   end
% end
classdef ElementWiseMultiplication < nnet.layer.Layer
    % Example custom ElementWiseMultiplication layer.
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
    end
    
    methods
        function layer = ElementWiseMultiplication(numInputs,name) 
            % layer = ElementWiseMultiplication(numInputs,name) creates a
            % element wise multiplication and specifies the number of inputs
            % and the layer name.
            % Set number of inputs.
            layer.NumInputs = numInputs;
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Element Wise Multiplication of " + numInputs +  ... 
                " inputs";
        
        end
        
%         function outputs = predict(layer, inputs, params)
%             % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
%             % ..., Xn through the layer and outputs the result Z.     
%             % Element Wise Multiplication
% %                         Z = X1 .*X2;
%                         outputs = inputs(1) .* inputs(2) ;
%         
%         end
        function outputs = predict(layer, varargin)
%           if numel(inputs) ~= 2
%               disp(numel(inputs));
%             error('Number of inputs is not 2');
%           end
            X1 = varargin{1};
            X2 = varargin{2};
%             disp(size(varargin));
%             disp(">>>>>>>>>>>>>>>");
%             disp(X1);
%             disp("???????????????");
%             disp(size(X1));
%             disp("???????????????");
%             disp(X2);
%             disp(">>>>>>>>>>>>>>");
%             disp(size(X2));
%             disp("???????????????");
          outputs = X1 .* X2 ;
        end
%         function [a,b] = backward(layer, X ,Z,dLdZ,memory,varargin)
%             X1 = varargin{1};
%             X2 = varargin{2};
%             a = zeros(X1);
%             b = zeros(X2);
% %             dLdX = dLdZ;
%         end
        function [derInputs1, derInputs2] = backward(layer ,X1,X2,~ ,~, varargin)
%             disp(size(varargin));
%             X1 = varargin{1};
%             X2 = varargin{2};
%             z = varargin(3);
%             dLdZ = varargin(4);
            sz1 = size(X1);
            sz2 = size(X2);
            derInputs1 = zeros(sz1,'like',X1);
            derInputs2 = zeros(sz2,'like',X2);
%             derInputs1 = dLdZ   ;
%             derInputs2 = dLdZ  ;
        end
    end
end