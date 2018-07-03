function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


classes = 1:num_labels;
X1 = [ones(m,1) X];
Z2 = X1*Theta1';
A2 = [ones(size(Z2,1),1) sigmoid(Z2)];
Z3 = A2*Theta2';
hx= [sigmoid(Z3)];

Y = cell2mat(arrayfun(@(yi) yi== classes,y,'UniformOutput',false));

cost=(-Y.*log(hx)-(1-Y).*log(1-hx));
J = (1/m) * (sum( cost(:)) );

reg1 = Theta1(:,2:end).^2;
reg1 = sum(reg1(:));

reg2 = Theta2(:,2:end).^2;
reg2 = sum(reg2(:));

reg = (lambda /(2*m)) * (reg1+reg2);

J = J + reg;

 
% for i=1:m
%     yiv = (classes == y(i));
%     for k=1:num_labels
%         J = J + ( -( yiv(k)*log(hx(i,k)) ) - ( (1-yiv(k))*(log(1-hx(i,k))) ) );
%     end
% end
% J=J/m;
% 
% reg1=0;
% reg2=0;
% 
% for j=1:hidden_layer_size
%     for k=2:input_layer_size+1
%         reg1 = reg1 + Theta1(j,k)^2;
%     end
% end
%         
% for j=1:num_labels
%     for k=2:hidden_layer_size+1
%         reg2 = reg2 + Theta2(j,k)^2;
%     end
% end
% 
% reg = (lambda /(2*m)) * (reg1+reg2) ;
% J = J +reg;

% D3 = zeros(m,num_labels);
% for t=1:m
%     for k=1:num_labels
%        D3(t,k)=hx(t,k)-Y(t,k); 
%     end
% end

d3 = hx - Y;
d2_0= (d3*Theta2) .* sigmoidGradient([ones(size(Z2,1),1) Z2]);
d2 = d2_0(:,2:end);

D1= d2' * X1;
D2= d3' * A2;

Theta1_grad = (1/m).*D1 +(lambda/m)* ([zeros(size(Theta1,1),1) Theta1(:,2:end)]);
Theta2_grad = (1/m).*D2 +(lambda/m)* ([zeros(size(Theta2,1),1) Theta2(:,2:end)]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
