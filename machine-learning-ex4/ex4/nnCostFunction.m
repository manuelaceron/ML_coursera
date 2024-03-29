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

%...............PART 1:Feedforward...............
K = num_labels;

%Adding bias in input
a1= [ones(m,1) X];

%Hidden layer output
z2= Theta1*a1';
a2= sigmoid(z2); %25x5000

%Adding bias at output layer input
a2= [ones(1, m); a2];
z3= a2'*Theta2';
a3= sigmoid(z3); 
h=a3; %5000x10

% (m,k) matrix with y output for all the input set
y_new = eye(num_labels);
y_new= y_new(y, :); %5000x10

%Unregularized term
a=-y_new.*log(h); 
b=(1-y_new).*log(1-h);
cost=a-b;
J = (1/m) * sum(cost(:));

%Regularized term

theta1_reg= Theta1(:,2:end);
theta2_reg= Theta2(:,2:end);

regTerm = sumsqr(theta1_reg(:)) + sumsqr(theta2_reg(:));
regTerm = (lambda/(2*m))*regTerm;

J= J + regTerm;


%...............PART 2: Backpropagation...............

D1=0;
D2=0;

for i = 1:m
delta3= h(i,:)-y_new(i,:); %1x10
delta3= delta3'; %10x1
delta2= Theta2(:,2:end)'*delta3 .* sigmoidGradient(z2(:,i)); %25x10x * 10x1 = 25x1 .* 25x1 = 25x1

D2= D2 + (delta3 * a2(:,i)'); % ... + 10x1 * 1x25

D1= D1 + (delta2 * a1(i,:)); % ... + 25x1 * 1x400


end
%Regularized term
Theta2_grad(:,2:end) = 1/m*D2(:,2:end) + (lambda/m)*Theta2(:,2:end); 
Theta1_grad(:,2:end) = 1/m*D1(:,2:end) + (lambda/m)*Theta1(:,2:end); 

%Unregularized term
Theta2_grad(:,1)     = 1/m*D2(:,1);         
Theta1_grad(:,1)     = 1/m*D1(:,1); 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
