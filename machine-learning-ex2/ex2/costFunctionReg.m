function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Calculates hypothesis
z=X*theta;
h=sigmoid(z);

%Calculates unregularized cost 
unreg_term=(-y'*log(h))-(1-y)'*log(1-(h));
unreg_term=unreg_term/m;

%Set theta0 as 0 to calculate the regularized term
theta(1)=0;

%Calculates reg term
%NOTE: It is a 1x1 term
regTerm= (lambda/(2*m)) * (theta'*theta);

%Cost
J=unreg_term+regTerm;

%Gradient
grad0=1/m*(h-y)'*X;
gradm=((lambda/m)*theta)';
grad=grad0+gradm;

% =============================================================

end
