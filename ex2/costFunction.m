function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
one = ones(m,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X*theta;
h = sigmoid(z);
logh = log(h);
log1h = log(one - h);
H = -y.*(logh) - (one - y).*(log1h);

J = (1/m).*(sum(H));

G = (h - y).*X;
grad1 = (1/m).*(sum(G(:,1)));
grad2 = (1/m).*(sum(G(:,2)));
grad3 = (1/m).*(sum(G(:,3)));

grad(1,1) = grad1;
grad(2,1) = grad2;
grad(3,1) = grad3;

grad;


% =============================================================

end
