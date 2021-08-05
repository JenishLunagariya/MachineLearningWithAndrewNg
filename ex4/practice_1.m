input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');
m = size(X, 1);


sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('\nLoading Saved Neural Network Parameters ...\n')

load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];

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

a1 = [ones(m,1) X];  %(5000,401)

z2 = Theta1 * a1';   %(25,5000) 
a2 = sigmoid(z2);    
a2 = [ones(m,1) a2'];%(5000,26)

z3 = a2 * Theta2' ;  %(5000,10)
a3 = sigmoid(z3);    %(5000,10)

yt = zeros(m,num_labels);
for i = 1:m;
	j = y(i);
    yt(i,j) = 1;
end;
H = -yt.*log(a3)-(1-yt).*log(1-a3);
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

J = 1/m * sum(sum(H)) + 1/(2*m) * (sum(sum(t1.^2)) + sum(sum(t2.^2)) )
%for i = 1:m;
%	a1 = a1(i,:);
%	z2 = Theta1 * a1';
%	a2 = sigmoid(z2);
%
%	a2 = [1;a2];
%	z3 = Theta2*a2;
%	a3 = sigmoid(z3);
%end; 
d3 = a3 - yt; %(5000,10);
Z2 = [ones(1,m) ; z2];  % (26,5000)
d2 = (d3 * Theta2).*sigmoidGradient(Z2');  %(5000,26)


Theta1_grad = Theta1_grad + (d2(:,2:end)' * a1);  %(25,401)
Theta2_grad = Theta2_grad + (d3' * a2);  %(10,26)

Theta1_grad = (1/m)*Theta1_grad;    %(25,401)
Theta2_grad = (1/m)*Theta2_grad;    %(10,26)

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

grad = [Theta1_grad(:) ; Theta2_grad(:)];
