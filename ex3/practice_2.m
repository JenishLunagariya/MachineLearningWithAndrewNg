input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;         


load('ex3data1.mat');
m = size(X,1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('\nLoading Saved Neural Network Parameters ...\n');
load('ex3weights.mat');

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);





