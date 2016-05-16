%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
% Implementing one-vs-all logistic regression and neural networks to 
% recognize hand-written digits.

% Data notes: Source data contains 5000 training examples of handwritten digits. 
% These matrices can be read directly using the load command. There are 5000 
% training examples, where each training example is a 20 pixel by 20 pixel 
% grayscale image of the digit. Each pixel is represented by a floating point 
% number indicating the grayscale intensity. The 20 by 20 grid of pixels is 
% "unrolled" into a 400-dimensional vector as a row of matrix X. A 
% 5000-dimensional vector y contains labels for the training set. 
% Note: to avoid a zero index, the digit zero has been mapped to the value ten. 

% Neural network weights: Theta1 (dim 25 x 401), Theta2 (dim 10 X 26)
%% Initialization
clear ; close all; clc

% Parameters...
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 digit labels  

% Functions...
						  
function [h, display_array] = displayData(X, example_width)
	% Function displays 2D data stored in X in a grid. 
	% Set default example_width ...
	if ~exist('example_width', 'var') || isempty(example_width) 
		example_width = round(sqrt(size(X, 2)));
	end
	colormap(gray); % Grayscale image
	% Compute rows, cols
	[m n] = size(X);
	example_height = (n / example_width);
	% Compute number of items to display
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);
	pad = 1; % Padding between images
	% Display blank grid...
	display_array = - ones(pad + display_rows * (example_height + pad), ...
						   pad + display_cols * (example_width + pad));
	% Copy each example into a patch on the display array
	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m, 
				break; 
			end
			% Get the max value of the patch
			max_val = max(abs(X(curr_ex, :)));
			display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
						  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
							reshape(X(curr_ex, :), example_height, example_width) / max_val;
			curr_ex = curr_ex + 1;
		end
		if curr_ex > m, 
			break; 
		end
	end
	% Display Image
	h = imagesc(display_array, [-1 1]);
	% Do not show axis
	axis image off
	drawnow;
end

function g = sigmoid(z)
	% Sigmoid function...
	g = 1 ./ (1 + e.^(-z));
end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
	% Function computes logistic regression cost with regularization
	% Initialize values...
	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));
	predictors = sigmoid(X * theta);
	% theta_0 (index = 1) is omitted from regularization...
	J = (-1/m) * sum(y .* log(predictors) + (1 - y) .* log(1 - predictors))+ lambda/(2*m) * sum(theta(2:size(X,2)) .^2);
	grad(1) = 1/m * ((predictors - y)' * X(:,1));
	grad(2:size(theta,1)) = 1/m * ((predictors - y)' * X(:,2:size(X,2)) + lambda * theta(2:size(theta,1))');
	grad = grad';	
end

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
	% trains num_labels logisitc regression classifiers and returns each of these classifiers
	% in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier for label i.
	% Initialize values...
	m = size(X, 1);
	n = size(X, 2);
	all_theta = zeros(num_labels, n + 1);
	% Add ones to the X data matrix
	X = [ones(m, 1) X];
	for c=1:num_labels
		initial_theta = zeros(n+1, 1);
		% Set options for fminunc
		options = optimset('GradObj', 'on', 'MaxIter', 50);
		% Run fminunc to obtain the optimal theta
		% This function will return theta and the cost 
		[theta] = ...
			fminunc(@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
				initial_theta, options);
		all_theta(c, :) = theta(:);
	end
end

function p = predictOneVsAll(all_theta, X)
	% returns a vector of predictions for each example in the matrix X. 
	% Initialize values...
	m = size(X, 1);
	num_labels = size(all_theta, 1);
	p = zeros(size(X, 1), 1);
	% Add ones to the X data matrix
	X = [ones(m, 1) X];
	% Predictions....
	predictions = sigmoid(X * all_theta');
	% Grab indices of max probability (these correspond to digit label)
	[pred_max, index_max] = max(predictions, [], 2);
	p = index_max;
end

function p = predict(Theta1, Theta2, X)
	% outputs the predicted label of X given the trained weights of a 
	% neural network (Theta1, Theta2)
	% Initialize values...
	m = size(X, 1);
	num_labels = size(Theta2, 1);
	p = zeros(size(X, 1), 1);
	% Add ones to the X data matrix
	X = [ones(m, 1) X];
	% Note: output prediction layer is layer 3 for this network.
	% Layer 1 to 2 mapped by Theta1 weights... add bias term = 1
	a2 = [ones(m,1) sigmoid(X * Theta1')];
	% Layer 2 to 3 mapped by Theta2 weights...
	a3 = a2 * Theta2';
	[pred_max, index_max] = max(a3, [], 2);
	p = index_max;
end

% Script...

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nLoading Saved Neural Network Parameters ...\n')
% Load the weights into variables Theta1 and Theta2
% These are given in this script example...
load('ex3weights.mat');

% Neural network predictions...
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

%  To see the network's output, run through the examples one at the a time...
%  Randomly permute examples
rp = randperm(m);
for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
	% Adding break....
	if i > 10, 
			break; 
	end
end

