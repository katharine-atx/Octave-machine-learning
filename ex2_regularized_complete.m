% Machine Learning Online Class - Exercise 2: Logistic Regression
% Part 2: Regularization

% Assignment:
% Implement regularized logistic regression to predict whether microchips 
% from a fabrication plant passes quality assurance (QA). You have the
% test results for some microchips on two different tests. From these two tests,
% you would like to determine whether the microchips should be accepted or
% rejected. To help you make the decision, you have a dataset of test results
% on past microchips, from which you can build a logistic regression model.

%% Initialization
clear ; close all; clc

%% Load Data
data = load('ex2data2.txt');
%  The first two columns contains the X values and the third column
%  contains the label (y).
X = data(:, [1, 2]); 
y = data(:, 3);

%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

function plotData(X, y)
	% Create New Figure
	figure; hold on;

	% Find Indices of Positive and Negative Examples
	pos = find(y==1); neg = find(y == 0);

	% Plot Examples
	plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
	'MarkerSize', 7);
	plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
	'MarkerSize', 7);
	hold off;

end

plotData(X, y);

hold on;
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;


% Part 1: Regularized Logistic Regression
% The decision boundary does not appear linear. Adding polynomial features 
% to the data matrix in order to use logistic regression effectively...

% MAPFEATURE(X1, X2) maps the two input features
% to quadratic features used in the regularization exercise.
% Returns a new feature array with X1, X2, X1.^2, X2.^2, X1*X2, 
% X1*X2.^2, etc.. as well as the intercept vector (ones)
% Note: Requires that X1 and X2 inputs be the same size

function out = mapFeature(X1, X2)

	degree = 6;
	out = ones(size(X1(:,1)));
	for i = 1:degree
		for j = 0:i
			out(:, end+1) = (X1.^(i-j)).*(X2.^j);
		end
	end

end

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
m = length(y)

% Set regularization parameter lambda to 1
lam = 1;

% J = SIGMOID(z) computes the sigmoid transformation for predictions z
% Note z can be a matrix, vector or scalar.

function g = sigmoid(z)
	% Initialize g... 
	g = zeros(size(z));
	g = 1./(1 + e.^(-z));

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)

	% Initialize values...
	m = length(y); 
	J = 0;
	initial_theta = zeros(size(X, 2), 1);
	grad = zeros(size(initial_theta));
	
	predictors = sigmoid(X * initial_theta);
	% theta_0 (index = 1) is omitted from regularization...
	J = (-1/m) * sum(y .* log(predictors) + (1 - y) .* log(1 - predictors))+ lambda/(2*m) * sum(initial_theta(2:size(X,2)) .^2);
	grad(1) = 1/m * ((predictors - y)' * X(:,1));
	grad(2:size(theta,1)) = 1/m * ((predictors - y)' * X(:,2:size(X,2)) + lambda * initial_theta(2:size(initial_theta,1))');
	% Transpose grad so it has the same dimensions as theta
	grad = grad';	
	
end

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lam);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Part 2: Regularization and Accuracies
% Try the following values of lambda (0, 1, 10, 100) and note impact
% on decision boundary, training set accuracy.

function plotDecisionBoundary(theta, X, y)
	% Plot Data
	plotData(X(:,2:3), y);
	hold on;
	if size(X, 2) <= 3,
		% Only need 2 points to define a line, so choose two endpoints
		plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
		% Calculate the decision boundary line
		plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
		% Plot, and adjust axes for better viewing
		plot(plot_x, plot_y)
        % Labels and Legend
		xlabel('Microchip Test 1')
		ylabel('Microchip Test 2')
		legend('y = 1', 'y = 0', 'Decision boundary')
	else
		% Here is the grid range
		u = linspace(-1, 1.5, 50);
		v = linspace(-1, 1.5, 50);
		z = zeros(length(u), length(v));
		% Evaluate z = theta*x over the grid
		for i = 1:length(u)
			for j = 1:length(v)
				z(i,j) = mapFeature(u(i), v(j))*theta;
			end
		end
		z = z'; % important to transpose z before calling contour
		% Plot z = 0
		% Notice you need to specify the range [0, 0]
		contour(u, v, z, [0, 0], 'LineWidth', 2)
	end
	hold off

end

function p = predict(theta, X)
	m = size(X, 1); % Number of training examples
	% initialize p...
	p = zeros(m, 1);

	for i = 1:m
		if sigmoid(X(i,:)*theta)>= 0.5,
			p(i) = 1;
		elseif sigmoid(X(i,:)*theta)< 0.5,
			p(i) = 0;
		else
			disp('Error: sigmoid function failure')
		end
	end
end




