%% Machine Learning Online Class - Exercise 2: Logistic Regression
%  Part 2: Adding regularization...

%% Initialization
clear ; close all; clc

% Functions...

function g = sigmoid(z)
	% Initialize g... 
	g = zeros(size(z));
	
	g = 1./(1 + e.^(-z));

end;

function out = mapFeature(X1, X2)

	degree = 6;
	out = ones(size(X1(:,1)));
	for i = 1:degree
		for j = 0:i
			out(:, end+1) = (X1.^(i-j)).*(X2.^j);
		end
	end

end

% Regularized cost function....
function [J, grad] = costFunctionReg(theta, X, y, lambda)

	% number of training examples
	m = length(y); 
	% Initializing J and grad...
	J = 0;
	grad = zeros(size(theta));
	
	predictors = sigmoid(X * theta);
	% theta_0 (index = 1) is omitted from regularization...
	J = (-1/m) * sum(y .* log(predictors) + (1 - y) .* log(1 - predictors))+ lambda/(2*m) * sum(theta(2:size(X,2)) .^2);
	
	grad(1) = 1/m * ((predictors - y)' * X(:,1));
	grad(2:size(theta,1)) = 1/m * ((predictors - y)' * X(:,2:size(X,2)) + lambda * theta(2:size(theta,1))');
	% Transpose grad so it has the same dimensions as theta
	grad = grad';	
	
end;

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
		end;
	end;
end;

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

end;

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Regularization and Accuracies...

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


