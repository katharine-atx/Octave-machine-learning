% Machine Learning Online Class - Exercise 2: Logistic Regression

% Assignment: You have historical data from previous applicants
% that you can use as a training set for logistic regression. For each training
% example, you have the applicant's scores on two exams and the admissions
% decision. Your task is to build a classifcation model that estimates an 
% applicant's probability of admission based the scores from those two exams.

%% Initialization
clear ; close all; clc

% Loading data...
data = load('ex2data1.txt');

% The first two columns contains the exam scores (X) and the third column
% contains the label (Y binary classification).
X = data(:, [1, 2]); y = data(:, 3);
m = length(y); 

% Part 1: Plotting the data
% PLOTDATA(x,y) plots the data points with + for the positive examples
% and o for the negative examples. 
% Note: X is assumed to be a Mx2 matrix.

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
end;

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Part 2: Compute Prediction Cost and Gradient 

% Initialize X matrix
[m, n] = size(X);

% Add intercept vector (ones) to X
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% J = SIGMOID(z) computes the sigmoid transformation for predictions z
% Note z can be a matrix, vector or scalar.

function g = sigmoid(z)
	% Initialize g... 
	g = zeros(size(z));
	
	g = 1./(1 + e.^(-z));

end;

% COSTFUNCTION Compute cost and gradient for logistic regression

function [J, grad] = costFunction(theta, X, y)
	% number of training examples
	m = length(y); 
	% Initializing J and grad...
	J = 0;
	grad = zeros(size(theta));

	predictors = sigmoid(X * theta);
	J = -(1/m * sum(y .* log(predictors) + (1 - y) .* log(1 - predictors)));

	% Indexing starts at 1; gradients for theta_0, theta_1 and theta_2
	% Transpose grad so it has the same dimensions as theta
	grad(1) = 1/m * (predictors - y)' * X(:,1);
	grad(2) = 1/m * (predictors - y)' * X(:,2);
	grad(3) = 1/m * (predictors - y)' * X(:,3);
	grad = grad';

end;

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% Part 3: Optimizing using fminunc

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta and cost
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


% PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

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
    
    % Legend, specific for the exercise
		legend('Admitted', 'Not admitted', 'Decision Boundary')
		axis([30, 100, 30, 100])
	else
    % Here is the grid range
		u = linspace(-1, 1.5, 50);
		v = linspace(-1, 1.5, 50);

		z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
		for i = 1:length(u)
			for j = 1:length(v)
				z(i,j) = mapFeature(u(i), v(j))*theta;
			end;
		end;
		z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
		contour(u, v, z, [0, 0], 'LineWidth', 2)
	end;
	hold off

end;

% Plot Boundary
plotDecisionBoundary(theta, X, y);

hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Part 4: Prediction and Accuracy

% Use the logistic regression model to predict the probability that a student with 
% score 45 on exam 1 and score 85 on exam 2 will be admitted.

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% p = PREDICT(theta, X) computes the predictions for X using a 
% threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
% returning a vector of predicted 0 and 1 values.		 

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

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

