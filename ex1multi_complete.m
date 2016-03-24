% Exercise 1: Linear regression with multiple variables
clear ; close all; clc 

fprintf('Loading data ...\n');
data = load('ex1data2.txt'); %% Load Data
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
fprintf('First 10 examples from the dataset: \n'); % Print out 10 rows of data
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Normalizing Features ...\n'); 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation is 1.
function [X_norm, mu, sigma] = featureNormalize(X)
	X_norm = X; % You need to set these values correctly
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2)); 
     
	mu = [mean(X(:, 1)) mean(X(:, 2))];
	sigma = [std(X(:, 1)) std(X(:, 2))];
	X_norm = (X - mu)./sigma;
	X = X_norm;
end

[X mu sigma] = featureNormalize(X); %run featureNormalize
X = [ones(m, 1) X]; % Add intercept term to X
theta = zeros(3, 1);

%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
function J = computeCostMulti(X, y, theta)
	m = length(y); % number of training examples
		J = 0; % You need to return the following variables correctly 
	J = (2*m)^-1 * sum(((theta' * X')' - y).^2);
end

pause;

fprintf('Running gradient descent ...\n');
alpha = 0.01; % Choose some alpha value - test different ones
num_iters = 400;

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1); % Initialize some useful values
	for iter = 1:num_iters
		theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)'; 
		J_history(iter) = computeCostMulti(X, y, theta); % Save the cost J in every iteration  
	end 
end

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = 0;
predict_values = zeros(1, (size(X,2) - 1));
predict_values = [1650 3];
price = [1 ((predict_values - mu)./sigma)] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Alternatively, use the normal equation:

fprintf('Solving with normal equations...\n');

data = csvread('ex1data2.txt'); %% Load Data
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

fprintf('Normalizing Features ...\n'); 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation is 1.
function [X_norm, mu, sigma] = featureNormalize(X)
	X_norm = X; % You need to set these values correctly
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2)); 
     
	mu = [mean(X(:, 1)) mean(X(:, 2))];
	sigma = [std(X(:, 1)) std(X(:, 2))];
	X_norm = (X - mu)./sigma;
	X = X_norm;
end

[X mu sigma] = featureNormalize(X); %run featureNormalize
X = [ones(m, 1) X]; % Add intercept term to X

function [theta] = normalEqn(X, y) %NORMALEQN Computes the closed-form solution to linear regression 
	theta = zeros(3, 1);
	theta = pinv(X'*X) * X' * y; 
end

theta = normalEqn(X, y); % Calculate the parameters from the normal equation

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house (with normal equation)
price = 0;
predict_values = zeros(1, (size(X,2) - 1));
predict_values = [1650 3];
price = [1 ((predict_values - mu)./sigma)] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

