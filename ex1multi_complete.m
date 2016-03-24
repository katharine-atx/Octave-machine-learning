%% Machine Learning Online Class - Exercise 1: Linear regression with multiple variables
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

%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples
J = 0; % You need to return the following variables correctly 
J = (2*m)^-1 * sum(((theta' * X')' - y).^2); %same as with one x vector?
end

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
% Hint: At prediction, make sure you do the same feature normalization.
%
fprintf('Running gradient descent ...\n');
alpha = 0.01; % Choose some alpha value - test different ones
num_iters = 400;

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % Initialize some useful values
for iter = 1:num_iters
	theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)'; %same as with one x vector?
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
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0;
price = theta * [1 ([1650 3]- mu)./sigma]; % You should change this

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.function [theta] = normalEqn(X, y)
theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

theta = (X'*X)^-1 * X'*y;

% -------------------------------------------------------------


% ============================================================

end

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
price = theta * [1 ([1650 3]- mu)./sigma];


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

