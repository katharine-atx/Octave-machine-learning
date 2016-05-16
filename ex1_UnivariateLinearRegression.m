% Exercise 1: Linear Regression
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s

% Initialization

clear ; close all; clc

%   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

function A = warmUpExercise()
	A = [];
	A = eye(5);
end

fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()
fprintf('Program paused. Press enter to continue.\n');
pause;

%% 	Plotting

fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

function plotData(x, y)
	figure; 
	% opens a new figure window
	plot (x,y, 'rx', 'MarkerSize', 10);
	ylabel ('Profit in $10,000s');
	xlabel ('Population of City in 10,000s');
end

plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Gradient descent

fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500; % Some gradient descent settings
alpha = 0.01;

%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

function J = computeCost(X, y, theta)
	m = length(y); % initialize useful values: number of training examples
	J = 0;  %You need to return this variable correctly 
	J = (2*m)^-1 * sum(((theta' * X')' - y).^2);
end

computeCost(X, y, theta); % compute and display initial cost


%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1); %initialize other values
	for iter = 1:num_iters
		theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)';
		J_history(iter) = computeCost(X, y, theta); % Save the cost J in every iteration 
	end
end

theta = gradientDescent(X, y, theta, alpha, iterations);

fprintf('Theta found by gradient descent: '); % print theta to screen
fprintf('%f %f \n', theta(1), theta(2));

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-') % Plot the linear fit
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

predict1 = [1, 3.5] *theta; % Predict values for population 35K and 70K
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

pause;

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals) % Surface plot
xlabel('\theta_0'); ylabel('\theta_1');

pause;

figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20)) % Contour plot
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
