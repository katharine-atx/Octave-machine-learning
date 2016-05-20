%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance

% Data notes: Regularized linear regression is used to predict the amount of water 
% owing out of a dam (y) using the change of water level in a reservoir (X). 
% The goal of this script is to practice some diagnostics of debugging learning 
% algorithms and examine the effects of bias v.s. variance.

% The dataset is divided into three parts:
% 1) A training set the model will learn on: X, y
% 2) A cross validation set for determining the regularization parameter: Xval, yval
% 3) A test set for evaluating performance: Xtest, ytest

clear ; close all; clc

% Load X, y, Xval, yval, Xtest, ytest to environment...
fprintf('Loading and Visualizing Data ...\n')
load ('ex5data1.mat');
m = size(X, 1); % m = Number of examples

% Functions...

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
	% computes the cost for linear regression. Returns the cost and gradient.
	% Initialize values...
	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));
	predictors = X * theta;
	% Linear cost function...
	J = (2*m)^-1 * sum((predictors - y).^2) + lambda/(2*m) * sum(theta(2:size(X,2)) .^2);
	% theta_0 (index = 1) is omitted from regularization...
	grad(1) = 1/m * ((predictors - y)' * X(:,1));
	grad(2:size(theta,1)) = 1/m * ((predictors - y)' * X(:,2:size(X,2)) + lambda * theta(2:size(theta,1))');
end

function [error_train, error_val] = ...
	learningCurve(X, y, Xval, yval, lambda)
	% returns the train and cross validation set errors for a learning curve.
	% Vectors error_train and error_val will be the same length. 
	% error_train(i) contains the training error for i examples and similarly 
	% for error_val(i)).
	% Initialize values...
	m = size(X, 1);
	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);
	for i = 1:m
		theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
		% setting lambda to zero...
		[error_train(i), train_grad] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
		[error_val(i), val_grad] = linearRegCostFunction(Xval, yval, theta, 0);
	end
end

function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
	% Minimizes a continuous differentialble multivariate function. 
	% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
	% Read options
	if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
		length = options.MaxIter;
	else
		length = 100;
	end
	RHO = 0.01;                            % a bunch of constants for line searches
	SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
	INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
	EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
	MAX = 20;                         % max 20 function evaluations per line search
	RATIO = 100;                                      % maximum allowed slope ratio
	argstr = ['feval(f, X'];                      % compose string used to call function
	for i = 1:(nargin - 3)
		argstr = [argstr, ',P', int2str(i)];
	end
	argstr = [argstr, ')'];
	if max(size(length)) == 2, 
		red=length(2); 
		length=length(1); 
	else 
		red=1; 
	end
	S=['Iteration '];
	i = 0;                                            % zero the run length counter
	ls_failed = 0;                             % no previous line search has failed
	fX = [];
	[f1 df1] = eval(argstr);                      % get function value and gradient
	i = i + (length<0);                                            % count epochs?!
	s = -df1;                                        % search direction is steepest
	d1 = -s'*s;                                                 % this is the slope
	z1 = red/(1-d1);                                  % initial step is red/(|s|+1)
	while i < abs(length)                                      % while not finished
		i = i + (length>0);                                      % count iterations?!
		X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
		X = X + z1*s;                                             % begin line search
		[f2 df2] = eval(argstr);
		i = i + (length<0);                                          % count epochs?!
		d2 = df2'*s;
		f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
		if length>0, 
			M = MAX; 
		else 
			M = min(MAX, -length-i); 
		end
		success = 0; limit = -1;                     % initialize quanteties
		while 1
			while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
				limit = z1;                                         % tighten the bracket
				if f2 > f1
					z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
				else
					A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
					B = 3*(f3-f2)-z3*(d3+2*d2);
					z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
				end
				if isnan(z2) || isinf(z2)
					z2 = z3/2;                  % if we had a numerical problem then bisect
				end
				  z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
				  z1 = z1 + z2;                                           % update the step
				  X = X + z2*s;
				  [f2 df2] = eval(argstr);
				  M = M - 1; i = i + (length<0);                           % count epochs?!
				  d2 = df2'*s;
				  z3 = z3-z2;                    % z3 is now relative to the location of z2
				end
				if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
				  break;                                                % this is a failure
				elseif d2 > SIG*d1
				  success = 1; break;                                             % success
				elseif M == 0
				  break;                                                          % failure
				end
				A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
				B = 3*(f3-f2)-z3*(d3+2*d2);
				z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
				if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
				  if limit < -0.5                               % if we have no upper limit
						z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
				  else
						z2 = (limit-z1)/2;                                   % otherwise bisect
				  end
				elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
				  z2 = (limit-z1)/2;                                               % bisect
				elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
				  z2 = z1*(EXT-1.0);                           % set to extrapolation limit
				elseif z2 < -z3*INT
				  z2 = -z3*INT;
				elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
				  z2 = (limit-z1)*(1.0-INT);
				end
				f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
				z1 = z1 + z2; X = X + z2*s;                      % update current estimates
				[f2 df2] = eval(argstr);
				M = M - 1; i = i + (length<0);                             % count epochs?!
				d2 = df2'*s;
		  end                                                      % end of line search
		if success                                         % if line search succeeded
			f1 = f2; fX = [fX' f1]';
			fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
			s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
			tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
			d2 = df1'*s;
			if d2 > 0                                      % new slope must be negative
				s = -df1;                              % otherwise use steepest direction
				d2 = -s'*s;    
			end
			z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
			d1 = d2;
			ls_failed = 0;                              % this line search did not fail
		else
			X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
			if ls_failed || i > abs(length)          % line search failed twice in a row
				break;                             % or we ran out of time, so we give up
			end
			tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
			s = -df1;                                                    % try steepest
			d1 = -s'*s;
			z1 = 1/(1-d1);                     
			ls_failed = 1;                                    % this line search failed
		end
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		end
		fprintf('\n');
	end
end

function [theta] = trainLinearReg(X, y, lambda)
	% trains linear regression using the dataset (X, y) and regularization 
	% parameter lambda. Returns trained parameters theta.
	% Initialize Theta...
	initial_theta = zeros(size(X, 2), 1); 
	% Create "short hand" for the cost function to be minimized
	costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
	% Now, costFunction is a function that takes in only one argument
	options = optimset('MaxIter', 200, 'GradObj', 'on');
	% Minimize using fmincg
	theta = fmincg(costFunction, initial_theta, options);
end


function [lambda_vec, error_train, error_val] = ...
	validationCurve(X, y, Xval, yval)
	% returns the train and validation errors (in error_train, error_val)
	% for different values of lambda. 
	% Initialize values...
	% Some selected values of lambda:
	lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
	error_train = zeros(length(lambda_vec), 1);
	error_val = zeros(length(lambda_vec), 1);
	for i = 1:length(lambda_vec)
		lambda = lambda_vec(i);
		theta = trainLinearReg(X, y, lambda);	
		[error_train(i), train_grad] = linearRegCostFunction(X, y, theta, 0);
		[error_val(i), val_grad] = linearRegCostFunction(Xval, yval, theta, 0);
	end
end

function [X_poly] = polyFeatures(X, p)
	% takes a data matrix X (size m x 1) and maps each example into its 
	% polynomial features where the p-th column of X contains the values 
	% of X to the p-th power.
	% i.e. X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ... X(i).^p]
	% Initialize values...
	X_poly = zeros(numel(X), p);
	for i = 1:p
		X_poly(:,i) = X(:,1).^i;
	end
end

function [X_norm, mu, sigma] = featureNormalize(X)
	% returns a normalized version of X where the mean value of each feature 
	% is 0 and the standard deviation is 1. 
	mu = mean(X);
	X_norm = bsxfun(@minus, X, mu);
	sigma = std(X_norm);
	X_norm = bsxfun(@rdivide, X_norm, sigma);
end

function plotFit(min_x, max_x, mu, sigma, theta, p)
	% plots the learned polynomial fit with power p and feature 
	% normalization (mu, sigma).
	% Hold on to the current figure...
	hold on;
	% Plot a range slightly bigger than the min and max values...
	x = (min_x - 15: 0.05 : max_x + 25)';
	% Map the X values 
	X_poly = polyFeatures(x, p);
	X_poly = bsxfun(@minus, X_poly, mu);
	X_poly = bsxfun(@rdivide, X_poly, sigma);
	% Add ones for bias terms...
	X_poly = [ones(size(x, 1), 1) X_poly];
	% Plot
	plot(x, X_poly * theta, '--', 'LineWidth', 2)
	% Hold off to the current figure
	hold off
end

% Script...

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Regularized cost function...
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Regression gradient...
theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));
fprintf('Program paused. Press enter to continue.\n');
pause;

% Training linear regression...
% Note: The data is non-linear, so this will be a poor fit.
%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;
fprintf('Program paused. Press enter to continue.\n');
pause;

% Learning curve to diagnose linear fit performance....
% Expect to see "high bias" results since data is non-linear (underfit).
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end
fprintf('Program paused. Press enter to continue.\n');
pause;

% One possible underfit remedy: add polynomial features...
p = 8;
% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones
% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones
fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Another potential remedy, experiment with decreasing lamba regularization parameter.
% Try out different values here...
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end
fprintf('Program paused. Press enter to continue.\n');
pause;

% Test performance of lambda selection...........
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);
close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end
fprintf('Program paused. Press enter to continue.\n');
pause;
