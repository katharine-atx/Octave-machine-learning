%% Machine Learning Online Class - Exercise 4 Neural Network Learning
% Training a neural network to recognize hand-written digits.

% Data notes: Source data contains 5000 training examples of handwritten digits. 
% These matrices can be read directly using the load command. There are 5000 
% training examples, where each training example is a 20 pixel by 20 pixel 
% grayscale image of the digit. Each pixel is represented by a floating point 
% number indicating the grayscale intensity. The 20 by 20 grid of pixels is 
% "unrolled" into a 400-dimensional vector as a row of matrix X. A 
% 5000-dimensional vector y contains labels for the training set. 
% Note: to avoid a zero index, the digit zero has been mapped to the value ten. 
% Neural network weights: Theta1 (dim 25 x 401), Theta2 (dim 10 X 26)

clear ; close all; clc

%% Setting up parameters...
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10; 
lambda = 0; % starting with unregularized run...         

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];
 
% Functions....

function [h, display_array] = displayData(X, example_width)
	% Displays 2D data in a grid. Returnss the figure handle h and the 
	% displayed array if requested.
	% Set example_width automatically if not passed in
	if ~exist('example_width', 'var') || isempty(example_width) 
		example_width = round(sqrt(size(X, 2)));
	end
	% Gray Image
	colormap(gray);
	% Compute rows, cols
	[m n] = size(X);
	example_height = (n / example_width);
	% Compute number of items to display
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);
	% Between images padding
	pad = 1;
	% Setup blank display
	display_array = - ones(pad + display_rows * (example_height + pad), ...
						   pad + display_cols * (example_width + pad));
	% Copy each example into a patch on the display array
	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m, 
				break; 
			end
			% Copy the patch
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

function g = sigmoidGradient(z)
	% computes the gradient of the sigmoid function at z. z is a matrix or a
	% vector and the function returns the gradient for each element.
	g = zeros(size(z));
	g = sigmoid(z) .* (1 - sigmoid(z));
end

function [J grad] = nnCostFunction(nn_params, ...
	input_layer_size, ...
	hidden_layer_size, ...
	num_labels, ...
	X, y, lambda)
	% Implements the neural network cost function for a two layer neural network 
	% which performs classification.
	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	% for the 2 layer neural network...
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
					 num_labels, (hidden_layer_size + 1));
	% Initialize values...
	m = size(X, 1);
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));
	% Vectorized forward propagation over all training examples...
	% Add bias units to a1, a2 with ones()...
	a1 = [ones(m, 1) X];
	z2 = a1 * Theta1';
	a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
	predictors = a3;
	% Reformat y from digit label to vector with zeros and 1 in the index position of label value.
	% Using eye() identity matrix function to create y 0|1 vectors...
	identity = eye(num_labels);
	vectY = zeros(m, num_labels);
	for i = 1:m
	  vectY(i, :)= identity(y(i), :);
	end
	% calculate J and the regularization cost (reg)...
	reg = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
	J = (1/m) * sum(sum(-vectY .* log(predictors) - (1 - vectY) .* log(1 - predictors), 2))+ lambda/(2*m) * reg;
	% Computing gradients for Theta1, Theta2 and accumulating these to D terms....
	delta3 = predictors - vectY;
	delta2 = delta3 * Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]);
	% Trim bias column...
	delta2 = delta2(:, 2:end);
	D1 = delta2' * a1;
	D2 = delta3' * a2;
	% regularization expression gradients...
	deltaReg1 = lambda/m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
	deltaReg2 = lambda/m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
	% Add regularization gradients to D terms...
	Theta1_grad = D1./m + deltaReg1;
	Theta2_grad = D2./m + deltaReg2;
	% Unroll gradients into a single vector...
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
end		

function W = randInitializeWeights(L_in, L_out)
	%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
	%incoming connections and L_out outgoing connections
	W = zeros(L_out, 1 + L_in);
	epsilon_init = 0.12;
	W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end

function W = debugInitializeWeights(fan_out, fan_in)
	% Initializes the weights of a layer with fan_in incoming connections and 
	% fan_out outgoing connections using a fixed set of values.
	% Set W to zeros and add bias column...
	W = zeros(fan_out, 1 + fan_in);
	% Initialize W using "sin". Note: this ensures that W is always of the same
	% values for debugging.
	W = reshape(sin(1:numel(W)), size(W)) / 10;
end

function numgrad = computeNumericalGradient(J, theta)
	% Computes the numerical gradient of the function J around theta. Calling 
	% y = J(theta) will return the function value at theta.         
	numgrad = zeros(size(theta));
	perturb = zeros(size(theta));
	e = 1e-4;
	for p = 1:numel(theta)
		% Set perturbation vector
		perturb(p) = e;
		loss1 = J(theta - perturb);
		loss2 = J(theta + perturb);
		% Compute Numerical Gradient
		numgrad(p) = (loss2 - loss1)./(2*e);
		perturb(p) = 0;
	end
end

function checkNNGradients(lambda)
	% Creates a small neural network to check the backpropagation gradients.
	% It outputs the analytical gradients produced by backprop code along with
	% the numerical gradients (from computeNumericalGradient). 
	if ~exist('lambda', 'var') || isempty(lambda)
		lambda = 0;
	end
	input_layer_size = 3;
	hidden_layer_size = 5;
	num_labels = 3;
	m = 5;
	% Generating some 'random' test data...
	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
	% Reusing debugInitializeWeights to generate X...
	X  = debugInitializeWeights(m, input_layer_size - 1);
	y  = 1 + mod(1:m, num_labels)';
	% Unroll parameters...
	nn_params = [Theta1(:) ; Theta2(:)];
	% Short hand for cost function
	costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
								   num_labels, X, y, lambda);
	[cost, grad] = costFunc(nn_params);
	numgrad = computeNumericalGradient(costFunc, nn_params);
	% Debugging: The two columns output should be very similar. 
	disp([numgrad grad]);
	fprintf(['The above two columns you get should be very similar.\n' ...
			 '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
	% Debugging: assuming EPSILON = 0.0001 in computeNumericalGradient,
	% difference should be less than 1e-9.
	diff = norm(numgrad-grad)/norm(numgrad+grad);
	fprintf(['If your backpropagation implementation is correct, then \n' ...
			 'the relative difference will be small (less than 1e-9). \n' ...
			 '\nRelative Difference: %g\n'], diff);
end

function p = predict(Theta1, Theta2, X)
	% Outputs the predicted label of X given the trained weights of a neural 
	% network (Theta1, Theta2)
	% Initialize values...
	m = size(X, 1);
	num_labels = size(Theta2, 1);
	p = zeros(size(X, 1), 1);
	% Hidden layer and output layer (h1, h2)...
	h1 = sigmoid([ones(m, 1) X] * Theta1');
	h2 = sigmoid([ones(m, 1) h1] * Theta2');
	% Prediction is label with max probability...
	[dummy, p] = max(h2, [], 2);
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

% Script...

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
pause;

% Forward propagation (lambda = 0)...
fprintf('\nFeedforward Using Neural Network ...\n')
% Weight regularization parameter (we set this to 0 here).
lambda = 0;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Adding regularization (lambda = 1)....
fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Evaluate sigmoid gradient for debugging...
fprintf('\nEvaluating sigmoid gradient...\n')
g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Initialize the weights of the neural network....
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters...
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Back propagation...
fprintf('\nChecking Backpropagation... \n');
%  Check gradients by running checkNNGradients
checkNNGradients;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Adding regularization (lambda = 3)....
fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);
fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Train the neural network with optimization algorithm...
fprintf('\nTraining Neural Network... \n')
% Optionally increase max iterations...
options = optimset('MaxIter', 50);
% You can also try different values of lambda...
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.\n');
pause;

% Visualize weights...
fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Predictions from neural network for digit labels...
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


