%  Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
% 
%  This script applies SVM with Gaussian kernel for serveral example datasets.

clear ; close all; clc

% Functions:

function plotData(X, y)
	% plots the data points with + for the positive examples
	% and o for the negative examples. X is assumed to be a Mx2 matrix.
	% Note: This was slightly modified such that it expects y = 1 or y = 0
	% Find Indices of Positive and Negative Examples...
	pos = find(y == 1); neg = find(y == 0);
	% Plot Examples
	plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
	hold on;
	plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
	hold off;
end

function visualizeBoundaryLinear(X, y, model)
	% plots a linear decision boundary learned by the SVM 
	% and overlays the data.
	w = model.w;
	b = model.b;
	xp = linspace(min(X(:,1)), max(X(:,1)), 100);
	yp = - (w(1)*xp + b)/w(2);
	plotData(X, y);
	hold on;
	plot(xp, yp, '-b'); 
	hold off
end

function visualizeBoundary(X, y, model, varargin)
	% plots a non-linear decision boundary learned by the SVM
	% and overlays the data
	plotData(X, y)
	% Make classification predictions over a grid of values
	x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
	x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
	[X1, X2] = meshgrid(x1plot, x2plot);
	vals = zeros(size(X1));
	for i = 1:size(X1, 2)
	   this_X = [X1(:, i), X2(:, i)];
	   vals(:, i) = svmPredict(model, this_X);
	end
	% Plot the SVM boundary
	hold on
	contour(X1, X2, vals ,[1 1], 'Color');
	hold off;
end

function sim = linearKernel(x1, x2)
	% returns a linear kernel between x1 and x2
	% and returns the value in sim
	% Ensure that x1 and x2 are column vectors
	x1 = x1(:); x2 = x2(:);
	% Compute the kernel
	sim = x1' * x2;  % dot product
end

function sim = gaussianKernel(x1, x2, sigma)
	% returns a gaussian kernel between x1 and x2 and returns the value in sim.
	% Ensuring that x1 and x2 are column vectors...
	x1 = x1(:); x2 = x2(:);
	% Initialize values...
	sim = 0;
	% gaussian kernel similarity forumla...
	sim = exp(-sum((x1 - x2).^2)/ (2 * sigma^2));
end

function [C, sigma] = dataset3Params(X, y, Xval, yval)
	% returns optimal choice of C and sigma to use for SVM with RBF kernel
	% based on a cross-validation set.
	% Initialize values...
	C = 1;
	sigma = 0.3;
	% We are looping over values of C and sigma and need to capture running 
	% error of each choice. Initialize at infinite...
	runningError = Inf;
	% Testing error across 8 possible values for C and sigma...
	for valueC = [0.01 0.03 0.1 0.3 1 3 10 30]
		for valueSigma = [0.01 0.03 0.1 0.3 1 3 10 30]
			model = svmTrain(X, y, valueC, @(x1, x2) gaussianKernel(x1, x2, valueSigma));
			predictions = svmPredict(model, Xval);
			meanError = mean(double(predictions ~= yval));
			if meanError < runningError
			  runningError = meanError;
			  C = valueC;
			  sigma = valueSigma;
			end
		end
	end
end

function [model] = svmTrain(X, Y, C, kernelFunction, ...
                            tol, max_passes)
	% trains an SVM classifier and returns trained model. X is the matrix 
	% of training examples. Each row is a training example, and the jth column 
	% holds the jth feature. Y is a column matrix containing 1 for positive 
	% examples and 0 for negative examples.  C is the standard SVM regularization 
	% parameter.  tol is a tolerance value used for determining equality of 
	% floating point numbers. max_passes controls the number of iterations
	% over the dataset (without changes to alpha) before the algorithm quits.
	% Note: This is a simplified version of the SMO algorithm for training SVMs. 
	if ~exist('tol', 'var') || isempty(tol)
		tol = 1e-3;
	end
	if ~exist('max_passes', 'var') || isempty(max_passes)
		max_passes = 5;
	end
	% Data parameters
	m = size(X, 1);
	n = size(X, 2);
	% Map 0 to -1
	Y(Y==0) = -1;
	% Variables
	alphas = zeros(m, 1);
	b = 0;
	E = zeros(m, 1);
	passes = 0;
	eta = 0;
	L = 0;
	H = 0;
	% Pre-compute the Kernel Matrix since our dataset is small
	% We have implemented optimized vectorized version of the Kernels...
	if strcmp(func2str(kernelFunction), 'linearKernel')
		% This is equivalent to computing the kernel on every pair of examples
		K = X*X';
	elseif strfind(func2str(kernelFunction), 'gaussianKernel')
		% This is equivalent to computing the kernel on every pair of examples
		X2 = sum(X.^2, 2);
		K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
		K = kernelFunction(1, 0) .^ K;
	else
		% Pre-compute the Kernel Matrix
		K = zeros(m);
		for i = 1:m
			for j = i:m
				 K(i,j) = kernelFunction(X(i,:)', X(j,:)');
				 K(j,i) = K(i,j); %the matrix is symmetric
			end
		end
	end
	% Train
	fprintf('\nTraining ...');
	dots = 12;
	while passes < max_passes,
		num_changed_alphas = 0;
		for i = 1:m,
			% Calculate Ei = f(x(i)) - y(i) using (2). 
			% E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
			E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
			if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0)),
				%In this simplified code, we select i and j randomly.
				j = ceil(m * rand());
				while j == i,  % Make sure i \neq j
					j = ceil(m * rand());
				end
				% Calculate Ej = f(x(j)) - y(j) using (2).
				E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);
				% Save old alphas
				alpha_i_old = alphas(i);
				alpha_j_old = alphas(j);
				% Compute L and H by (10) or (11). 
				if (Y(i) == Y(j)),
					L = max(0, alphas(j) + alphas(i) - C);
					H = min(C, alphas(j) + alphas(i));
				else
					L = max(0, alphas(j) - alphas(i));
					H = min(C, C + alphas(j) - alphas(i));
				end
				if (L == H),
					% continue to next i. 
					continue;
				end
				% Compute eta by (14).
				eta = 2 * K(i,j) - K(i,i) - K(j,j);
				if (eta >= 0),
					% continue to next i. 
					continue;
				end
				% Compute and clip new value for alpha j using (12) and (15).
				alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
				% Clip
				alphas(j) = min (H, alphas(j));
				alphas(j) = max (L, alphas(j));
				% Check if change in alpha is significant
				if (abs(alphas(j) - alpha_j_old) < tol),
					% continue to next i. 
					% replace anyway
					alphas(j) = alpha_j_old;
					continue;
				end
				% Determine value for alpha i using (16). 
				alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
				% Compute b1 and b2 using (17) and (18) respectively. 
				b1 = b - E(i) ...
					 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
					 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
				b2 = b - E(j) ...
					 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
					 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
				% Compute b by (19). 
				if (0 < alphas(i) && alphas(i) < C),
					b = b1;
				elseif (0 < alphas(j) && alphas(j) < C),
					b = b2;
				else
					b = (b1+b2)/2;
				end
				num_changed_alphas = num_changed_alphas + 1;
			end
		end
		if (num_changed_alphas == 0),
			passes = passes + 1;
		else
			passes = 0;
		end
		fprintf('.');
		dots = dots + 1;
		if dots > 78
			dots = 0;
			fprintf('\n');
		end
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		end
	end
	fprintf(' Done! \n\n');
	% Save the model
	idx = alphas > 0;
	model.X= X(idx,:);
	model.y= Y(idx);
	model.kernelFunction = kernelFunction;
	model.b= b;
	model.alphas= alphas(idx);
	model.w = ((alphas.*Y)'*X)';
end


function pred = svmPredict(model, X)
	% returns a vector of predictions using a trained SVM model (svmTrain). 
	% X is a mxn matrix where there each example is a row. model is a svm 
	% model returned from svmTrain. predictions pred is a m x 1 column of 
	% predictions of {0, 1} values.
	% Check if we are getting a column vector, if so, then assume that we only
	% need to do prediction for a single example
	if (size(X, 2) == 1)
		% Examples should be in rows
		X = X';
	end
	% Dataset 
	m = size(X, 1);
	p = zeros(m, 1);
	pred = zeros(m, 1);
	if strcmp(func2str(model.kernelFunction), 'linearKernel')
		% We can use the weights and bias directly if working with the 
		% linear kernel
		p = X * model.w + model.b;
	elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
		% Vectorized RBF Kernel
		% This is equivalent to computing the kernel on every pair of examples
		X1 = sum(X.^2, 2);
		X2 = sum(model.X.^2, 2)';
		K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
		K = model.kernelFunction(1, 0) .^ K;
		K = bsxfun(@times, model.y', K);
		K = bsxfun(@times, model.alphas', K);
		p = sum(K, 2);
	else
		% Other Non-linear kernel
		for i = 1:m
			prediction = 0;
			for j = 1:size(model.X, 1)
				prediction = prediction + ...
					model.alphas(j) * model.y(j) * ...
					model.kernelFunction(X(i,:)', model.X(j,:)');
			end
			p(i) = prediction + model.b;
		end
	end
	% Convert predictions into 0 / 1
	pred(p >= 0) =  1;
	pred(p <  0) =  0;
end

% Script...

fprintf('Loading and Visualizing Data ...\n')
% Load from ex6data1: X, y
load('ex6data1.mat');
% Plot training data
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nTraining Linear SVM ...\n')
% Load from ex6data1: X, y
load('ex6data1.mat');
% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Now for the next dataset...
fprintf('Loading and Visualizing Data ...\n')
% Load from ex6data2: X, y
load('ex6data2.mat');
% Plot training data
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
% Load from ex6data2: X, y
load('ex6data2.mat');
% SVM Parameters
C = 1; sigma = 0.1;

% Setting tolerance and max_passes lower here for a faster run for testing...
% Note: In practice, run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;

% A third dataset for training...
fprintf('Loading and Visualizing Data ...\n')
% Load from ex6data3: X, y
load('ex6data3.mat');
% Plot training data
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Load from ex6data3: X,y
load('ex6data3.mat');
% Note: Try different SVM Parameters here...
[C, sigma] = dataset3Params(X, y, Xval, yval);
% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;



